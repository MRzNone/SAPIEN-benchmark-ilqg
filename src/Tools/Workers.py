import sapien.core as sapien
from sapien.core import Pose
import jax.numpy as np
import jax
import numpy as onp
import time
from multiprocessing import Queue, Process, Array, Value
import ctypes as c
from . import misc, ModelDerivator


class Worker(Process):
    def __init__(self, request_queue, ans_arr, num_task, timestep, workId, create_scene, debug):
        super(Worker, self).__init__()
        self.request_queue = request_queue
        self.ans_arr = ans_arr
        self.num_task = num_task
        self.workId = workId
        self.debug = debug

        # init scene
        self.scene, self.robot = create_scene(timestep, False)
        self.dof = self.robot.dof

    def get_state(self):
        return misc.get_state(self.robot)

    def fu(self, ini_pack, u1, u2, index):
        # simulate
        self.robot.unpack(ini_pack)
        self.robot.set_qf(u1)
        self.scene.step()
        x1 = self.get_state()

        self.robot.unpack(ini_pack)
        # simulate
        self.robot.set_qf(u2)
        self.scene.step()
        x2 = self.get_state()

        if self.debug and (misc.check_val(x1) or misc.check_val(x2)):
            print("\nFU")
            print(f"U1: {u1}")
            print(f"U2: {u2}")
            print(f"X1: {x1}")
            print(f"X2: {x2}")
            raise Exception("FU invalid")

        self.ans_arr[index] = x2 - x1

    def fx(self, ini_pack, u, state, new_val1, new_val2, index):
        self.robot.unpack(ini_pack)
        if state == 'qpos':
            self.robot.set_qpos(new_val1)
        elif state == 'robo_pos':
            p, q = new_val1
            self.robot.set_pose(Pose(p, q))
        # simulate
        self.robot.set_qf(u)
        self.scene.step()
        x1 = self.get_state()

        self.robot.unpack(ini_pack)
        if state == 'qpos':
            self.robot.set_qpos(new_val2)
        elif state == 'robo_pos':
            p, q = new_val2
            self.robot.set_pose(Pose(p, q))
        # simulate
        self.robot.set_qf(u)
        self.scene.step()
        x2 = self.get_state()

        if self.debug and (misc.check_val(x1) or misc.check_val(x2)):
            print("\nFX")
            print(f"U : {u}")
            print(f"S : {state}")
            print(f"I1: {new_val1}")
            print(f"I2: {new_val2}")
            print(f"X1: {x1}")
            print(f"X2: {x2}")
            raise Exception("FX invalid")

        self.ans_arr[index] = x2 - x1

    def sim(self, u):
        self.robot.set_qf(u)
        self.scene.step()

        state = self.get_state()

        if self.debug and misc.check_val(state):
            print("\nSIM")
            print(f"U : {u}")
            print(f"state: {state}")
            raise Exception("SIM invalid")

        self.ans_arr[:len(state)] = state

    def robo_set(self, pack):
        self.robot.unpack(pack)

    def robo_get(self):
        pack = self.robot.pack()
        self.ans_arr[:len(pack)] = pack

    def run(self):
        print(f'Worker-{self.workId} started')

        for task_arg, index, workType in iter(self.request_queue.get, None):
            if workType == 'fu':
                ini_pack, u1, u2 = task_arg
                self.fu(ini_pack, u1, u2, index)
            elif workType == 'fx':
                ini_pack, u, state, new_val1, new_val2 = task_arg
                self.fx(ini_pack, u, state, new_val1, new_val2, index)
            elif workType == 'sim':
                u = task_arg
                self.sim(u)
            elif workType == 'unpack':
                pack = task_arg
                self.robo_set(pack)
            elif workType == 'pack':
                self.robo_get()

            with self.num_task.get_lock():
                self.num_task.value -= 1

        print(f'Worker-{self.workId} exits')


def create_workers(num_workers, m, n, time_step, create_scene, debug):
    request_queue = Queue()
    mp_arr = Array(c.c_double, m * n)
    ans_arr = onp.frombuffer(mp_arr.get_obj())
    if m > 1:
        ans_arr = ans_arr.reshape(m, n)
    else:
        ans_arr = ans_arr.reshape(n, )
    num_task = Value('i', 1)

    # create worker
    woks = []
    for i in range(num_workers):
        wok = Worker(request_queue, ans_arr, num_task, time_step, i, create_scene, debug)
        wok.start()
        woks.append(wok)

    return request_queue, ans_arr, num_task, woks


class SimWorker:
    def __init__(self, robot: sapien.Articulation, create_scene, time_step, debug):
        self.num_x = len(misc.get_state(robot))
        self.pack = robot.pack()
        self.num_pack = len(self.pack)

        bigger_dim = max(self.num_x, self.num_pack)
        self.request_queue, self.ans_arr, self.num_task, self.woks = \
            create_workers(1, 1, bigger_dim, time_step, create_scene, debug)

        self.reset()

    def terminate(self):
        for wok in self.woks:
            self.request_queue.put(None)
            wok.terminate()
            self.request_queue.close()
        print("Workers terminated")

    def flush_work(self):
        while not self.request_queue.empty():
            self.request_queue.get()

        with self.num_task.get_lock():
            self.num_task.value = 0

    def get_pack(self):
        self.flush_work()

        with self.num_task.get_lock():
            self.num_task.value = 1

        wok_args = (None, None, 'pack')
        self.request_queue.put(wok_args)

        # block until done
        while self.num_task.value > 0 and self.request_queue.qsize() > 0:
            time.sleep(0.00001)

        pack = self.ans_arr[:self.num_pack]
        return pack

    def checkpoint(self):
        self.pack = self.get_pack()

    def reset(self):
        self.set(self.pack)

    def set(self, pack):
        self.flush_work()
        with self.num_task.get_lock():
            self.num_task.value = 1

        wok_args = (pack, None, 'unpack')
        self.pack = pack
        self.request_queue.put(wok_args)

        # block until done
        while self.num_task.value > 0 and self.request_queue.qsize() > 0:
            time.sleep(0.00001)

    def sim(self, u):
        self.flush_work()
        with self.num_task.get_lock():
            self.num_task.value = 1

        wok_args = (u, None, 'sim')
        self.request_queue.put(wok_args)

        # block until done
        while self.num_task.value > 0 and self.request_queue.qsize() > 0:
            time.sleep(0.00001)

        state = self.ans_arr[:self.num_x]
        return state


class DerivativeFactory(ModelDerivator):
    def __init__(self, num_x, num_u, dof, create_scene, num_workers, time_step, debug):
        self.num_u = num_u
        self.num_x = num_x
        self.dof = dof
        self.iniPack = None
        self.num_workers = num_workers

        # create workers
        bigger_dim = max(self.num_u, self.num_x)
        self.request_queue, self.ans_arr, self.num_task, self.woks = \
            create_workers(num_workers, bigger_dim, bigger_dim, time_step, create_scene, debug)

    def set_pack(self, pack):
        self.iniPack = pack

    def flush_work(self):
        while not self.request_queue.empty():
            self.request_queue.get()

        with self.num_task.get_lock():
            self.num_task.value = 0

    def terminate(self):
        for wok in self.woks:
            self.request_queue.put(None)
            wok.terminate()
            self.request_queue.close()
        print("Workers terminated")

    def fu(self, x, u, eps=1e-3):
        '''
            Only doing pos now, return (n_x, n_u)
        '''
        self.flush_work()

        ini_pack = self.iniPack

        #     passive_force = robot.compute_passive_force()
        u = onp.array(u)

        with self.num_task.get_lock():
            self.num_task.value = self.num_u

        # dispatch work
        for i in range(self.num_u):
            # prep args
            new_u2 = u.copy()
            new_u2[i] += eps

            # prep args
            new_u1 = u.copy()
            new_u1[i] -= eps

            ini_pack = ini_pack
            task_arg = (ini_pack, new_u1, new_u2)

            wok_args = (task_arg, i, 'fu')
            self.request_queue.put(wok_args)

        # block until done
        while self.num_task.value > 0 and self.request_queue.qsize() > 0:
            time.sleep(0.00001)

        res = self.ans_arr[:self.num_u] / (2 * eps)
        res = res.T

        return res

    def fx(self, x, u, eps=1e-3):
        '''
            Only doing pos now, return (n_x, n_x)
        '''
        return np.array(onp.eye(self.num_x))

        #
        # self.flush_work()
        #
        # ini_pack = self.iniPack
        #
        # u = u.tolist()
        #
        # with self.num_task.get_lock():
        #     self.num_task.value = self.num_x
        #
        # for i in range(self.num_x):
        #     # prep args
        #     state = None
        #     new_val1 = None
        #     new_val2 = None
        #
        #     # ini_pack, u, state, new_val, index
        #     if i < self.dof:
        #         state = 'q_pos'
        #         new_val2 = x[:self.dof].tolist()
        #         new_val2[i] += eps
        #
        #         new_val1 = x[:self.dof].tolist()
        #         new_val1[i] += eps
        #     else:
        #         state = 'robo_pos'
        #         p = x[self.dof: self.dof + 3].tolist()
        #         q = x[-4:].tolist()
        #
        #         j = i - self.dof
        #
        #         if j < 3:
        #             p[j] += eps
        #         else:
        #             j -= 3
        #             q[j] += eps
        #
        #         new_val2 = (p, q)
        #
        #         p = x[self.dof: self.dof + 3].tolist()
        #         q = x[-4:].tolist()
        #
        #         j = i - self.dof
        #
        #         if j < 3:
        #             p[j] -= eps
        #         else:
        #             j -= 3
        #             q[j] -= eps
        #
        #         new_val1 = (p, q)
        #
        #     task_args = (ini_pack, u, state, new_val1, new_val2)
        #     wok_args = (task_args, i, 'fx')
        #
        #     self.request_queue.put(wok_args)
        #
        # while self.num_task.value > 0 and self.request_queue.qsize() > 0:
        #     time.sleep(0.00001)
        #
        # res = self.ans_arr[:self.num_x] / (2 * eps)
        # res = np.array(res).T
        #
        # return res


class ModelSim:
    def __init__(self, create_scene, timestep):
        self.scene, self.robot = create_scene(timestep, False)

    def set(self, pack):
        self.robot.unpack(pack)

    def sim(self, u):
        self.robot.set_qf(u)
        self.scene.step()

        return misc.get_state(self.robot)

    def get_pack(self):
        return self.robot.pack()


class DerivativeWorker(ModelDerivator):
    def __init__(self, num_x, num_u, create_scene, time_step):
        self.num_u = num_u
        self.num_x = num_x
        self.iniPack = None
        self.s, self.robot = create_scene(time_step, False)

    def set_pack(self, pack):
        self.iniPack = pack

    def terminate(self):
        pass

    def fu(self, x, u, eps=1e-3):
        '''
            Only doing pos now, return (n_x, n_u)
        '''
        ini_pack = self.iniPack

        #     passive_force = robot.compute_passive_force()
        u = onp.array(u)

        res = []

        # dispatch work
        for i in range(self.num_u):
            self.robot.unpack(ini_pack)
            new_u2 = u.copy()
            new_u2[i] += eps
            self.robot.set_qf(new_u2)
            self.s.step()
            x2 = misc.get_state(self.robot)

            self.robot.unpack(ini_pack)
            new_u1 = u.copy()
            new_u1[i] -= eps
            self.robot.set_qf(new_u1)
            self.s.step()
            x1 = misc.get_state(self.robot)

            res.append((x2 - x1) / (2 * eps))

        res = np.array(onp.array(res)).T

        return res

    def fx(self, x, u, eps=1e-3):
        '''
            Only doing pos now, return (n_x, n_x)
        '''
        return np.array(onp.eye(self.num_x))
