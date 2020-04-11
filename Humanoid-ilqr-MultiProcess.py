#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sapien.core as sapien
from sapien.core import Pose
import jax.numpy as np
import numpy as onp
from jax import grad, jacfwd, jacrev, random, jit
import jax
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from multiprocessing import Queue, Process, Array, Value
import ctypes as c
# import threading

# get_ipython().run_line_magic('matplotlib', 'notebook')


# # Construct Scene

# In[2]:


sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

stabled = False 

def create_scene(timestep, visual):
    s = sim.create_scene()
    s.add_ground(-2)
    s.set_timestep(timestep)

    loader = s.create_urdf_loader()
    loader.fix_root_link = 0
    if visual:
        loader.collision_is_visual = True
        s.set_ambient_light([0.5, 0.5, 0.5])
        s.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
    robot = loader.load("humanoid.urdf")
    
    return s, robot

sim_timestep = 1/30
optim_timestep = 1/20
s0, robot0 = create_scene(sim_timestep, True)
    
render_controller.set_camera_position(-5, 0, 0)
render_controller.set_current_scene(s0)


# # Wrap IO
# 
# ## State
#     - COM pos (x y)
#         - 13 * 2
#     - foot pos (x y z)
#         - additional 1
#     - torso pos (x y z)
#         - additional 1
#     - COM velo?
#         - additional 13 * 3
# ## Action
#     - 21 params

# In[3]:


links_cache = {}

def get_links(robot):
    '''
        Return all the link that have significant mass for the robot
        
        target_body_parts : [str body parts]
    '''
    if robot in links_cache:
        return links_cache[robot]
    
    links = {}
    
    for l in robot.get_links():
        name = l.get_name()
        mass = l.get_mass()
        if mass > 1:
            links[name] = l
    
    links_cache[robot] = links
    return links

def get_target_link(links, target_body_names):
    target_body_parts = {}
    
    for name, l in links.items():
        if name in target_body_names:
            target_body_parts[name] = l
    
    return target_body_parts

def report_link(l):
    '''
        Report name, mass, velocity, and position of the link.
    '''
    name = l.get_name()
    mass = l.get_mass()
    velo = l.get_velocity()
    pos = l.get_pose().p
    print(f"{name}\n\tmass: {mass:.6f}\n\tvelo: {velo}\n\tpos:  {pos}\n")

def report_all_links(links):
    '''
        Repot info for all links.
    '''
    for _, l in links.items():
        report_link(l)
        
def get_mass(links):
    '''
        Input:
            links: the links dict
        
        Output:
            array of masses for each link
    '''
    mass = []
    for name, l in links.items():
        mass.append(l.get_mass())
    return np.array(mass)

def get_pos(links):
    '''
        Input:
            links: the links dict (n_l # links)
            
        Output:
            array of shape (n_l, 3)
    '''
    pos = []
#     i = 0
    for name, l in links.items():
        pos.append(l.get_pose().p)
        
#         if name in target_body_parts:
#             print(f'{name}-{i}:   {l.get_pose().p}')
#         i += 1
    
    return np.array(pos)

def get_state(robot):
    # Only using position now, might use velocity for COM VELO constraint
    state = get_pos(get_links(robot)).flatten()
    return state


# In[4]:


# the derivaties
# def num_fx(u, scene, robot, links=None, eps=1e-6):
#     '''
#         Only doing pos now, return (n_x, n_x)
#     '''
#     # TODO: get ini_pack from base scene, for multi-processing
    
#     if links == None:
#         links = get_links(robot)
    
#     ini_pack = robot.pack()
#     ini_state = get_state(robot)
    
#     res = []
    
#     for _, l in links.items():
#         for i in range(3):
#             robot.unpack(ini_pack)
#             scene.step()
            
#             # modify 
#             orig_p = l.get_pose()
#             pos = orig_p.p
#             pos[i] += eps
#             orig_p.set_p(pos)
            
#             # simulate
#             robot.set_qf(u)
#             scene.step()
            
#             new_state = get_state(robot) 
#             res.append((new_state - ini_state) / eps)
    
            
#     robot.unpack(ini_pack)
            
#     return np.array(res).T
    
# def num_fu(u, scene, robot, links=None, eps=1e-6):
#     '''
#         Only doing pos now, return (n_x, n_u)
#     '''
#     # TODO: get ini_state from base scene, for multi-processing
    
#     if links == None:
#         links = get_links(robot)
    
#     ini_pack = robot.pack()
#     ini_state = get_state(robot)
    
#     res = []
#     passive_force = robot.compute_passive_force()
#     u = (u + passive_force).tolist()
    
#     for i in range(robot.dof):
#         robot.unpack(ini_pack)
        
#         # modify
#         new_u = u.copy()
#         new_u[i] += eps
        
#         # simulate
#         robot.set_qf(new_u)
#         scene.step()
#         new_state = get_state(robot)
#         res.append((new_state - ini_state) / eps)
        
#     robot.unpack(ini_pack)
    
#     return np.array(res).T
        


# In[5]:


class Worker(Process):
    def __init__(self, request_queue, ans_arr, num_task, timestep, workId, workType):
        super(Worker, self).__init__()
        self.request_queue = request_queue
        self.ans_arr = ans_arr
        self.num_task = num_task
        self.workId = workId
        self.workType = workType
        
        # init scene
        self.scene, self.robot = create_scene(timestep, False)
        
        self.links = self.get_links()
    
    def get_links(self):
        '''
            Return all the link that have significant mass for the robot

            target_body_parts : [str body parts]
        '''
        links = {}

        for l in self.robot.get_links():
            name = l.get_name()
            mass = l.get_mass()
            if mass > 1:
                links[name] = l

        return links    
    
    def get_state(self):
        new_state = []
        for name, l in self.links.items():
            p = l.get_pose().p
            new_state.append(p[0])
            new_state.append(p[1])
            new_state.append(p[2])
        
        return new_state
    
    def fu(self, ini_pack, u, index):
        self.robot.unpack(ini_pack)

        # simulate
        self.robot.set_qf(u)

        self.scene.step()
        new_state = self.get_state()

        self.ans_arr[index] = new_state
        
    def fx(self, ini_pack, u, l_name, i, new_pos, index):
        self.robot.unpack(ini_pack)
        
        link = self.links[l_name]
        
        # modify 
        orig_p = link.get_pose()
        orig_p.set_p(new_pos)

        # simulate
        self.robot.set_qf(u)
        self.scene.step()

        self.ans_arr[index] = self.get_state()
    
    '''
        Args:
            - task_arg
            - i
            - workType
    '''
    def run(self):
        print(f'Worker({self.workType})-{self.workId} started')
        
        for task_arg, index in iter(self.request_queue.get, None):
            if self.workType == 'fu':
                ini_pack, new_u = task_arg
                self.fu(ini_pack, new_u, index)
            elif self.workType == 'fx':
                ini_pack, u, l_name, i, new_pos = task_arg
                self.fx(ini_pack, u, l_name, i, new_pos, index)
                
            with self.num_task.get_lock():
                self.num_task.value -= 1
                
        print(f'Worker-{self.workId} exits')


# In[6]:


def create_workers(workType, num_workers, m, n, optim_timestep):
    request_queue = Queue()
    mp_arr = Array(c.c_double, m * n)
    ans_arr = onp.frombuffer(mp_arr.get_obj())
    ans_arr = ans_arr.reshape(m, n)
    num_task = Value('i', 1)
    
    # create worker
    for i in range(num_workers):
        wok = Worker(request_queue, ans_arr, num_task, optim_timestep, i, workType)
        wok.start()
        
    return request_queue, ans_arr, num_task

def mp_num_fu(request_queue, ans_arr, u, robot, num_task, eps=1e-6):
    '''
        Only doing pos now, return (n_x, n_u)
    '''
    
    ini_pack = robot.pack()
    ini_state = get_state(robot)
    
    res = []
    passive_force = robot.compute_passive_force()
    u = (u + passive_force).tolist()

    with num_task.get_lock():
        num_task.value = robot.dof

    # dispatch work
    for i in range(robot.dof):
        # prep args
        new_u = u.copy()
        new_u[i] += eps
        
        ini_pack = ini_pack
        task_arg = (ini_pack, new_u)

        wok_args = (task_arg, i)
        request_queue.put(wok_args)

    # block until done
    while num_task.value > 0:
        time.sleep(0.001)
    
    res = (ans_arr - ini_state) / eps
    
    return res.T

def mp_num_fx(request_queue, ans_arr, u, robot, num_task, eps=1e-6):
    '''
        Only doing pos now, return (n_x, n_x)
    '''
    ini_pack = robot.pack()
    ini_state = get_state(robot)
    
    links = get_links(robot)
    
    with num_task.get_lock():
        num_task.value = len(links) * 3
    
    index = 0
    for l_name, l in links.items():
        orig_p = l.get_pose()
        pos = orig_p.p
        for i in range(3):
            # prep args
            new_pos = pos.copy()
            new_pos[i] += eps
            
            task_args = (ini_pack, u, l_name, i, new_pos)
            wok_args = (task_args, index)
            index += 1
            request_queue.put(wok_args)
        
    while num_task.value > 0:
        time.sleep(0.001)
        
    res = (ans_arr - ini_state) / eps
            
    return np.array(res).T


# In[7]:


# n_state = get_state(robot0).shape[0]
# n_dof = robot0.dof

# fu_request_queue, fu_ans_arr, fu_num_task = create_workers('fu', 2, n_dof, n_state)
# fx_request_queue, fx_ans_arr, fx_num_task = create_workers('fx', 4, n_state, n_state)

# arr = mp_num_fu(fu_request_queue, fu_ans_arr, [0]*n_dof, robot0, fu_num_task, eps=1e-3)
# arr = mp_num_fx(fx_request_queue, fx_ans_arr, [0]*n_dof, robot0, fx_num_task, eps=1e-3)


# # ILQR

# In[8]:


class ILQR:
    def __init__(self, final_cost, running_cost, model, u_range, horizon, per_iter, model_der=None):
        '''
            final_cost:     v(x)    ->  cost, float
            running_cost:   l(x, u) ->  cost, float
            model:          f(x, u) ->  new state, [n_x]
        '''
        self.f = model
        self.v = final_cost
        self.l = running_cost

        self.u_range = u_range
        self.horizon = horizon
        self.per_iter = per_iter

        # specify derivatives
        self.l_x = grad(self.l, 0)
        self.l_u = grad(self.l, 1)
        self.l_xx = jacfwd(self.l_x, 0)
        self.l_uu = jacfwd(self.l_u, 1)
        self.l_ux = jacrev(self.l_u, 0)

        self.v_x = grad(self.v)
        self.v_xx = jacfwd(self.v_x)

        if model_der == None:
            self.f_x = jacrev(self.f, 0)
            self.f_u = jacfwd(self.f, 1)
            
            (self.f, self.f_u, self.f_x,) = [jit(e) for e in [self.f, self.f_u, self.f_x,]]
        else:
            # using provided function for step
            self.f_x = model_der['f_x']
            self.f_u = model_der['f_u']
            

        # speed up
        (self.l, self.l_u, self.l_uu, self.l_ux, self.l_x, self.l_xx,
         self.v, self.v_x, self.v_xx) = \
            [jit(e) for e in [self.l, self.l_u, self.l_uu, self.l_ux, self.l_x, self.l_xx,
                              self.v, self.v_x, self.v_xx]]


    def cal_K(self, x_seq, u_seq):
        '''
            Calculate all the necessary derivatives, and compute the Ks
        '''
        state_dim = x_seq[0].shape[-1]
#         v_seq = [None] * self.horizon
        v_x_seq = [None] * self.horizon
        v_xx_seq = [None] * self.horizon

        last_x = x_seq[-1]
#         v_seq[-1] = self.v(last_x)
        v_x_seq[-1] = self.v_x(last_x)
        v_xx_seq[-1] = self.v_xx(last_x)

        k_seq = [None] * self.horizon
        kk_seq = [None] * self.horizon

        for i in tqdm(range(self.horizon - 2, -1, -1), desc='forward', leave=False):
            x, u = x_seq[i], u_seq[i]

            # get all grads
            lx = self.l_x(x, u)
            lu = self.l_u(x, u)
            lxx = self.l_xx(x, u)
            luu = self.l_uu(x, u)
            lux = self.l_ux(x, u)

            fx = self.f_x(x, u)
            fu = self.f_u(x, u)
#             fxx = self.f_xx(x, u)
#             fuu = self.f_uu(x, u)
#             fux = self.f_ux(x, u)

            vx = v_x_seq[i+1]
            vxx = v_xx_seq[i+1]

            # cal Qs
            q_x = lx + fx.T @ vx
            q_u = lu + fu.T @ vx
            q_xx = lxx + fx.T @ vxx @ fx
            q_uu = luu + fu.T @ vxx @ fu
            q_ux = lux + fu.T @ vxx @ fx
#             q_xx = lxx + fx.T @ vxx @ fx + vx @ fxx
#             q_uu = luu + fu.T @ vxx @ fu + (fuu.T @ vx).T
#             q_ux = lux + fu.T @ vxx @ fx + (fux.T @ vx).T

#             names = ['lx', 'lu', 'lxx', 'luu', 'lux', 'fx', 'fu', 'vx', 'vxx', 'qx', 'qu', 'qxx', 'quu', 'qux']
#             Ms = [lx, lu, lxx, luu, lux, fx, fu, vx, vxx, q_x, q_u, q_xx, q_uu, q_ux]
        
#             print(f"ITER {i}")
#             for n, m in zip(names, Ms):
#                 print(f"{n}\n\t{np.max(m)}\n")
    
            # cal Ks
            inv_quu = np.linalg.inv(q_uu)
            k = - inv_quu @ q_u
            kk = - inv_quu @ q_ux

            # cal Vs
            new_v = q_u @ k / 2
            new_vx = q_x + q_u @ kk
            new_vxx = q_xx + q_ux.T @ kk

            # record
            k_seq[i] = k
            kk_seq[i] = kk
            v_x_seq[i] = new_vx
            v_xx_seq[i] = new_vxx

        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        new_x_seq = [None] * self.horizon
        new_u_seq = [None] * self.horizon

        new_x_seq[0] = x_seq[0]  # copy

        for i in trange(self.horizon - 1, desc='backward', leave=False):
#             print(f"forward {i}")
            x = new_x_seq[i]

            new_u = u_seq[i] + k_seq[i] + kk_seq[i] @ (x - x_seq[i])
            new_u = np.clip(new_u, self.u_range[0], self.u_range[1])
            new_x = self.f(x, new_u)

            new_u_seq[i] = new_u
            new_x_seq[i+1] = new_x

        return new_x_seq, new_u_seq

    def predict(self, x_seq, u_seq):
        for _ in trange(self.per_iter, desc='ILQR', leave=False):
            k_seq, kk_seq = self.cal_K(x_seq, u_seq)
            x_seq, u_seq = self.forward(x_seq, u_seq, k_seq, kk_seq)
        
        u_seq[-1] = u_seq[0] # filling
        return np.array(x_seq), np.array(u_seq)


# In[9]:


def sim_step(scene, robot, action):
    robot.set_qf(action)
    scene.step()
        
    return get_state(robot)


target_body_parts=['torso', 'left_foot', 'right_foot']
target_body_parts_indx = [0, 9 ,12]

masses = get_mass(get_links(robot0))
masses = np.expand_dims(masses, axis=1)
mass_sum = np.sum(masses)
def final_cost(x, alpha=0.2):
    # only doing POS, TODO: add velocity:  com_v
    pos = x.reshape(-1, 3)
    torso_pos, lfoot_pos, rfoot_pos = pos[target_body_parts_indx]
    
    # calculate com_pos
    com_pos = np.average(pos[:, :2] * masses, axis=0) / mass_sum
    
    smooth_abs = lambda x : np.sum(np.sqrt(x**2 + alpha**2) - alpha)
    
    # calculate terms
    mean_foot = (lfoot_pos + rfoot_pos) / 2
    term1 = smooth_abs(com_pos[0:2] - mean_foot[:2])
    
    term2 = smooth_abs(com_pos[0:2] - torso_pos[:2])
    
    mean_foot_air = mean_foot[2] + 1.3
    term3 = smooth_abs(torso_pos[2] - mean_foot_air)
    
#     term4 = np.linalg.norm(com_v)
    
    return term1 + term2 + term3
    


def running_cost(x, u, alpha=0.3):
    return np.sum((alpha ** 2) * (np.cosh(u/alpha) - 1))


# In[10]:


u_range = np.array([[-10] * robot0.dof, [10] * robot0.dof])
pred_time = 5
horizon = int(pred_time / optim_timestep) + 1
per_iter = 3

# prep for multiprocess
n_state = get_state(robot0).shape[0]
n_dof = robot0.dof

fu_request_queue, fu_ans_arr, fu_num_task = create_workers('fu', 6, n_dof, n_state, optim_timestep)
fx_request_queue, fx_ans_arr, fx_num_task = create_workers('fx', 6, n_state, n_state, optim_timestep)

eps = 1e-3
model_der = {
    'f_x' : lambda x, u : mp_num_fx(fx_request_queue, fx_ans_arr, u, robot0, fx_num_task, eps=1e-3),
    'f_u' : lambda x, u : mp_num_fu(fu_request_queue, fu_ans_arr, u, robot0, fu_num_task, eps=1e-3)
}

ilqr = ILQR(final_cost, running_cost, lambda x, u : sim_step(s0, robot0, u), u_range, horizon, per_iter, model_der)


# In[11]:


# prepare simulation

# run to stable
if not stabled:
    for i in range(1000):
        s0.step()
    stabled = True

u_seq = np.zeros((horizon, robot0.dof))
x_seq = []
x_seq.append(get_state(robot0))
for i in range(horizon - 1):
    state = sim_step(s0, robot0, u_seq[i])
    x_seq.append(state)
x_seq = np.array(x_seq)


# In[12]:


ctrl = []

render_controller.show_window()

from tqdm.notebook import tqdm

l0 = robot0.get_links()[0]

# use another thread for rendering
# def thread_render():
#     render_controller.focus(l0)
#     render_controller.render()

# thread = threading.Thread(target=thread_render)

# s0.update_render()
# thread.start()
for i in trange(1000):
    x_seq, u_range = ilqr.predict(x_seq, u_seq)
    
    u = u_seq[0]
    ctrl.append(u.tolist())
    print(u)
    
    robot0.set_qf(u)
    s0.step()
    s0.update_render()
    
    render_controller.focus(l0)
    render_controller.render()

    
print(ctrl)
np.save('ctrl', ctrl)

