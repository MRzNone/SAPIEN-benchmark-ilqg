# %%
from datetime import datetime

import sapien.core as sapien
from sapien.core import Pose
from Tools import misc, ForwardKinematics, ModelSim, NumForwardDynamicsDer, MathForwardDynamics
import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad
from ilq import ILQG_Human
import timeit
import os
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=12")

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

# %%

DEBUG = False


def create_scene(timestep, visual):
    s = sim.create_scene()
    s.add_ground(-1)
    s.set_timestep(timestep)

    loader = s.create_urdf_loader()
    loader.fix_root_link = False
    if visual:
        loader.collision_is_visual = True
        s.set_ambient_light([0.5, 0.5, 0.5])
        s.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])

    # build
    robot = loader.load("../../../assets/Humanoid/humanoid.urdf")
    robot.set_pose(Pose([0, 0, -0.7], [0, 0.7071068, 0, 0.7071068]))

    for joint in robot.get_joints():
        joint.set_drive_property(stiffness=0, damping=10)

    for _ in range(int(1/timestep) * 4):
        s.step()

    return s, robot


# renderer_timestep = 1 / 60
optim_timestep = 1 / 500
sim_timestep = 1 / 60
render_steps = int(sim_timestep / optim_timestep) + 1
s0, robot = create_scene(optim_timestep, True)

render_controller.set_camera_position(-5, 0, 0)
render_controller.set_current_scene(s0)

# %%

# factor = np.array([150, 150, 150, 100, 300, 300, 300, 10, 10])
# u_range = np.array([-factor, factor])
# render_controller.show_window()
# while not render_controller.should_quit:
#     u = onp.random.random(robot.dof) * factor * 2 - factor
#     u = np.clip(u, u_range[0], u_range[1])
#     robot.set_qf(u)
#     s0.step()
#     s0.update_render()
#     render_controller.render()

# %%

# _, robot2 = create_scene(optim_timestep, False)
deri = MathForwardDynamics(create_scene, optim_timestep, robot.pack(), True, True, True)
# deri = NumForwardDynamicsDer(robot, sim_timestep)
fk = ForwardKinematics(robot)
sim_worker = ModelSim(create_scene, optim_timestep)


# %%

def smooth_abs(x, alpha):
    return (alpha ** 2) * (np.cosh(x / alpha) - 1)


robo_pose = robot.get_root_pose()

dof = robot.dof
factor = np.array([200] * robot.dof)
u_range = np.array([-factor, factor])
q_range = robot.get_qlimits().T
q_mean = np.mean(q_range, axis=0)
q_radius = q_range[1] - q_mean

pred_time = 0.8
horizon = int(pred_time / optim_timestep) + 1
per_iter = 2

masses = {}
for l in robot.get_links():
    masses[l.get_name()] = np.array(l.get_mass())
mass_ls = np.array(list(masses.values())).flatten()
mass_sum = np.sum(list(masses.values()))

fk_names = ['torso', 'link1_30', 'link1_31', 'left_upper_arm', 'link1_33', 'left_lower_arm', 'link1_25', 'link1_26', 'right_upper_arm', 'link1_28', 'right_lower_arm', 'link1_2', 'link1_3', 'lwaist', 'link1_5', 'pelvis', 'link1_16', 'link1_17', 'link1_18', 'left_thigh', 'link1_20', 'left_shin', 'link1_22', 'link1_23', 'left_foot', 'link1_7', 'link1_8', 'link1_9', 'right_thigh', 'link1_11', 'right_shin', 'link1_13', 'link1_14', 'right_foot']
mass_fk_ls = np.array([masses[name] for name in fk_names])

@jit
def final_cost(x, root_pos, jaco, alpha1=0.2, alpha2=0.5, alpha3=0.5):
    # add base pose
    qpos = x[:dof]
    qvel = x[dof:]
    pos = np.concatenate((qpos, root_pos))

    pos_dict = fk.fk(pos, True)
    # cart_pos = np.array(list(pos_dict.values())).reshape(-1, 3)

    cart_pos = []
    for n, v in pos_dict.items():
        cart_pos.append(v)
    cart_pos = np.array(cart_pos)

    # COM
    com = cart_pos.T @ mass_ls.T / mass_sum

    # feet
    lfoot = pos_dict['left_foot']
    rfoot = pos_dict['right_foot']
    feet_mid = (lfoot + rfoot) / 2.0

    # torso
    torso = pos_dict['torso']

    # com velocity, doubtful
    velo = (jaco @ qvel).reshape(-1, 6)
    velo_com = np.mean(velo.T * mass_ls[1:], axis=0).T

    # com and feet mid
    term1 = np.sum(smooth_abs(com[:2] - feet_mid[:2], alpha1))

    # torso and com
    term2 = np.sum(smooth_abs(torso[:2] - com[:2], alpha1))

    # torso and air point
    air_pt = feet_mid + np.array([0, 0, 1.3])
    term3 = np.sum(smooth_abs(torso - air_pt, alpha1))
    term3 = term3 * 1.3

    # com velo
    term4 = np.sum(velo_com[:2] ** 2)

    return term1 + term2 + term3 + term4

@jit
def running_cost(x, u, alpha=0.7):
    term1 = smooth_abs(u / factor, alpha) / horizon
    return np.mean(term1)


# %%

ilqg = ILQG_Human(final_cost, running_cost, None, u_range, horizon, per_iter, deri, sim_worker, DEBUG)

# %%

state = misc.get_state(robot)
num_x = len(state)
num_u = robot.dof


def prep():
    # prep seq
    x_seq = []
    u_seq = []
    pack_seq = []

    bak_pack = robot.pack()

    for i in range(horizon):
        u = np.zeros((robot.dof,))
        # u = np.clip(u, u_range[0], u_range[1])

        x = misc.get_state(robot)
        pack = robot.pack()

        x_seq.append(x)
        pack_seq.append(pack)
        u_seq.append(u)

        robot.set_qf(u)
        s0.step()
    robot.unpack(bak_pack)
    return x_seq, u_seq, pack_seq


# %%

x_seq, u_seq, pack_seq = prep()

root_pos = robot.get_pose()
root_pos = np.concatenate([root_pos.p, root_pos.q])
jaco = np.array(robot.compute_jacobian())

last_cost = 0
for x, u in zip(x_seq[:-1], u_seq[:-1]):
    last_cost += running_cost(x, u)
last_cost += final_cost(x_seq[-1], root_pos, jaco)


IF_RECORD = True
IF_PLOT = False

if IF_RECORD:
    # records
    ctrl_record = []
    x_record = []
    u_record = []
    ini_state = [robot.pack()]

if IF_RECORD or IF_PLOT:
    run_cost_record = []
    f_cost_record = []

if IF_PLOT:
    fig, axs = plt.subplots(3, figsize=(12, 9))

    total = []
    f_ax, r_ax, t_ax = axs
    PLOT_LEN = 30

    plt.ion()

    fig.show()
    fig.canvas.draw()

last_cost = 0

u_file = open("HUMAN_U_"+str(datetime.now()), 'a')

render_controller.show_window()
render_controller.focus(robot.get_links()[0])
for i in range(2000):
    s0.update_render()
    render_controller.render()

    root_pos = robot.get_pose()
    root_pos = np.concatenate([root_pos.p, root_pos.q])
    jaco = np.array(robot.compute_jacobian())

    st = timeit.default_timer()
    x_seq, u_seq, pack_seq, last_cost = ilqg.predict(x_seq, u_seq, pack_seq, last_cost, root_pos, jaco)
    print(timeit.default_timer() - st)

    u = u_seq[0]
    u_file.write(str(u) + "\n")

    robot.set_qf(u + robot.compute_passive_force())
    for _ in range(render_steps):
        s0.step()

    new_x = misc.get_state(robot)
    new_pack = robot.pack()

    if IF_RECORD or IF_PLOT:
        f_cost = final_cost(x_seq[-1], root_pos, jaco)
        run_cost = onp.sum([running_cost(x, u) for x, u in zip(x_seq[:-1], u_seq[:-1])])
        cost = f_cost + run_cost

    if IF_RECORD:
        # record
        ctrl_record.append(u)
        x_record.append(x_seq)
        u_record.append(u_seq)

    if IF_RECORD or IF_PLOT:
        f_cost_record.append(f_cost)
        run_cost_record.append(run_cost)

    # update x and u here, since we need to record old x u
    x_seq[0] = new_x
    pack_seq[0] = new_pack

    # plot
    if IF_PLOT:
        total.append(cost)
        f_ax.clear()
        r_ax.clear()
        t_ax.clear()

        y_lim = np.max(total[-PLOT_LEN:]) * 1.2
        f_ax.set_ylim(0, y_lim)
        r_ax.set_ylim(0, y_lim)
        t_ax.set_ylim(0, y_lim)

        x_lim = max(0, i - PLOT_LEN) - 1
        f_ax.set_xlim(x_lim, i)
        r_ax.set_xlim(x_lim, i)
        t_ax.set_xlim(x_lim, i)

        f_ax.plot(f_cost_record)
        f_ax.set_title("Final")
        r_ax.plot(run_cost_record)
        r_ax.set_title("Running")

        t_ax.plot(total, label="Total")
        t_ax.plot(f_cost_record, label="Final")
        t_ax.plot(run_cost_record, label="Running")
        t_ax.legend()
        t_ax.set_title("Total")

        fig.canvas.draw()
        fig.show()

u_file.close()

if IF_RECORD:
    records = {
        'ctrl_record': ctrl_record,
        'x_record': x_record,
        'u_record': u_record,
        'run_cost_record': run_cost_record,
        'f_cost_record': f_cost_record,
        'ini_state': ini_state,
    }

    np.save('records' + str(datetime.utcnow()), records)
