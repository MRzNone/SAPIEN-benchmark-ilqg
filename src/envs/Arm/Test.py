#%%

import sapien.core as sapien
from Tools import misc, ForwardKinematics, ModelSim, NumForwardDynamicsDer, MathForwardDynamicsDer
import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad
from ilqr import ILQR
from datetime import datetime
from tqdm import trange
# from tqdm.notebook import trange
import matplotlib.pyplot as plt

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

#%%

DEBUG = True

stabled = False


def create_scene(timestep, visual):
    s = sim.create_scene([0, 0, 0])
    s.add_ground(-1)
    s.set_timestep(timestep)

    loader = s.create_urdf_loader()
    loader.fix_root_link = True
    if visual:
        loader.collision_is_visual = True
        s.set_ambient_light([0.5, 0.5, 0.5])
        s.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])

    # build
    robot = loader.load("../../../assets/Arm/panda.urdf")

    for joint in robot.get_joints():
        joint.set_drive_property(stiffness=0, damping=50)
    robot.set_qpos(np.array([0, 1.7, 0, -1.5, 0, 2.5, 0.7, 0.4, 0.4]))

    return s, robot


# renderer_timestep = 1 / 60
optim_timestep = 1 / 500
sim_timestep = 1 / 60
s0, robot = create_scene(sim_timestep, True)

render_controller.set_camera_position(-5, 0, 0)
render_controller.set_current_scene(s0)

#%%

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

#%%

deri = MathForwardDynamicsDer(robot, optim_timestep, False, False, False)
# deri = NumForwardDynamicsDer(robot, sim_timestep)
fk = ForwardKinematics(robot)
sim_worker = ModelSim(create_scene, optim_timestep)


#%%

def smooth_abs(x, alpha):
    return np.mean((alpha ** 2) * (np.cosh(x / alpha) - 1))


robo_pose = robot.get_root_pose()

dof = robot.dof
factor = np.array([150, 150, 150, 100, 300, 300, 300, 10, 10])
u_range = np.array([-factor, factor])

pred_time = 0.5
horizon = int(pred_time / optim_timestep) + 1
per_iter = 2


@jit
def final_cost(x, alpha1=0.3, alpha2=0.5):
    # add base pose
    qpos = x[:dof]
    qvel = x[dof:]
    pos = np.concatenate((qpos, robo_pose.p, robo_pose.q))

    cart_pos = fk.fk(pos).reshape(-1, 3)[:, :3]
    end_effector_pos = cart_pos[-3]

    target_pos = robo_pose.p + [0, 0, 1.2]
    term1 = smooth_abs(end_effector_pos[2] - target_pos[2], alpha1)

    term2 = smooth_abs(qvel/50, alpha2)

    return term1 * 5 + term2


@jit
def running_cost(x, u, alpha=0.3):
    term1 = smooth_abs(u / factor, alpha) / horizon * 5
    return term1


#%%

ilqr = ILQR(final_cost, running_cost, None, u_range, horizon, per_iter, deri, sim_worker, DEBUG)

#%%

state = misc.get_state(robot)
num_x = len(state)
num_u = robot.dof

# prep seq
x_seq = []
u_seq = []
pack_seq = []

bak_pack = robot.pack()

for i in range(horizon):
    u = onp.random.random(robot.dof) * factor * 2 - factor
    # u = np.clip(u, u_range[0], u_range[1])

    x = misc.get_state(robot)
    pack = robot.pack()

    x_seq.append(x)
    pack_seq.append(pack)
    u_seq.append(u)

    robot.set_qf(u)
    s0.step()
robot.unpack(bak_pack)

#%%

# records
ctrl_record = []
x_record = []
u_record = []
run_cost_record = []
f_cost_record = []
ini_state = [bak_pack]

# plots
IF_PLOT = True

if IF_PLOT:
    fig, axs = plt.subplots(3, figsize=(12, 9))

    total = []
    f_ax, r_ax, t_ax = axs
    PLOT_LEN = 30

    plt.ion()

    fig.show()
    fig.canvas.draw()

render_controller.show_window()
for i in trange(2000):
    x_seq, u_seq, pack_seq = ilqr.predict(x_seq, u_seq, pack_seq)

    u = u_seq[0]
    print(u)

    robot.set_qf(u + robot.compute_passive_force())
    s0.step()
    s0.update_render()
    render_controller.render()

    new_x = misc.get_state(robot)
    new_pack = robot.pack()

    f_cost = final_cost(x_seq[-1])
    run_cost = onp.sum([running_cost(x, u) for x, u in zip(x_seq[:-1], u_seq[:-1])])
    cost = f_cost + run_cost

    # if f_cost < 0.01:
    #     for u in u_seq[1:]:
    #         robot.set_qf(u)
    #         s0.step()
    #         s0.update_render()
    #     break

    # record
    #     print(u)
    ctrl_record.append(u)
    f_cost_record.append(f_cost)
    run_cost_record.append(run_cost)
    x_record.append(x_seq)
    u_record.append(u_seq)

    # update x and u, since we need to record old x u
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

        x_lim = max(0, i - PLOT_LEN)
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

#%%

records = {
    'ctrl_record': ctrl_record,
    'x_record': x_record,
    'u_record': u_record,
    'run_cost_record': run_cost_record,
    'f_cost_record': f_cost_record,
    'ini_state': ini_state,
}

np.save('records' + str(datetime.utcnow()), records)
