# %%

import sapien.core as sapien
from Tools import misc, ForwardKinematics, ModelSim, NumForwardDynamicsDer, MathForwardDynamics
import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad
from ilq import ILQG
import timeit
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

# %%

DEBUG = False


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
    # robot.set_qpos(np.array([0, 1.7, 0, -1.5, 0, 2.5, 0.7, 0.04, 0.04]))
    p = onp.random.random(robot.dof) * 3 - 1.5
    lim = robot.get_qlimits().T
    p = onp.clip(p, lim[0], lim[1])
    robot.set_qpos(p)

    return s, robot


# renderer_timestep = 1 / 60
optim_timestep = 1 / 500
sim_timestep = 1 / 60
s0, robot = create_scene(sim_timestep, True)

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
    return np.mean((alpha ** 2) * (np.cosh(x / alpha) - 1))


robo_pose = robot.get_root_pose()

dof = robot.dof
factor = np.array([150, 150, 150, 100, 300, 300, 300, 10, 10])
u_range = np.array([-factor, factor])
q_range = robot.get_qlimits().T
q_mean = np.mean(q_range, axis=0)
q_radius = q_range[1] - q_mean

pred_time = 0.4
horizon = int(pred_time / optim_timestep) + 1
per_iter = 2


@jit
def final_cost(x, alpha1=0.2, alpha2=0.5, alpha3=0.5):
    # add base pose
    qpos = x[:dof]
    qvel = x[dof:]
    pos = np.concatenate((qpos, robo_pose.p, robo_pose.q))

    cart_pos = fk.fk(pos).reshape(-1, 3)[:, :3]
    end_effector_pos = cart_pos[-3]

    # target_pos = robo_pose.p + [0, 0, 1.2]
    target_pos = robo_pose.p + [0.5, 0.5, 0]
    diff = target_pos - end_effector_pos
    term1 = smooth_abs(diff, alpha1)

    # penalize high velocity
    term2 = smooth_abs(qvel / (q_radius * 2 * 40), alpha2)  # 1/40 seconds go full qpos span

    return term1 * 10 + term2


@jit
def running_cost(x, u, alpha=0.3):
    term1 = smooth_abs(u / factor, alpha) / horizon
    return term1


# %%

ilqg = ILQG(final_cost, running_cost, None, u_range, horizon, per_iter, deri, sim_worker, DEBUG)

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

last_cost = 0

render_controller.show_window()
for i in range(2000):
    s0.update_render()
    render_controller.render()

    st = timeit.default_timer()
    x_seq, u_seq, pack_seq, last_cost = ilqg.predict(x_seq, u_seq, pack_seq, last_cost)
    print(timeit.default_timer() - st)

    u = u_seq[0]

    robot.set_qf(u + robot.compute_passive_force())
    s0.step()

    new_x = misc.get_state(robot)
    new_pack = robot.pack()
