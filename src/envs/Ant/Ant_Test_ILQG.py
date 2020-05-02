# %%
import jax
import sapien.core as sapien
from sapien.core import Pose
from Tools import misc, ForwardKinematics, ModelSim, NumForwardDynamicsDer, MathForwardDynamics
import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad
from ilq import ILQG
import timeit
import os
import matplotlib.pyplot as plt
from transforms3d.quaternions import axangle2quat as aa

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true "
                           "intra_op_parallelism_threads=12")

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

#%%

copper = sapien.PxrMaterial()
copper.set_base_color([0.875, 0.553, 0.221, 1])
copper.metallic = 1
copper.roughness = 0.2

ant_poses = {
    'j1': (
        Pose([0.282, 0, 0], [0.7071068, 0, 0.7071068, 0]),
        Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0])),
    'j2': (
        Pose([-0.282, 0, 0], [0, -0.7071068, 0, 0.7071068]),
        Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0])),
    'j3': (
        Pose([0, 0.282, 0], [0.5, -0.5, 0.5, 0.5]),
        Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0])),
    'j4': (
        Pose([0, -0.282, 0], [0.5, 0.5, 0.5, -0.5]),
        Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0])),
    'j11': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j21': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j31': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j41': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
}

def create_ant_builder(scene):
    builder = scene.create_articulation_builder()
    body = builder.create_link_builder()
    body.add_sphere_shape(Pose(), 0.25)
    body.add_sphere_visual_complex(Pose(), 0.25, copper)
    body.add_capsule_shape(Pose([0.141, 0, 0]), 0.08, 0.141)
    body.add_capsule_visual_complex(Pose([0.141, 0, 0]), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([-0.141, 0, 0]), 0.08, 0.141)
    body.add_capsule_visual_complex(Pose([-0.141, 0, 0]), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
    body.add_capsule_visual_complex(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
    body.add_capsule_visual_complex(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, copper)
    body.set_name("body")

    l1 = builder.create_link_builder(body)
    l1.set_name("l1")
    l1.set_joint_name("j1")
    l1.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j1'][0], ant_poses['j1'][1], 0.1)
    l1.add_capsule_shape(Pose(), 0.08, 0.141)
    l1.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l2 = builder.create_link_builder(body)
    l2.set_name("l2")
    l2.set_joint_name("j2")
    l2.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j2'][0], ant_poses['j2'][1], 0.1)
    l2.add_capsule_shape(Pose(), 0.08, 0.141)
    l2.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l3 = builder.create_link_builder(body)
    l3.set_name("l3")
    l3.set_joint_name("j3")
    l3.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j3'][0], ant_poses['j3'][1], 0.1)
    l3.add_capsule_shape(Pose(), 0.08, 0.141)
    l3.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l4 = builder.create_link_builder(body)
    l4.set_name("l4")
    l4.set_joint_name("j4")
    l4.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j4'][0], ant_poses['j4'][1], 0.1)
    l4.add_capsule_shape(Pose(), 0.08, 0.141)
    l4.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    f1 = builder.create_link_builder(l1)
    f1.set_name("f1")
    f1.set_joint_name("j11")
    f1.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j11'][0], ant_poses['j11'][1], 0.1)
    f1.add_capsule_shape(Pose(), 0.08, 0.282)
    f1.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f2 = builder.create_link_builder(l2)
    f2.set_name("f2")
    f2.set_joint_name("j21")
    f2.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j21'][0], ant_poses['j21'][1], 0.1)
    f2.add_capsule_shape(Pose(), 0.08, 0.282)
    f2.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f3 = builder.create_link_builder(l3)
    f3.set_name("f3")
    f3.set_joint_name("j31")
    f3.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j31'][0], ant_poses['j31'][1], 0.1)
    f3.add_capsule_shape(Pose(), 0.08, 0.282)
    f3.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f4 = builder.create_link_builder(l4)
    f4.set_name("f4")
    f4.set_joint_name("j41")
    f4.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j41'][0], ant_poses['j41'][1], 0.1)
    f4.add_capsule_shape(Pose(), 0.08, 0.282)
    f4.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    return builder

# %%

DEBUG = False


def create_scene(timestep, visual):
    s = sim.create_scene()
    s.add_ground(-1)
    s.set_timestep(timestep)

    if visual:
        s.set_ambient_light([0.5, 0.5, 0.5])
        s.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])

    # build
    ant_builder = create_ant_builder(s)
    robot = ant_builder.build()

    for joint in robot.get_joints():
        joint.set_drive_property(stiffness=0, damping=5)
    # robot.set_qpos(np.array([0, 1.7, 0, -1.5, 0, 2.5, 0.7, 0.04, 0.04]))

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
    return np.mean((alpha ** 2) * (np.cosh(x / alpha) - 1))


dof = robot.dof
factor = np.array([10000] * robot.dof)
u_range = np.array([-factor, factor])
q_range = robot.get_qlimits().T
q_mean = np.mean(q_range, axis=0)
q_radius = q_range[1] - q_mean

pred_time = 0.4
horizon = int(pred_time / optim_timestep) + 1
per_iter = 1

robo_pose = Pose()

@jit
def final_cost(x, alpha1=0.2, alpha2=0.5, alpha3=0.5):
    # add base pose
    qpos = x[:dof]
    qvel = x[dof:]
    pos = np.concatenate((qpos, robo_pose.p, robo_pose.q))

    cart_pos = fk.fk(pos).reshape(-1, 3)[:, :3]
    foot_center_pos = np.mean(cart_pos[-4:], axis=0)

    target_pos = robo_pose.p + [0, 0, -2]
    # target_pos = robo_pose.p + [0.5, 0.5, 0]
    diff = target_pos - foot_center_pos
    term1 = smooth_abs(diff, alpha1)

    # penalize high velocity
    term2 = smooth_abs(qvel / (q_radius * 2 * 40), alpha2)  # 1/40 seconds go full qpos span

    return term1 + term2


@jit
def running_cost(x, u, alpha=0.7):
    term1 = smooth_abs(u / factor, alpha) / horizon * 5
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

IF_RECORD = False
IF_PLOT = True

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

render_controller.show_window()
s0.update_render()
render_controller.render()
for i in range(2000):
    s0.update_render()
    render_controller.render()

    st = timeit.default_timer()
    x_seq, u_seq, pack_seq, last_cost = ilqg.predict(x_seq, u_seq, pack_seq, last_cost)
    print(timeit.default_timer() - st)

    u = u_seq[0]

    robot.set_qf(u + robot.compute_passive_force())
    for _ in range(render_steps):
        s0.step()

    new_x = misc.get_state(robot)
    new_pack = robot.pack()

    if IF_RECORD or IF_PLOT:
        f_cost = final_cost(x_seq[-1])
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
