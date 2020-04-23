import sapien.core as sapien
from Tools import misc, ForwardKinematics, SimWorker, NumForwardDynamicsDer
import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad
from ilqr import ILQR
from tqdm import trange

# from jax.config import config
# config.update("jax_debug_nans", True)

DEBUG = False

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

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

    return s, robot


sim_timestep = 1 / 60
optim_timestep = 1 / 60
s0, robot = create_scene(sim_timestep, True)

render_controller.set_camera_position(-5, 0, 0)
render_controller.set_current_scene(s0)

if not stabled:
    for _ in range(3000):
        s0.step()
    stabled = True


# render_controller.show_window()
# while True:
#     u = onp.random.randn(robot.dof) * 0.5
#     u = onp.clip(u, -1, 1)
#     robot.set_qf(u)
#     s0.step()
#     s0.update_render()
#     render_controller.render()


def smooth_abs(x, alpha):
    return np.sum((alpha ** 2) * (np.cosh(x / alpha) - 1))


robo_pose = robot.get_root_pose()


@jit
def final_cost(x, alpha=0.3):
    # add base pose
    x = np.concatenate((x, robo_pose.p, robo_pose.q))

    cart_pos = fk.fk(x).reshape(-1, 3)[:, :3]
    end_effector_pos = cart_pos[-3]

    target_height = x[-1] + 1.2

    term1 = smooth_abs((target_height - end_effector_pos[2]) * 2, alpha)

    return term1 * 10


limits = robot.get_qlimits().reshape(2, -1)
limits_sum = np.abs((limits[1] - limits[0]))/2
limits_mid = np.mean(limits, axis=0)

@jit
def running_cost(x, u, alpha=0.5):
    term1 = np.sum(smooth_abs(u / 5, alpha))
    dist2limit = np.abs(x - limits_mid) / limits_sum
    # term2 = np.sum(smooth_abs(dist2limit / limits_sum, alpha))
    term2 = smooth_abs(dist2limit, 2)
    term2 /= 20
    return term1 + term2

state = misc.get_state(robot)
num_x = len(state)
num_u = robot.dof
dof = robot.dof

num_deri = NumForwardDynamicsDer(robot, sim_timestep)
fk = ForwardKinematics(robot)
sim_worker = SimWorker(robot, create_scene, optim_timestep, DEBUG)

u_range = np.array([[-10] * robot.dof, [10] * robot.dof])
pred_time = 1.3
horizon = int(pred_time / optim_timestep) + 1
per_iter = 3

ilqr = ILQR(final_cost, running_cost, None, u_range, horizon, per_iter, num_deri, sim_worker, DEBUG)

# prep seq
x_seq = []
u_seq = []
pack_seq = []

bak_pack = robot.pack()

for i in range(horizon):
    u = onp.random.randn(robot.dof) * 0.5
    u = onp.clip(u, -1, 1)

    x = misc.get_state(robot)
    pack = robot.pack()

    x_seq.append(x)
    pack_seq.append(pack)
    u_seq.append(u)

    robot.set_qf(u)
    s0.step()
robot.unpack(bak_pack)

ctrl = []

vx = jacfwd(final_cost)
vxx = jacfwd(vx)

vx_ca = []
vxx_ca = []
for x in x_seq:
    vx_ca.append(vx(x))
    vxx_ca.append(vxx(x))

render_controller.show_window()
for i in range(1000):
    x_seq, u_seq, pack_seq = ilqr.predict(x_seq, u_seq, pack_seq)

    u = u_seq[0]
    ctrl.append(u)

    robot.set_qf(u)
    s0.step()
    s0.update_render()
    render_controller.render()

    new_x = misc.get_state(robot)
    new_pack = robot.pack()

    cost = final_cost(x_seq[-1]) + onp.sum([running_cost(x, u) for x, u in zip(x_seq, u_seq)])
    print(f"ITER {i}\ncost: {cost}\nu: {u}")
    x_seq[0] = new_x
    pack_seq[0] = new_pack

np.save('ctrl', ctrl)
