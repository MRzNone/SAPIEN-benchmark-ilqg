#%%

import sapien.core as sapien
from Tools import DerivativeFactory, misc, ForwardKinematics, SimWorker
import numpy as onp
import jax.numpy as np
import jax
from jax import jit, jacfwd, jacrev, grad
from ilqr import ILQR
from tqdm import trange

# from jax.config import config
# config.update("jax_debug_nans", True)

#%%

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
sim.set_renderer(renderer)
render_controller = sapien.OptifuserController(renderer)

stabled = False

def create_scene(timestep, visual):
    s = sim.create_scene([0,0,0])
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

sim_timestep = 1/60
optim_timestep = 1/60
s0, robot = create_scene(sim_timestep, True)

render_controller.set_camera_position(-5, 0, 0)
render_controller.set_current_scene(s0)

#%%

if not stabled:
    for _ in range(3000):
        s0.step()
    stabled = True


#%%

state = misc.get_state(robot)
num_x = len(state)
num_u = robot.dof
dof = robot.dof

num_deri = DerivativeFactory(num_x, num_u, dof, create_scene, 6, optim_timestep)
fk = ForwardKinematics(robot)
sim_worker = SimWorker(robot, create_scene, optim_timestep)

#%%

def smooth_abs(x, alpha):
    return np.sum((alpha ** 2) * (np.cosh(x / alpha) - 1))

@jit
def final_cost(x, alpha=0.2):
    cart_pos = fk.fk(x).reshape(-1, 3)[:, :3]
    end_effector_pos = cart_pos[-3]

    target_height = x[-1] + 1

    term1 = smooth_abs(target_height - end_effector_pos[2], alpha)

    return term1 * 1000

@jit
def running_cost(x, u, alpha=0.3):
    return np.sum(smooth_abs(u, alpha))


#%%

u_range = np.array([[-100] * robot.dof, [100] * robot.dof])
pred_time = 0.6
horizon = int(pred_time / optim_timestep) + 1
per_iter = 3

ilqr = ILQR(final_cost, running_cost, None, u_range, horizon, per_iter, num_deri, sim_worker)

#%%

#prep seq
x_seq = []
u_seq = list(np.ones((horizon, dof)))
pack_seq = []

bak_pack = robot.pack()

for i in range(horizon):
    x = misc.get_state(robot)
    pack = robot.pack()

    x_seq.append(x)
    pack_seq.append(pack)

    u = u_seq[i]
    robot.set_qf(u)
    s0.step()
robot.unpack(bak_pack)

#%%

ctrl = []

render_controller.show_window()
for i in trange(1000):
   x_seq, u_seq, pack_seq = ilqr.predict(x_seq, u_seq, pack_seq)

   u = u_seq[0]
   ctrl.append(u)
   print(u)

   robot.set_qf(u)
   s0.step()
   s0.update_render()
   render_controller.render()

   new_x = misc.get_state(robot)
   new_pack = robot.pack()

   x_seq[0] = new_x
   pack_seq[0] = new_pack

np.save('ctrl', ctrl)

#%%

for u in ctrl:
   robot.set_qf(u)
   s0.step()
   s0.update_render()

   render_controller.render()

