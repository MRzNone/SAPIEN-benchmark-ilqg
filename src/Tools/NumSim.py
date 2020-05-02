from . import misc, ModelDerivator
import jax.numpy as np
import jax
from jax import jacfwd, jacrev, grad
import numpy as onp
from sapien.core import Pose
import sapien.core as sapien


class JointInfo:
    def __init__(self, joint: sapien.Joint):
        parentPose = joint.get_pose_in_parent_frame()
        childPose = joint.get_pose_in_child_frame()
        movable = joint.get_dof() != 0

        self.joint = joint
        self.parentPoseMat = misc.compose_p(parentPose)
        self.childPoseMat = np.linalg.inv(misc.compose_p(childPose))
        self.movable = movable
        self.childs = []
        self.qId = None
        self.index = None
        self.name = joint.get_child_link().get_name()

    def add_child(self, child):
        self.childs.append(child)

    def get_name(self):
        return self.name

    def getForward(self, theta=0.0):
        jp_mat = self.parentPoseMat
        jc_mat = self.childPoseMat

        if self.movable:
            joint_mat = np.eye(4)
            m = misc.m_axangle2mat([1, 0, 0], theta)
            joint_mat = jax.ops.index_update(joint_mat, jax.ops.index[:3, :3], m)

            return jp_mat @ joint_mat @ jc_mat
        else:
            return jp_mat @ jc_mat


class ForwardKinematics:
    def __init__(self, robot: sapien.Articulation):
        joint_dict = {}
        self.names = []
        self.root = None
        # build
        qId = 0
        index = 0
        for j in robot.get_joints():
            info = JointInfo(j)
            info.index = index
            index += 1
            if j.get_dof() != 0:
                info.qId = qId
                qId += 1
                self.names.append(j.get_name())

            joint_dict[j.get_child_link()] = info
            if self.root is None:
                self.root = info

        # form graph
        for j in robot.get_joints():
            if j.get_parent_link() is None:
                continue

            parentInfo = joint_dict[j.get_parent_link()]

            parentInfo.add_child(joint_dict[j.get_child_link()])

    def fk_recurse(self, pMat, joint: JointInfo, qPos: list, res: dict):
        # get theta
        theta = 0.0
        if joint.qId is not None:
            theta = qPos[joint.qId]

        # get this pose
        forward_mat = joint.getForward(theta)

        mat = pMat @ forward_mat

        res[joint.get_name()] = misc.decompose_pos_only(mat)

        for child in joint.childs:
            self.fk_recurse(mat, child, qPos, res)

    @jax.partial(jax.jit, static_argnums=(0, 2))
    def fk(self, state, ifDict=False):
        if self.root is None:
            return None

        bPose = state[-7:]
        qPos = state[:-7]

        res = {}
        base_mat = misc.compose(bPose[:3], bPose[3:])

        self.fk_recurse(base_mat, self.root, qPos, res)

        if ifDict:
            # arr = []
            # for n, v in res.items():
            #     arr.append(v)
            # return res, np.array(arr)
            return res
        else:
            arr = []
            for name in self.names:
                arr.append(res[name])
            return np.array(arr)


class Dynamics:
    def __init__(self, robot: sapien.Articulation, timestep: float, gravity: bool = True,
                 coriolisAndCentrifugal: bool = True, external: bool = True):
        self.robot = robot
        self.dof = self.robot.dof
        self.timestep = timestep
        self.gravity = gravity
        self.coriolisAndCentrifugal = coriolisAndCentrifugal
        self.external = external

        with jax.disable_jit():
            self.pack = self.robot.pack()

    def set_pack(self, pack):
        self.pack = pack

    @jax.partial(jax.jit, static_argnums=(0,))
    def forward(self, x, u: np.ndarray):
        """
                compute forward dynamics
                u: the force on genralised coordinate
                timestep: [optional] simulation timestep

                return the change in qpos purely due to u
        """
        self.robot.unpack(self.pack)
        other_force = self.robot.compute_passive_force(self.gravity, self.coriolisAndCentrifugal, self.external)

        qpos = x[:self.dof]
        qvel = x[self.dof:]

        f = u - other_force

        mass = self.robot.compute_mass_matrix()
        inv_mass = np.linalg.inv(mass)

        qacc = inv_mass @ f

        delta_qvel = qacc * self.timestep
        delta_qpos = delta_qvel * self.timestep

        new_qpos = qpos + delta_qpos
        new_qvel = qvel + delta_qvel

        # stupid delta update
        return np.concatenate((new_qpos, new_qvel))

    @jax.partial(jax.jit, static_argnums=(0,))
    def stupid_forward_deri_u(self, u, x):
        mass = self.robot.compute_mass_matrix()
        inv_mass = np.linalg.inv(mass)

        dpdu = inv_mass.T * self.timestep ** 2
        dvdu = inv_mass.T * self.timestep

        return np.vstack((dpdu, dvdu))

    @jax.partial(jax.jit, static_argnums=(0,))
    def inverse(self, x):
        """
            force from acceleration(state)
        """
        qAcc = x[:self.robot.dof]
        # linear
        linear_force = self.mass @ qAcc

        # other
        self.robot.unpack(self.pack)
        F = self.robot.compute_passive_force(self.gravity, self.coriolisAndCentrifugal, self.external)

        return linear_force + F


class MathForwardDynamics(ModelDerivator):

    def __init__(self, create_scene, timestep: float, pack, gravity: bool = True,
                 coriolisAndCentrifugal: bool = True, external: bool = True):
        super()
        self.scene, self.robot = create_scene(timestep, False)
        self.dym = Dynamics(self.robot, timestep, gravity, coriolisAndCentrifugal, external)
        self.dym_fu = jacfwd(self.dym.forward, 1)
        self.robot = self.robot
        self.num_x = len(misc.get_state(self.robot))
        self.set_pack(pack)

    def set_pack(self, pack):
        with jax.disable_jit():
            self.pack = pack
            self.dym.set_pack(pack)

    def f(self, x: np.array, u: np.array):
        return self.dym.forward(x, u)

    @jax.partial(jax.jit, static_argnums=(0,))
    def fu(self, x: np.array, u: np.array, eps=None) -> np.array:
        return self.dym_fu(x, u)

    @jax.partial(jax.jit, static_argnums=(0,))
    def fx(self, x: np.array, u: np.array, eps=None) -> np.array:
        return onp.eye(self.num_x)


class NumForwardDynamicsDer(ModelDerivator):
    def __init__(self, robot: sapien.Articulation, timestep: float, def_eps=1e-3, include_external=False):
        super()
        self.robot = robot
        self.timestep = timestep
        self.def_eps = def_eps
        self.num_x = self.robot.dof
        self.num_u = len(misc.get_state(robot))
        self.dof = self.robot.dof
        self.pack = self.robot.pack()

        self.limits = self.robot.get_qlimits().T

    def set_pack(self, pack):
        self.pack = pack

    def fx(self, x: np.array, u: np.array, eps: float = 1e-3) -> np.array:
        """
            0.5*(a2 - a1)t
        """
        return onp.eye(self.num_x)

        # res = []
        #
        # orig_pack = self.robot.pack()
        # orig_x = self.robot.get_qpos()[:self.robot.dof].tolist()
        # for i in range(self.num_x):
        #     self.robot.unpack(self.pack)
        #     a2, a1 = None, None
        #
        #     if i < self.robot.dof:
        #         new_x1 = orig_x.copy()
        #         new_x2 = orig_x.copy()
        #
        #         new_x1[i] -= eps
        #         new_x2[i] += eps
        #
        #         self.robot.set_qpos(new_x1)
        #         a1 = self.robot.compute_forward_dynamics(u)
        #         self.robot.set_qpos(new_x2)
        #         a2 = self.robot.compute_forward_dynamics(u)
        #     else:
        #         check = i - self.robot.dof
        #         index = check if check < 3 else check - 3
        #
        #         robo_pose = self.robot.get_pose()
        #         p1 = robo_pose.p.tolist()
        #         q1 = robo_pose.q.tolist()
        #
        #         p2 = p1.copy()
        #         q2 = q1.copy()
        #
        #         if check < 3:
        #             p1[index] -= eps
        #             p2[index] += eps
        #         else:
        #             q1[index] -= eps
        #             q2[index] += eps
        #
        #             # sketchy normalization
        #             q1 = q1 / np.linalg.norm(q1)
        #             q1 = q1 if q1[0] > 0 else -q1
        #
        #             q2 = q2 / np.linalg.norm(q2)
        #             q2 = q2 if q2[0] > 0 else -q2
        #
        #         self.robot.set_pose(Pose(p1, q1))
        #         a1 = self.robot.compute_forward_dynamics(u)
        #
        #         self.robot.set_pose(Pose(p2, q2))
        #         a2 = self.robot.compute_forward_dynamics(u)
        #
        #     res.append(a2 - a1)
        # self.robot.unpack(orig_pack)
        #
        # return np.array(res).T * self.timestep / 2.0

    def fu(self, x: np.array, u: np.array, eps: float = 1e-3) -> np.array:
        """
            0.5*(a2 - a1)t
        """
        res = []
        u = np.clip(u, self.limits[0], self.limits[1])

        orig_pack = self.robot.pack()
        orig_u = u.tolist()
        for i in range(self.num_x):
            self.robot.unpack(self.pack)
            new_u1 = orig_u.copy()
            new_u2 = orig_u.copy()
            new_u1[i] -= eps
            new_u2[i] += eps

            a1 = self.robot.compute_forward_dynamics(new_u1)
            a2 = self.robot.compute_forward_dynamics(new_u2)

            # bound check due to u
            x1 = x + a1 * self.timestep ** 2 / 2.0
            x2 = x + a2 * self.timestep ** 2 / 2.0

            l1 = self.limits[0]
            l2 = self.limits[1]
            a1 = np.where((x1 >= l1) * (x1 <= l2), a1, np.zeros_like(a1))
            a2 = np.where((x2 >= l1) * (x2 <= l2), a2, np.zeros_like(a2))

            res.append(a2 - a1)

        self.robot.unpack(orig_pack)
        return np.array(res).T * self.timestep / 2.0
