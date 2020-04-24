from . import misc, ModelDerivator
import jax.numpy as np
import jax
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
        self.name = joint.get_name()

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
        self.jointTree = {}

        joint_dict = {}
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

    @jax.partial(jax.jit, static_argnums=(0,))
    def fk(self, state, ifDict=False):
        if self.root is None:
            return None

        bPose = state[-7:]
        qPos = state[:-7]

        res = {}
        base_mat = misc.compose(bPose[:3], bPose[3:])

        self.fk_recurse(base_mat, self.root, qPos, res)

        if ifDict:
            return res
        else:
            return np.array(list(res.values()))


class NumForwardDynamicsDer(ModelDerivator):

    def __init__(self, robot: sapien.Articulation, timestep: float, def_eps=1e-3, include_external=False):
        self.robot = robot
        self.timestep = timestep
        self.def_eps = def_eps
        self.num_x = self.robot.dof
        self.num_u = len(misc.get_state(robot))
        self.dof = self.robot.dof
        self.pack = self.robot.pack()

    def set_pack(self, pack):
        self.pack = pack

    def fx(self, u: np.array, x: np.array, eps: float = 1e-3) -> np.array:
        """
            0.5*(a2 - a1)t
        """

        return np.eye(self.num_x)
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

    def fu(self, u: np.array, x: np.array, eps: float = 1e-3) -> np.array:
        """
            0.5*(a2 - a1)t
        """
        res = []

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

            res.append(a2 - a1)

        self.robot.unpack(orig_pack)
        return np.array(res).T * self.timestep / 2.0