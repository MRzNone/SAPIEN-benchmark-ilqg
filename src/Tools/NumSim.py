from . import misc as misc
import jax.numpy as np
import jax
import sapien.core as sapien


class JointInfo:
    def __init__(self, joint:sapien.Joint):
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
    def __init__(self, robot:sapien.Articulation):
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

            joint_dict[j.get_child_link()]  = info
            if self.root is None :
                self.root = info

        # form graph
        for j in robot.get_joints():
            if j.get_parent_link() is None:
                continue

            parentInfo = joint_dict[j.get_parent_link()]

            parentInfo.add_child(joint_dict[j.get_child_link()])

    def fk_recurse(self, pMat, joint:JointInfo, qPos: list, res: dict):
        # get theta
        theta = 0.0
        if joint.qId is not None :
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