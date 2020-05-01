import jax.numpy as np
import jax
import numpy as onp
import sapien.core as sapien

links_cache = {}
joints_cache = {}


def get_links(robot, sig_mass=-1):
    '''
        Return all the link that have significant mass for the robot
    '''
    if robot in links_cache:
        return links_cache[robot]

    links = {}

    for l in robot.get_links():
        name = l.get_name()
        mass = l.get_mass()
        if mass > sig_mass:
            links[name] = l

    links_cache[robot] = links
    return links


def get_joints(robot):
    '''
        Return all the joints
    '''
    if robot in joints_cache:
        return joints_cache[robot]

    joints = {}

    for j in robot.get_joints():
        if j.get_dof() > 0:
            name = j.get_name()
            joints[name] = j

    joints_cache[robot] = joints
    return joints


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


def get_link_state(robot):
    '''
        Input:
            robot

        Output:
            array of shape (n_l, 6)
    '''
    links = get_links(robot)
    state = []

    for name, l in links.items():
        state.append(np.concatenate((l.get_pose().p, l.get_velocity())))

    return np.array(state)


def get_joint_state(robot):
    '''
        Input:
            robot

        Output:
            array of shape (n_j, 7)
    '''
    joints = get_joints(robot)
    state = []

    for name, j in joints.items():
        pose = j.get_global_pose()
        state.append(np.concatenate((pose.p, pose.q)))

    return np.array(state)


def is_PD(m):
    return onp.alltrue(onp.linalg.eigvals(m) > 0)


def get_state(robot: sapien.Articulation):
    '''
        dof + 3 + 4
        [qpos(8), pos(3), quat(4)]
    '''
    # ant_pos = robot.get_pose()
    # return np.concatenate((robot.get_qpos(), ant_pos.p, ant_pos.q))
    return onp.concatenate((robot.get_qpos(), robot.get_qvel()))


def inbound(x, l1, l2):
    return np.alltrue(x > l1) and onp.alltrue(x < l2)


def compose_p(pose):
    '''
        Get the affine matrix from pose
    '''

    return compose(pose.p, pose.q)


def m_quat2mat(q):
    '''
        Borrowed from transform3d, altered a bit
    '''
    float_eps = 2.220446049250313e-16

    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    #     if Nq < float_eps:
    #         return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X;
    wY = w * Y;
    wZ = w * Z
    xX = x * X;
    xY = x * Y;
    xZ = x * Z
    yY = y * Y;
    yZ = y * Z;
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def m_mat2quat(M):
    '''
        Borrowed from transform3d, altered a bit
    '''
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flatten()
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0, 0, 0],
        [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
        [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]
    ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    #     if q[0] < 0:
    #         q *= -1
    return q * q[0] / np.abs(q[0])


def m_axangle2mat(axis, angle):
    '''
        Borrowed from transform3d
    '''
    x, y, z = axis
    n = np.sqrt(x * x + y * y + z * z)
    x = x / n
    y = y / n
    z = z / n
    c = np.cos(angle);
    s = np.sin(angle);
    C = 1 - c
    xs = x * s;
    ys = y * s;
    zs = z * s
    xC = x * C;
    yC = y * C;
    zC = z * C
    xyC = x * yC;
    yzC = y * zC;
    zxC = z * xC
    return np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])


def decompose(m):
    '''
        Get the pose
    '''
    p = m[:3, -1]
    q = m_mat2quat(m[:3, :3])
    #     q = quat.mat2quat(m[:3,:3])

    return np.concatenate((p, q))


def check_val(x):
    return not np.alltrue(np.isfinite(x))


def decompose_pos_only(m):
    """
        Get the positino
    """
    p = m[:3, -1]
    #     q = quat.mat2quat(m[:3,:3])

    return p


def compose(p, q):
    '''
        Get the affine matrix from pose
    '''
    rot = m_quat2mat(q)
    #     rot = quat.quat2mat(q)

    affine_matrix = np.eye(4)

    affine_matrix = jax.ops.index_update(affine_matrix, jax.ops.index[:3, :3], rot)
    affine_matrix = jax.ops.index_update(affine_matrix, jax.ops.index[:3, -1], p)

    return np.array(affine_matrix)
