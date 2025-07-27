import casadi as ca
import um_dynamics.utils.transformation_matrix as tm
import numpy as np

def motion_cross_product(v):
    """Returns the motion cross product matrix of a spatial vector."""

    mcp = ca.SX.zeros(6, 6)

    mcp[0, 1] = -v[2]
    mcp[0, 2] = v[1]
    mcp[1, 0] = v[2]
    mcp[1, 2] = -v[0]
    mcp[2, 0] = -v[1]
    mcp[2, 1] = v[0]

    mcp[3, 4] = -v[2]
    mcp[3, 5] = v[1]
    mcp[4, 3] = v[2]
    mcp[4, 5] = -v[0]
    mcp[5, 3] = -v[1]
    mcp[5, 4] = v[0]

    mcp[3, 1] = -v[5]
    mcp[3, 2] = v[4]
    mcp[4, 0] = v[5]
    mcp[4, 2] = -v[3]
    mcp[5, 0] = -v[4]
    mcp[5, 1] = v[3]

    return mcp


def force_cross_product(v):
    """Returns the force cross product matrix of a spatial vector."""
    return -motion_cross_product(v).T


def spatial_inertia_matrix_IO(ixx, ixy, ixz, iyy, iyz, izz, mass, c):
    """Returns the 6x6 spatial inertia matrix expressed at the origin in symbolic representation"""
    IO_sym = ca.SX.zeros(6, 6)

    _Ic = ca.SX(3,3)
    _Ic[0, :] = ca.horzcat(ixx, ixy, ixz)
    _Ic[1, :] = ca.horzcat(ixy, iyy, iyz)
    _Ic[2, :] = ca.horzcat(ixz, iyz, izz)

    m_eye = ca.diag(ca.vertcat(mass, mass, mass))

    c_sk = ca.skew(c)
    IO_sym[:3,:3] = _Ic + (mass*(c_sk@c_sk.T))

    IO_sym[3:,3:] = m_eye
    IO_sym[:3,3:] = mass*c_sk
    IO_sym[3:,:3] = mass*c_sk.T

    return IO_sym


def rotation_rpy(roll, pitch, yaw):
    R = ca.SX(3, 3)

    R[0,0] = ca.cos(yaw)*ca.cos(pitch)
    R[0,1] = -ca.sin(yaw)*ca.cos(roll) + ca.cos(yaw)*ca.sin(pitch)*ca.sin(roll)
    R[0,2] = ca.sin(yaw)*ca.sin(roll) + ca.cos(yaw)*ca.cos(roll)*ca.sin(pitch)

    R[1,0] = ca.sin(yaw)*ca.cos(pitch)
    R[1,1] = ca.cos(yaw)*ca.cos(roll) + ca.sin(roll)*ca.sin(pitch)*ca.sin(yaw)
    R[1,2] = -ca.cos(yaw)*ca.sin(roll) + ca.sin(pitch)*ca.sin(yaw)*ca.cos(roll)

    R[2,0] = -ca.sin(pitch)
    R[2,1] = ca.cos(pitch)*ca.sin(roll)
    R[2,2] = ca.cos(pitch)*ca.cos(roll)
    return R

# def spatial_force_transform(R, r):
#     """Returns the spatial force transform from a 3x3 rotation matrix
#     and a 3x1 displacement vector."""
#     X = ca.SX.zeros(6, 6)
#     X[:3, :3] = R.T
#     X[3:, 3:] = R.T
#     X[:3, 3:] = ca.mtimes(ca.skew(r), R.T)
#     return X


def spatial_transform(R, r):
    """Returns the spatial motion transform from a 3x3 rotation matrix
    and a 3x1 displacement vector."""
    X = ca.SX.zeros(6, 6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -ca.mtimes(R, ca.skew(r))
    return X

def XJT_revolute(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a revolute connecting joint."""
    T = tm.revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix, displacement)


def XJT_prismatic(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a prismatic connecting joint."""
    T = tm.prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix, displacement)


def XT(xyz, rpy):
    """Returns a general spatial transformation matrix matrix"""
    rotation_matrix = rotation_rpy(rpy[0], rpy[1], rpy[2])
    return spatial_transform(rotation_matrix, xyz)

def extractEr(i_X_p):
    "returns E and rx"
    E0 = i_X_p[:3,:3]
    E4 = i_X_p[3:,3:] #E0 == E4
    _Er_x = i_X_p[3:,:3]
    rx = -E0.T@_Er_x
    return E0, E4, rx

def spatial_to_homogeneous(X):
    R, _, rx = extractEr(X)
    p = ca.inv_skew(rx)
    T = ca.SX.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T, R, p

def spatial_mtimes(X_i, X_O):
    T_ixp, _, _ = spatial_to_homogeneous(X_i)
    T_pxO, _, _ = spatial_to_homogeneous(X_O)
    T_ix0 = T_pxO@T_ixp
    R = T_ix0[:3, :3]
    r = T_ix0[:3, 3]
    return spatial_transform(R,r)

def inverse_spatial_transform(i_X_p):
    "Returns p_X_i_"
    E0 = i_X_p[:3,:3]
    E4 = i_X_p[3:,3:] #E0 == E4
    Er_x = -i_X_p[3:,:3]

    p_X_i = ca.SX.zeros(6, 6)
    E_T = E0.T
    p_X_i[:3,:3] = E_T
    p_X_i[3:,3:] = E_T

    r_x_E_T = Er_x.T
    p_X_i[3:,:3] = r_x_E_T
    return p_X_i
 
def analytic_to_geometric(Ja, T):
    """
    Convert a 6×n analytic Jacobian (XYZ Euler) to geometric.
    Ja  – analytic Jacobian  (6×n SX/DM)
    """
    # split linear / angular blocks
    Jv = Ja[0:3, :]                  # linear rows stay the same
    Jθ = Ja[3:6, :]                  # Euler‑rate rows → map
    Jω = T @ Jθ                      # angular velocity rows
    return ca.vertcat(Jv, Jω)        # 6×n geometric Jacobian

def rotation_matrix_to_euler(R, order='zyx'):
    """
    Convert a rotation matrix to Euler angles using CasADi symbolic expressions.

    Parameters:
    R (SX or MX): 3x3 rotation matrix
    order (str): Order of Euler angles axes. 
                 Common orders include 'zyx', 'xyz', 'zyz', etc.

    Returns:
    euler (SX or MX): 3x1 vector of Euler angles in radians
    """
    # Validate the order
    # Define supported orders
    supported_orders = [
        'zyx', 'xyz', 'zyz', 'xzx', 'yxy', 'yzy',
        'zxy', 'yxz', 'yzx', 'xyx', 'xzy', 'zyx'
    ]
    if order not in supported_orders:
        raise ValueError(f"Unsupported Euler order '{order}'. Supported orders are {supported_orders}.")

    # Ensure R is a 3x3 matrix
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."

    # Helper function to compute Euler angles for 'zyx' order
    if order == 'zyx':
        # yaw (psi)   : rotation about z-axis
        # pitch (theta): rotation about y-axis
        # roll (phi)  : rotation about x-axis

        # Compute pitch
        pitch = ca.asin(-R[2, 0])

        # To handle the singularity when cos(pitch) is close to zero
        cos_pitch = ca.sqrt(1 - R[2, 0]**2)

        # Define a small threshold to detect singularity
        epsilon = 1e-6

        # Compute yaw and roll
        yaw = ca.if_else(cos_pitch > epsilon,
                        ca.atan2(R[1, 0], R[0, 0]),
                        0)  # When cos(pitch) is near zero, set yaw to zero

        roll = ca.if_else(cos_pitch > epsilon,
                         ca.atan2(R[2, 1], R[2, 2]),
                         ca.atan2(-R[1, 2], R[1, 1]))

        euler = ca.vertcat(roll, pitch, yaw)

    elif order == 'xyz':
        # Compute pitch
        pitch = ca.asin(R[0, 2])

        # Handle singularity
        cos_pitch = ca.sqrt(1 - R[0, 2]**2)
        epsilon = 1e-6

        roll = ca.if_else(cos_pitch > epsilon,
                         ca.atan2(-R[1, 2], R[2, 2]),
                         0)

        yaw = ca.if_else(cos_pitch > epsilon,
                        ca.atan2(-R[0, 1], R[0, 0]),
                        ca.atan2(R[1, 0], R[1, 1]))

        euler = ca.vertcat(roll, pitch, yaw)

    elif order == 'zyz':
        # Compute theta
        theta = ca.acos(R[2, 2])

        # Handle singularity when theta is 0 or pi
        sin_theta = ca.sin(theta)
        epsilon = 1e-6

        phi = ca.if_else(sin_theta > epsilon,
                        ca.atan2(R[2, 0], R[2, 1]),
                        0)

        psi = ca.if_else(sin_theta > epsilon,
                        ca.atan2(R[0, 2], -R[1, 2]),
                        ca.atan2(-R[1, 0], R[0, 0]))

        euler = ca.vertcat(phi, theta, psi)

    else:
        raise NotImplementedError(f"Euler order '{order}' is not implemented yet.")

    # Normalize angles to be within [-pi, pi]
    euler = ca.fmod(euler + ca.pi, 2 * ca.pi) - ca.pi

    return euler