"""Functions for getting casadi expressions for transformation matrices from
joint type."""
import casadi as ca

def rot_from_rpy(roll, pitch, yaw):
    cr, sr = ca.cos(roll),  ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw),   ca.sin(yaw)
    Rz = ca.vertcat(
        ca.hcat([cy, -sy, 0]),
        ca.hcat([sy,  cy, 0]),
        ca.hcat([0,    0,  1]),
    )
    Ry = ca.vertcat(
        ca.hcat([ cp, 0, sp]),
        ca.hcat([  0, 1,  0]),
        ca.hcat([-sp, 0, cp]),
    )
    Rx = ca.vertcat(
        ca.hcat([1,  0,  0]),
        ca.hcat([0, cr, -sr]),
        ca.hcat([0, sr,  cr]),
    )
    return Rz @ Ry @ Rx  # URDF uses RPY about fixed axes Z, Y, X

def T_from_xyz_rpy(xyz, rpy):
    R = rot_from_rpy(rpy[0], rpy[1], rpy[2])
    t = ca.vertcat(xyz[0], xyz[1], xyz[2])
    T = ca.SX.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

def vee(omega_hat):
    return ca.vertcat(
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0]
    )

def T_revolute(xyz, rpy, axis, qval):
    # parent to joint origin
    T_pj = T_from_xyz_rpy(xyz, rpy)
    # rotation about joint axis in joint frame
    ax = ca.vertcat(axis[0], axis[1], axis[2])
    ax = ax / ca.norm_2(ax)
    # Rodrigues' rotation formula
    K = ca.skew(ax)
    I3 = ca.SX.eye(3)
    Rq = I3 + ca.sin(qval) * K + (1 - ca.cos(qval)) * (K @ K)
    T_q = ca.SX.eye(4)
    T_q[0:3, 0:3] = Rq
    return T_pj @ T_q

def T_prismatic(xyz, rpy, axis, qval):
    # parent to joint origin
    T_pj = T_from_xyz_rpy(xyz, rpy)
    ax = ca.vertcat(axis[0], axis[1], axis[2])
    ax = ax / ca.norm_2(ax)
    t = qval * ax
    T_q = ca.SX.eye(4)
    T_q[0:3, 3] = t
    return T_pj @ T_q

def euler_to_spatial_rate_T(phi, theta, psi):
    sφ, cφ = ca.sin(phi),  ca.cos(phi)
    sθ, cθ = ca.sin(theta), ca.cos(theta)

    row1 = ca.horzcat(1, 0, sθ)
    row2 = ca.horzcat(0, cφ, -sφ*cθ)
    row3 = ca.horzcat(0, sφ,  cφ*cθ)
    return ca.vertcat(row1, row2, row3)

def euler_to_body_rate_T(phi, theta, psi):
    sψ, cψ = ca.sin(psi),   ca.cos(psi)
    sθ, cθ = ca.sin(theta), ca.cos(theta)

    row1 = ca.horzcat( cθ*cψ,  sψ,  0 )
    row2 = ca.horzcat(-cθ*sψ,  cψ,  0 )
    row3 = ca.horzcat(     sθ,   0,   1 )
    return ca.vertcat(row1, row2, row3)

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


def rotation_matrix_to_quaternion(R, order='wxyz'):
    """
    Convert a rotation matrix to a quaternion using CasADi symbolic expressions.

    Parameters:
    R (ca.SX or ca.MX): 3x3 rotation matrix
    order (str): Order of quaternion components. 
                 'wxyz' for [w, x, y, z],
                 'xyzw' for [x, y, z, w].

    Returns:
    q (ca.SX or ca.MX): 4x1 quaternion in the specified order
    """
    # Validate the order
    valid_orders = ['wxyz', 'xyzw']
    if order not in valid_orders:
        raise ValueError(f"Invalid order '{order}'. Supported orders are {valid_orders}.")

    # Ensure R is a 3x3 matrix
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."

    # Compute the trace of the matrix
    trace = R[0,0] + R[1,1] + R[2,2]

    # Case 1: Trace > 0
    cond1 = trace > 0
    S1 = ca.sqrt(trace + 1.0) * 2  # S = 4*q0 (w)
    q0_expr1 = 0.25 * S1
    q1_expr1 = (R[2,1] - R[1,2]) / S1
    q2_expr1 = (R[0,2] - R[2,0]) / S1
    q3_expr1 = (R[1,0] - R[0,1]) / S1

    # Case 2: R00 is the largest diagonal element
    cond2 = ca.logic_and(R[0,0] > R[1,1], R[0,0] > R[2,2])
    S2 = ca.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # S=4*q1 (x)
    q0_expr2 = (R[2,1] - R[1,2]) / S2
    q1_expr2 = 0.25 * S2
    q2_expr2 = (R[0,1] + R[1,0]) / S2
    q3_expr2 = (R[0,2] + R[2,0]) / S2

    # Case 3: R11 is the largest diagonal element
    cond3 = ca.logic_and(R[1,1] > R[0,0], R[1,1] > R[2,2])
    S3 = ca.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # S=4*q2 (y)
    q0_expr3 = (R[0,2] - R[2,0]) / S3
    q1_expr3 = (R[0,1] + R[1,0]) / S3
    q2_expr3 = 0.25 * S3
    q3_expr3 = (R[1,2] + R[2,1]) / S3

    # Case 4: R22 is the largest diagonal element
    # No explicit condition needed; it's the else case
    S4 = ca.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # S=4*q3 (z)
    q0_expr4 = (R[1,0] - R[0,1]) / S4
    q1_expr4 = (R[0,2] + R[2,0]) / S4
    q2_expr4 = (R[1,2] + R[2,1]) / S4
    q3_expr4 = 0.25 * S4

    # Compute each component using conditional selection
    # If cond1 is true, use case1 expressions
    # Else, if cond2 is true, use case2 expressions
    # Else, if cond3 is true, use case3 expressions
    # Else, use case4 expressions

    q0 = ca.if_else(cond1, q0_expr1,
                   ca.if_else(cond2, q0_expr2,
                             ca.if_else(cond3, q0_expr3, q0_expr4)))
    q1 = ca.if_else(cond1, q1_expr1,
                   ca.if_else(cond2, q1_expr2,
                             ca.if_else(cond3, q1_expr3, q1_expr4)))
    q2 = ca.if_else(cond1, q2_expr1,
                   ca.if_else(cond2, q2_expr2,
                             ca.if_else(cond3, q2_expr3, q2_expr4)))
    q3 = ca.if_else(cond1, q3_expr1,
                   ca.if_else(cond2, q3_expr2,
                             ca.if_else(cond3, q3_expr3, q3_expr4)))

    # Assemble the quaternion based on the desired order
    if order == 'wxyz':
        q = ca.vertcat(q0, q1, q2, q3)  # [w, x, y, z]
    elif order == 'xyzw':
        q = ca.vertcat(q1, q2, q3, q0)  # [x, y, z, w]

    # Normalize the quaternion to ensure it's a unit quaternion
    q = q / ca.norm_2(q)

    return q