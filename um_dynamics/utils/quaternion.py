"""Functions for getting casadi expressions for quaternions from joint type."""
import casadi as ca
import numpy as np


def revolute(xyz, rpy, axis, qi):
    """Gives a casadi function for the quaternion. [xyz, w] form."""
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = ca.cos(roll/2.0)
    sr = ca.sin(roll/2.0)
    cp = ca.cos(pitch/2.0)
    sp = ca.sin(pitch/2.0)
    cy = ca.cos(yaw/2.0)
    sy = ca.sin(yaw/2.0)

    # The quaternion associated with the origin rotation
    # Note: quat = [ xyz, w], where w is the scalar part
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    q_or = [x_or, y_or, z_or, w_or]
    # Joint rotation from axis angle
    cqi = ca.cos(qi/2.0)
    sqi = ca.sin(qi/2.0)
    x_j = axis[0]*sqi
    y_j = axis[1]*sqi
    z_j = axis[2]*sqi
    w_j = cqi
    q_j = [x_j, y_j, z_j, w_j]
    # Resulting quaternion
    return product(q_or, q_j)


def product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = ca.SX.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat


def numpy_rpy(roll, pitch, yaw):
    """Returns a quaternion ([x,y,z,w], w scalar) from roll pitch yaw ZYX
    convention."""
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    # Remember to normalize:
    nq = np.sqrt(x*x + y*y + z*z + w*w)
    return np.array([x/nq,
                     y/nq,
                     z/nq,
                     w/nq])


def numpy_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = np.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat


def numpy_ravani_roth_dist(q1, q2):
    """Quaternion distance designed by ravani and roth.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))


def numpy_inner_product_dist(q1, q2):
    """Quaternion distance based on innerproduct.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return 1.0 - abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])


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