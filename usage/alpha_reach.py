import numpy as np
import casadi as cs
import itertools

class Params:
    root = "base_link"
    tip = "alpha_standard_jaws_base_link"

    relative_urdf_path = f"/resources/urdf/alpha_5_robot.urdf"

    joint_min = np.array([1.00, 0.01, 0.01, 0.01])
    joint_max = np.array([5.50, 3.40, 3.40, 5.70])

    joint_limits = list(zip(joint_min.tolist(), joint_max.tolist()))
    joint_limits_configurations = np.array(list(itertools.product(*joint_limits)))

    u_min = np.array([-1.5, -1, -1, -0.54])
    u_max = np.array([1.5, 1, 1, 0.54])
    
    Kp = cs.vertcat(10.0, 10.0, 10.0, 2.0)
    Ki = cs.vertcat(0.0, 0.0, 0.0, 0.0)
    Kd = cs.vertcat(1.0, 1.0, 1.0, 1.0)

    gravity = 9.81

    base_T0_new = [0.190, 0.000, -0.120, 3.141592653589793, 0.000, 0.000]
    baumgarte_alpha = 200
    lock_mask = cs.vertcat(0.0, 0.0, 0.0, 0.0)
    sim_p = cs.vertcat(
        1.94000000e-01, 4.29000000e-01, 1.14999999e-01, 3.32999998e-01,
        -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -4.29000003e-02,
        1.96649101e-02, 4.29000003e-02, 2.88077923e-03, 7.23516749e-03,
        9.16434754e-03, 2.16416476e-03, -1.19076924e-03, 8.07346553e-03,
        7.10109586e-01, 7.10109586e-01, 1.99576149e-06, -0.00000000e+00,
        -0.00000000e+00, -0.00000000e+00, 1.10178508e-01, 1.83331277e-01,
        1.04292121e-01, -3.32240937e-02, -8.30350362e-02, -3.83631263e-02,
        1.18956416e-01, 1.22363853e-01, 4.34411664e-03, -3.96112974e-04,
        -2.13904668e-02, -1.77228242e-03, 1.92510932e-02, 2.56548460e-02,
        7.17220917e-03, 1.48789886e-03, 4.53687373e-04, -1.09861913e-03,
        2.39569756e+00, 2.23596482e+00, 8.19671021e-01, 3.57249665e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
        0, 0, 0, 0,
        0, 0, gravity,
        0, 0, 0, 0,
        0.19, 0, -0.12, 3.14159, 0, 0,
        0, 0, 0, 0, 0, 0
    )

    sim_n = 20
    delta_t = 0.035
    N = int(sim_n / delta_t)

    # DH like constants for analytic IK
    a0 = 20e-3
    a1 = cs.sqrt(40.0**2 + 154.3**2) * 1e-3
    a2 = 20e-3
    a3 = 0.0
    a4 = 0.0

    d0 = 46.2e-3
    d1 = 0.0
    d2 = 0.0
    d3 = -180e-3
    d4 = 0.0

    @staticmethod
    def casadi_clip(x, lo, hi):
        return cs.fmin(cs.fmax(x, lo), hi)

    @staticmethod
    def build_analytic_ik_casadi():
        # position of the end effector in base frame
        p_base = cs.SX.sym("p_base", 3)
        x = p_base[0]
        y = p_base[1]
        z = p_base[2]

        a0 = Params.a0
        a1 = Params.a1
        a2 = Params.a2
        d0 = Params.d0
        d3 = Params.d3

        l1 = a1
        l2 = cs.sqrt(a2**2 + d3**2)

        R = cs.sqrt(x**2 + y**2)

        thet0 = cs.atan2(y, x) + cs.pi
        l3 = cs.sqrt((R - a0)**2 + (z - d0)**2)

        arg1 = (l1**2 + l2**2 - l3**2) / (2 * l1 * l2)
        arg1 = Params.casadi_clip(arg1, -1.0, 1.0)
        term1 = cs.acos(arg1)

        term2 = cs.asin(Params.casadi_clip((2 * a2) / l1, -1.0, 1.0))
        term3 = cs.asin(Params.casadi_clip(a2 / l2, -1.0, 1.0))

        thet2 = term1 - term2 - term3

        arg2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
        arg2 = Params.casadi_clip(arg2, -1.0, 1.0)
        term4 = cs.acos(arg2)

        thet1 = (cs.pi / 2.0) + cs.atan2(z - d0, R - a0) - term4 - term2

        q012 = cs.vertcat(thet0, thet1, thet2)
        q012_fun = cs.Function("ik_analytic_base", [p_base], [q012])
        return q012_fun, q012, p_base
