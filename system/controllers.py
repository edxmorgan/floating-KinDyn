import casadi as ca

class RobotControllers():
    def __init__(self, n_joints):
        n = int(n_joints)

        # Symbols
        self.q       = ca.SX.sym("q",      n, 1)
        self.q_dot   = ca.SX.sym("q_dot",  n, 1)
        self.q_ref   = ca.SX.sym("q_ref",  n, 1)
        self.Kp      = ca.SX.sym("Kp",     n, 1)
        self.Ki      = ca.SX.sym("Ki",     n, 1)
        self.Kd      = ca.SX.sym("Kd",     n, 1)
        self.sum_e   = ca.SX.sym("sum_e",  n, 1)
        self.dt      = ca.SX.sym("dt")             # scalar
        self.u_max   = ca.SX.sym("u_max",  n, 1)
        self.u_min   = ca.SX.sym("u_min",  n, 1)

    def build_arm_pid(self):
        """
        Build a single CasADi Function named 'pid' for an n DoF arm.

        Inputs  q, q_dot, q_ref, Kp, Ki, Kd, sum_e, dt, u_max, u_min  all n×1 except dt which is scalar.
        Outputs u_sat, err, sum_e_next.
        """


        # Error and integral
        err         = self.q - self.q_ref
        sum_e_next  = self.sum_e + err * self.dt

        # PID torque
        u_raw = -ca.diag(self.Kp) @ err - ca.diag(self.Kd) @ self.q_dot - ca.diag(self.Ki) @ sum_e_next

        # Saturation
        u_sat = ca.fmin(ca.fmax(u_raw, self.u_min), self.u_max)

        # One function, three outputs
        pid = ca.Function(
            "pid",
            [self.q, self.q_dot, self.q_ref, self.Kp, self.Ki, self.Kd, self.sum_e, self.dt, self.u_max, self.u_min],
            [u_sat, err, sum_e_next],
        )
        return pid

    