# controllers.py
import casadi as ca

class RobotControllers:
    def __init__(self, n_joints: int):
        n = int(n_joints)

        # Symbols, all vectors are n×1 and column shaped
        self.q       = ca.SX.sym("q",      n, 1)  # actual position
        self.q_dot   = ca.SX.sym("q_dot",  n, 1)  # actual velocity
        self.q_ref   = ca.SX.sym("q_ref",  n, 1)  # reference position

        self.Kp      = ca.SX.sym("Kp",     n, 1)
        self.Ki      = ca.SX.sym("Ki",     n, 1)
        self.Kd      = ca.SX.sym("Kd",     n, 1)

        self.sum_e   = ca.SX.sym("sum_e",  n, 1)  # integral buffer
        self.dt      = ca.SX.sym("dt")            # scalar step

        self.g_ff    = ca.SX.sym("g_ff",   n, 1)  # gravity feedforward, torque units

        self.u_max   = ca.SX.sym("u_max",  n, 1)
        self.u_min   = ca.SX.sym("u_min",  n, 1)

    def build_arm_pid(self):
        """
        Returns a single CasADi Function named 'pid' for an n DoF arm.

        Inputs
            q, q_dot, q_ref, Kp, Ki, Kd, sum_e, dt, g_ff, u_max, u_min
            All vectors are n×1, dt is scalar.

        Outputs
            u_sat, err, sum_e_next
        """
        # Reference minus actual so gains appear positive
        err = self.q_ref - self.q

        # Integrator update
        sum_e_next = self.sum_e + err * self.dt

        # PID with positive gains, velocity target is zero
        u_raw = (
            ca.diag(self.Kp) @ err
            + ca.diag(self.Kd) @ (self.q_dot)
            + ca.diag(self.Ki) @ sum_e_next
            + self.g_ff
        )

        # Elementwise saturation
        u_sat = ca.fmin(ca.fmax(u_raw, self.u_min), self.u_max)

        pid = ca.Function(
            "pid",
            [
                self.q,
                self.q_dot,
                self.q_ref,
                self.Kp,
                self.Ki,
                self.Kd,
                self.sum_e,
                self.dt,
                self.g_ff,
                self.u_max,
                self.u_min,
            ],
            [u_sat, err, sum_e_next],
        )
        return pid
