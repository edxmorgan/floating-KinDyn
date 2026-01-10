# controllers.py
import casadi as ca
from system.robot import RobotDynamics
class RobotControllers:
    def __init__(self, n_joints: int, model: RobotDynamics, has_endeffector: bool=False):
        n = int(n_joints)
        cm_parms, m_params, I_params, fv_coeff, fc_coeff, fs_coeff, v_s_coeff, vec_g, r_com_payload, m_p, q, q_dot, q_dotdot, tau, base_pose, world_pose, tip_offset_pose = model.kinematic_dict['parameters']

        self.g_ff = model.id_g # feedforward term

        # Symbols, all vectors are n×1 and column shaped
        self.q       = q
        self.q_dot   = q_dot

        if has_endeffector:
            grasper_q    = ca.SX.sym("grasper_q", 1, 1)
            grasper_qdot = ca.SX.sym("grasper_qdot", 1, 1)
            self.q = ca.vertcat(q, grasper_q)
            self.q_dot = ca.vertcat(q_dot, grasper_qdot)
            self.g_ff = ca.vertcat(self.g_ff, 0)
            n = n+1

        self.q_ref   = ca.SX.sym("q_ref",  n, 1)  # reference position

        self.Kp      = ca.SX.sym("Kp",     n, 1)
        self.Ki      = ca.SX.sym("Ki",     n, 1)
        self.Kd      = ca.SX.sym("Kd",     n, 1)

        self.sum_e   = ca.SX.sym("sum_e",  n, 1)  # integral buffer
        self.dt      = ca.SX.sym("dt")            # scalar step

        self.u_max   = ca.SX.sym("u_max",  n, 1)
        self.u_min   = ca.SX.sym("u_min",  n, 1)

        self.sim_params = ca.vertcat(model._sys_id_coeff["masses_id_syms_vertcat"],
                    model._sys_id_coeff["first_moments_id_vertcat"], 
                    model._sys_id_coeff["inertias_id_vertcat"],
                    model._sys_id_coeff["fv_id_vertcat"],
                    model._sys_id_coeff["fc_id_vertcat"], 
                    model._sys_id_coeff["fs_id_vertcat"],
                    v_s_coeff, vec_g, r_com_payload, m_p, base_pose, world_pose)

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
            self.g_ff # feedforward term
            + ca.diag(self.Kp) @ err # proportional term
            + ca.diag(self.Ki) @ sum_e_next # integral term
            - ca.diag(self.Kd) @ (self.q_dot) # derivative term
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
                self.u_max,
                self.u_min,
                self.sim_params
            ],
            [u_sat, err, sum_e_next],
        )
        return pid
