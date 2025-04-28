"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
import casadi as cs
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import kinodyn_um.utils.transformation_matrix as T
import kinodyn_um.utils.plucker as plucker
import kinodyn_um.utils.quaternion as quaternion
import kinodyn_um.utils.dual_quaternion as dual_quaternion


class URDFparser(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]
    func_opts = {}
    jit_func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}
    # OS/CPU dependent specification of compiler
    if system().lower() == "darwin" or machine().lower() == "aarch64":
        jit_func_opts["compiler"] = "shell"

    def __init__(self, func_opts=None, use_jit=True):
        self.robot_desc = None
        if func_opts:
            self.func_opts = func_opts
        if use_jit:
            # NOTE: use_jit=True requires that CasADi is built with Clang
            for k, v in self.jit_func_opts.items():
                self.func_opts[k] = v

    def from_file(self, filename):
        """Uses an URDF file to get robot description."""
        self.robot_desc = URDF.from_xml_file(filename)

    def get_joint_info(self, root, tip):
        """Using an URDF to extract joint information, i.e list of
        joints, actuated names and upper and lower limits."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        joint_list = []
        upper = []
        lower = []
        actuated_names = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                joint_list += [joint]
                if joint.type in self.actuated_types:
                    actuated_names += [joint.name]
                    if joint.type == "continuous":
                        upper += [cs.inf]
                        lower += [-cs.inf]
                    else:
                        upper += [joint.limit.upper]
                        lower += [joint.limit.lower]
                    if joint.axis is None:
                        joint.axis = [1., 0., 0.]
                    if joint.origin is None:
                        joint.origin = Pose(xyz=[0., 0., 0.],
                                            rpy=[0., 0., 0.])
                    elif joint.origin.xyz is None:
                        joint.origin.xyz = [0., 0., 0.]
                    elif joint.origin.rpy is None:
                        joint.origin.rpy = [0., 0., 0.]

        return joint_list, actuated_names, upper, lower


    def get_n_joints(self, root, tip):
        """Returns number of actuated joints."""

        chain = self.robot_desc.get_chain(root, tip)
        n_actuated = 0

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    n_actuated += 1

        return n_actuated


    def approximate_workspace(self, root, tip, joint_limits, base_T, floating_base=False, num_samples=100000):
        """
        fk_func: a function that takes a vector of joint angles
                and returns the [x, y, z] end-effector position.
        joint_limits: list of (min_angle, max_angle) for each joint.
        num_samples: how many random samples to draw in joint space.
        """
        n_joints = self.get_n_joints(root, tip)
        baseT_xyz = cs.SX.sym('T_xyz', 3) # manipulator-vehicle mount link xyz origin 
        baseT_rpy = cs.SX.sym('T_rpy', 3) # manipulator-vehicle mount link rpy origin
        base_T_sym = cs.vertcat(baseT_rpy, baseT_xyz) # transform from origin to 1st child

        q = cs.SX.sym("q", n_joints)
        x = cs.SX.sym('x')
        y = cs.SX.sym('y')
        z = cs.SX.sym('z')
        tr_n = cs.vertcat(x, y, z) # x, y ,z of uv wrt to ned origin
        thet = cs.SX.sym('thet')
        phi = cs.SX.sym('phi')
        psi = cs.SX.sym('psi')
        eul = cs.vertcat(phi, thet, psi)  # NED euler angular velocity
        p_n = cs.vertcat(tr_n, eul) # ned total states
        ned_pose_sym = cs.vertcat(p_n, q) #NED position

        i_X_0fs = self.forward_kinematics(root, tip, floating_base = floating_base)
        i_X_0s = i_X_0fs(q, tr_n, eul, baseT_xyz, baseT_rpy)
        H4 , R4, p4 = plucker.spatial_to_homogeneous(i_X_0s[-1])
        T4_euler = cs.vertcat(p4, plucker.rotation_matrix_to_euler(R4, order='xyz'))
        internal_fk_eval_euler = cs.Function("internal_fkeval_euler", [ned_pose_sym, base_T_sym], [T4_euler])
        
        # Arrays to track min/max
        min_pos = np.array([np.inf, np.inf, np.inf])
        max_pos = -np.array([np.inf, np.inf, np.inf])

        # collect points for plotting
        positions = np.zeros((num_samples, 3))

        for i in range(num_samples):
            # Sample joint angles
            thetas = [
                np.random.uniform(low=joint_limits[j][0], 
                                high=joint_limits[j][1])
                for j in range(n_joints)
            ]

            # Forward kinematics: end-effector position
            config = cs.vertcat([0,0,0,0,0,0], thetas)
            # config= cs.vertcat([0, 0, 0, 0, 0, 0, 1.07164, 2.05017, 1.065, 0.926672])
            # print(f'configuration sent: {config}')
            pose = internal_fk_eval_euler(config, base_T)
            # print(f'pose rececievetn: {pose}')
            x = pose[0]
            y = pose[1]
            z = pose[2]

            # Store point for later plotting
            positions[i] = [float(x), float(y), float(z)]

            # Update min/max
            if x < min_pos[0]: min_pos[0] = x
            if x > max_pos[0]: max_pos[0] = x
            if y < min_pos[1]: min_pos[1] = y
            if y > max_pos[1]: max_pos[1] = y
            if z < min_pos[2]: min_pos[2] = z
            if z > max_pos[2]: max_pos[2] = z

        # min_pos, max_pos define an axis-aligned bounding box
        return min_pos, max_pos, positions

    def _tool_Jq(self, root, tip):
        """Internal the forward kinematics as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)

        i_X_p, Si, Ic , tip_ofs = self._model_calculation(root, tip, q)

        for i in range(0, n_joints):
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
            else:
                i_X_0 = i_X_p[i]

        endeff_X_0 = plucker.spatial_mtimes( tip_ofs , i_X_0)

        H_eff , R_eff, p_eff = plucker.spatial_to_homogeneous(endeff_X_0)
        Fk_eff = cs.vertcat(plucker.rotation_matrix_to_euler(R_eff, order='xyz'), p_eff)

        J = cs.jacobian(Fk_eff, q)

        J_tool = cs.Function("J_tool", [q], [J] , self.func_opts)
        return J_tool


    def forward_kinematics(self, root, tip, floating_base = False):
        """Returns the inverse dynamics as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        baseT_xyz = cs.SX.sym('T_xyz', 3) # manipulator-vehicle mount link xyz origin 
        baseT_rpy = cs.SX.sym('T_rpy', 3) # manipulator-vehicle mount link rpy origin

        x = cs.SX.sym('x')
        y = cs.SX.sym('y')
        z = cs.SX.sym('z')
        tr_n = cs.vertcat(x, y, z) # x, y ,z of uv wrt to ned origin
        thet = cs.SX.sym('thet')
        phi = cs.SX.sym('phi')
        psi = cs.SX.sym('psi')
        eul = cs.vertcat(phi, thet, psi)  # NED euler angular velocity

        i_X_p, Si, Ic , tip_ofs = self._model_calculation(root, tip, q)
        T_Base = plucker.XT(baseT_xyz, baseT_rpy)

        i_X_0s = []
        for i in range(0, n_joints):
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
            else:
                if floating_base:
                    NED_0_ = plucker.XT(tr_n, eul)
                    T_Base_X_NED_0 = plucker.spatial_mtimes(T_Base, NED_0_)
                    i_X_0 = plucker.spatial_mtimes(i_X_p[i],T_Base_X_NED_0)
                else:
                    i_X_0 = i_X_p[i]
            
            i_X_0s.append(i_X_0)  # transformation of joint i wrt origin 0

        forward_kin = plucker.spatial_mtimes( tip_ofs , i_X_0)
        i_X_0s.append(forward_kin)
        i_X_f = cs.Function("i_X_f", [q, tr_n, eul, baseT_xyz, baseT_rpy], i_X_0s , self.func_opts)
        return i_X_f

    def _model_calculation(self, root, tip, q):
        """Calculates and returns model information needed in the
        dynamics algorithms caluculations, i.e transforms, joint space
        and inertia."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        chain = self.robot_desc.get_chain(root, tip)
        spatial_inertias = []
        i_X_0 = []
        i_X_p = []
        Sis = []
        tip_offset = cs.DM_eye(6)
        prev_joint = None
        n_actuated = 0
        i = 0

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]

                if joint.type == "fixed":
                    if prev_joint == "fixed":
                        XT_prev = cs.mtimes(
                            plucker.XT(joint.origin.xyz, joint.origin.rpy),
                            XT_prev)
                    else:
                        XT_prev = plucker.XT(
                            joint.origin.xyz,
                            joint.origin.rpy)
                    inertia_transform = XT_prev
                    prev_inertia = spatial_inertia

                elif joint.type == "prismatic":
                    if n_actuated != 0:
                        spatial_inertias.append(spatial_inertia)
                    n_actuated += 1
                    XJT = plucker.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q[i])
                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)
                    Si = cs.SX([0, 0, 0,
                                joint.axis[0],
                                joint.axis[1],
                                joint.axis[2]])
                    i_X_p.append(XJT)
                    Sis.append(Si)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    if n_actuated != 0:
                        spatial_inertias.append(spatial_inertia)
                    n_actuated += 1

                    XJT = plucker.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q[i])
                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)
                    Si = cs.SX([
                                joint.axis[0],
                                joint.axis[1],
                                joint.axis[2],
                                0,
                                0,
                                0])
                    i_X_p.append(XJT)
                    Sis.append(Si)
                    i += 1

                prev_joint = joint.type
                if joint.child == tip:
                    if joint.type == "fixed":
                        tip_offset = XT_prev

            if item in self.robot_desc.link_map:
                link = self.robot_desc.link_map[item]

                if link.inertial is None:
                    spatial_inertia = np.zeros((6, 6))
                else:
                    I = link.inertial.inertia
                    spatial_inertia = plucker.spatial_inertia_matrix_IO(
                        I.ixx,
                        I.ixy,
                        I.ixz,
                        I.iyy,
                        I.iyz,
                        I.izz,
                        link.inertial.mass,
                        link.inertial.origin.xyz)

                if prev_joint == "fixed":
                    spatial_inertia = prev_inertia + cs.mtimes(
                        inertia_transform.T,
                        cs.mtimes(spatial_inertia, inertia_transform))

                if link.name == tip:
                    spatial_inertias.append(spatial_inertia)

        return i_X_p, Sis, spatial_inertias, tip_offset

    def _apply_external_forces(self, external_f, f, i_X_p):
        """Internal function for applying external forces in dynamics
        algorithms calculations."""
        for i in range(0, len(f)):
            f[i] -= cs.mtimes(i_X_p[i].T, external_f[i])
        return f

    def get_inverse_dynamics_rnea(self, root, tip,
                                  gravity=None, f_ext=None):
        """Returns the inverse dynamics as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        q_dot = cs.SX.sym("q_dot", n_joints)
        q_ddot = cs.SX.sym("q_ddot", n_joints)
        i_X_p, Si, Ic, tip_ofs = self._model_calculation(root, tip, q)

        v = []
        a = []
        f = []
        tau = cs.SX.zeros(n_joints)

        for i in range(0, n_joints):
            vJ = cs.mtimes(Si[i], q_dot[i])
            if i == 0:
                v.append(vJ)
                if gravity is not None:
                    ag = np.array([0.,
                                   0.,
                                   0.,
                                   gravity[0],
                                   gravity[1],
                                   gravity[2]])
                    a.append(
                        cs.mtimes(i_X_p[i], -ag) + cs.mtimes(Si[i], q_ddot[i]))
                else:
                    a.append(cs.mtimes(Si[i], q_ddot[i]))
            else:
                v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
                a.append(
                    cs.mtimes(i_X_p[i], a[i-1])
                    + cs.mtimes(Si[i], q_ddot[i])
                    + cs.mtimes(plucker.motion_cross_product(v[i]), vJ))

            f.append(
                cs.mtimes(Ic[i], a[i])
                + cs.mtimes(
                    plucker.force_cross_product(v[i]),
                    cs.mtimes(Ic[i], v[i])))

        # if f_ext is not None:
        #     f = self._apply_external_forces(f_ext, f, i_X_p)

        for i in range(n_joints-1, -1, -1):
            tau[i] = cs.mtimes(Si[i].T, f[i])
            if i != 0:
                f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

        tau = cs.Function("C", [q, q_dot, q_ddot], [tau], self.func_opts)
        return tau

    def get_gravity_rnea(self, root, tip, gravity):
        """Returns the gravitational term as a casadi function."""

        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        i_X_p, Si, Ic , tip_ofs = self._model_calculation(root, tip, q)

        v = []
        a = []
        ag = cs.SX([0., 0., 0., gravity[0], gravity[1], gravity[2]])
        f = []
        tau = cs.SX.zeros(n_joints)

        for i in range(0, n_joints):
            if i == 0:
                a.append(cs.mtimes(i_X_p[i], -ag))
            else:
                a.append(cs.mtimes(i_X_p[i], a[i-1]))
            f.append(cs.mtimes(Ic[i], a[i]))

        for i in range(n_joints-1, -1, -1):
            tau[i] = cs.mtimes(Si[i].T, f[i])
            if i != 0:
                f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

        tau = cs.Function("C", [q], [tau],
                          self.func_opts)
        return tau

    def _get_M(self, Ic, i_X_p, Si, n_joints, q):
        """Internal function for calculating the inertia matrix."""
        M = cs.SX.zeros(n_joints, n_joints)
        Ic_composite = [None]*len(Ic)

        for i in range(0, n_joints):
            Ic_composite[i] = Ic[i]

        for i in range(n_joints-1, -1, -1):
            if i != 0:
                Ic_composite[i-1] = (Ic[i-1]
                  + cs.mtimes(i_X_p[i].T,
                              cs.mtimes(Ic_composite[i], i_X_p[i])))

        for i in range(0, n_joints):
            fh = cs.mtimes(Ic_composite[i], Si[i])
            M[i, i] = cs.mtimes(Si[i].T, fh)
            j = i
            while j != 0:
                fh = cs.mtimes(i_X_p[j].T, fh)
                j -= 1
                M[i, j] = cs.mtimes(Si[j].T, fh)
                M[j, i] = M[i, j]

        return M

    def get_inertia_matrix_crba(self, root, tip):
        """Returns the inertia matrix as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        i_X_p, Si, Ic, tip_ofs  = self._model_calculation(root, tip, q)
        M = cs.SX.zeros(n_joints, n_joints)
        Ic_composite = [None]*len(Ic)

        for i in range(0, n_joints):
            Ic_composite[i] = Ic[i]

        for i in range(n_joints-1, -1, -1):
            if i != 0:
                Ic_composite[i-1] = Ic[i-1] + cs.mtimes(i_X_p[i].T, cs.mtimes(Ic_composite[i], i_X_p[i]))

        for i in range(0, n_joints):
            fh = cs.mtimes(Ic_composite[i], Si[i])
            M[i, i] = cs.mtimes(Si[i].T, fh)
            j = i
            while j != 0:
                fh = cs.mtimes(i_X_p[j].T, fh)
                j -= 1
                M[i, j] = cs.mtimes(Si[j].T, fh)
                M[j, i] = M[i, j]

        M = cs.Function("M", [q], [M], self.func_opts)
        return M

    def _get_C(self, i_X_p, Si, Ic, q, q_dot, n_joints,
               gravity=None, f_ext=None):
        """Internal function for calculating the joint space bias matrix."""

        v = []
        a = []
        f = []
        C = cs.SX.zeros(n_joints)

        for i in range(0, n_joints):
            vJ = cs.mtimes(Si[i], q_dot[i])
            if i == 0:
                v.append(vJ)
                if gravity is not None:
                    ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
                    a.append(cs.mtimes(i_X_p[i], -ag))
                else:
                    a.append(cs.SX([0., 0., 0., 0., 0., 0.]))
            else:
                v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
                a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

            f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

        # if f_ext is not None:
        #     f = self._apply_external_forces(f_ext, f, i_X_0)

        for i in range(n_joints-1, -1, -1):
            C[i] = cs.mtimes(Si[i].T, f[i])
            if i != 0:
                f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

        return C

    def get_coriolis_rnea(self, root, tip, f_ext=None):
        """Returns the Coriolis matrix as a casadi function."""

        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        q_dot = cs.SX.sym("q_dot", n_joints)
        i_X_p, Si, Ic , tip_ofs = self._model_calculation(root, tip, q)

        v = []
        a = []
        f = []
        tau = cs.SX.zeros(n_joints)

        for i in range(0, n_joints):
            vJ = cs.mtimes(Si[i], q_dot[i])

            if i == 0:
                v.append(vJ)
                a.append(cs.SX([0., 0., 0., 0., 0., 0.]))
            else:
                v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
                a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(plucker.motion_cross_product(v[i]), vJ))

            f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

        # if f_ext is not None:
        #     f = self._apply_external_forces(f_ext, f, i_X_0)

        for i in range(n_joints-1, -1, -1):
            tau[i] = cs.mtimes(Si[i].T, f[i])
            if i != 0:
                f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

        C = cs.Function("C", [q, q_dot], [tau], self.func_opts)
        return C

    def _add_rotor_inertia(self, Ic, Sis, G_Jmotor):
        """
        Add reflected rotor inertia along each revolute joint axis.

        Ic         : list of 6×6 SX matrices  (output of _model_calculation)
        Sis        : list of 6×1 SX vectors   (motion subspaces)
        gear_ratio : n×1 SX   (G_i)
        Jmotor     : n×1 SX   (motor inertias J_m,i)
        -------------------------------------------------------------
        Ic[i] ← Ic[i] + G_i²·J_m,i · (ê_i ê_iᵀ)   if the joint is revolute
        """
        n = len(Sis)
        for i in range(n):
            axis = Sis[i][:3]                          # rotational part
            if cs.norm_2(axis) < 1e-9:                 # prismatic joint → skip
                print('prismatic joint found')
                continue
            e_hat = axis / cs.norm_2(axis)             # unit axis ê_i
            Jr    = G_Jmotor[i]     # reflected inertia (scalar)

            # rank-1 3×3 matrix Jr·ê êᵀ
            rot_add = Jr * cs.mtimes(e_hat, e_hat.T)   # (3×3)

            # build a full 6×6 matrix to add
            deltaIc = cs.blockcat([[rot_add,                cs.SX.zeros(3,3)],
                                [cs.SX.zeros(3,3),       cs.SX.zeros(3,3)]])

            Ic[i] = Ic[i] + deltaIc                       # update link-i inertia

        return Ic
    
    def _add_payload(self, root, tip, q, m_load):
        # FK (6×1 transform) and 6×n geometric Jacobian
        J_tool_func = self._tool_Jq(root, tip)
        J_tool = J_tool_func(q)
        #  Payload force
        g_val  = -9.81              # m/s²
        F_payload = cs.vertcat(0, 0, 0, 0, 0, m_load*g_val)
        tau_payload = cs.mtimes(J_tool.T, F_payload) # τ = JᵀF
        return tau_payload

    def get_forward_dynamics_crba(self, root, tip, f_ext=None):
        """Returns the forward dynamics as a casadi function by
        solving the Lagrangian eq. of motion.  OBS! Not appropriate
        for robots with a high number of dof -> use
        get_forward_dynamics_aba().
        """
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        q_dot = cs.SX.sym("q_dot", n_joints)
        tau = cs.SX.sym("tau", n_joints)
        q_ddot = cs.SX.zeros(n_joints)

        gravity = cs.SX.sym("g")
        g_vec = cs.vertcat(0,0,gravity)

        viscous = cs.SX.sym('visc', n_joints)
        coulomb = cs.SX.sym('coul', n_joints)
        I_Grotor  = cs.SX.sym("Jm",    n_joints)             # rotor inertias
        m_load  = cs.SX.sym("m_load")

        B_vec = cs.vertcat(viscous)
        F_vec = cs.vertcat(coulomb)

        k = 50.0                  # sharpness of tanh: ↑k → closer to true sign()
        sgn_qdot = cs.tanh(k*q_dot)
        tau_fric = cs.diag(B_vec) @ q_dot + cs.diag(F_vec) @ sgn_qdot

        i_X_p, Si, Ic, tip_ofs  = self._model_calculation(root, tip, q)
        Ic = self._add_rotor_inertia(Ic, Si, I_Grotor) 
        M = self._get_M(Ic, i_X_p, Si, n_joints, q)
        M_inv = cs.solve(M, cs.SX.eye(M.size1()))

        C = self._get_C(i_X_p, Si, Ic, q, q_dot, n_joints, g_vec, f_ext)
        tau_Pload = self._add_payload(root, tip, q, m_load)
        q_ddot = cs.mtimes(M_inv, (tau - tau_fric - C - tau_Pload))
        q_ddot = cs.Function("q_ddot", [q, q_dot, tau, gravity, viscous, coulomb, I_Grotor, m_load],
                             [q_ddot], self.func_opts)

        return q_ddot