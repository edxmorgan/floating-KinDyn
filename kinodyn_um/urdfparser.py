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

    def _tool_Jq(self, n_joints, i_X_p, Si, tip_ofs, q):
        """Internal the forward kinematics as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        for i in range(0, n_joints):
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
            else:
                i_X_0 = i_X_p[i]

        endeff_X_0 = plucker.spatial_mtimes( tip_ofs , i_X_0)

        H_eff , R_eff, p_eff = plucker.spatial_to_homogeneous(endeff_X_0)
        Fk_eff = cs.vertcat(plucker.rotation_matrix_to_euler(R_eff, order='xyz'), p_eff)

        coeffs = cs.vertcat(*(S.T @ cs.fabs(S) for S in Si))      # n×1 SX
        q_on_axis = cs.diag(q) @ coeffs

        Fk_eff_fun = cs.Function("Fk_eff", [q], [Fk_eff] , self.func_opts)
        Fk_eff_ = Fk_eff_fun(q_on_axis)

        J = cs.jacobian(Fk_eff_, q)

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

        forward_kin = plucker.spatial_mtimes(tip_ofs , i_X_0)
        i_X_0s.append(forward_kin)

        coeffs = cs.vertcat(*(S.T @ cs.fabs(S) for S in Si))      # n×1 SX
        q_on_axis = cs.diag(q) @ coeffs

        if floating_base:
            i_X_f = cs.Function("i_X_f", [q, tr_n, eul, baseT_xyz, baseT_rpy], i_X_0s , self.func_opts)
            i_X_0s_ = i_X_f(q_on_axis, tr_n, eul, baseT_xyz, baseT_rpy)
            i_X_f_ = cs.Function("i_X_f_", [q, tr_n, eul, baseT_xyz, baseT_rpy], i_X_0s_ , self.func_opts)
        else:
            i_X_f = cs.Function("i_X_f", [q], i_X_0s , self.func_opts)
            i_X_0s_ = i_X_f(q_on_axis)
            i_X_f_ = cs.Function("i_X_f_", [q], i_X_0s_ , self.func_opts)

        return i_X_f_

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

    def _get_gravity_rnea(self, root, tip):
        """Returns the gravitational term as a casadi function."""

        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        i_X_p, Si, Ic , tip_ofs = self._model_calculation(root, tip, q)
        gravity = cs.SX.sym("g",3)
        v = []
        a = []
        ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
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

        g_tau = cs.Function("G", [q, gravity], [tau],
                          self.func_opts)
        return g_tau

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
    
    def _add_payload(self, n_joints, i_X_p, Si, tip_ofs, q, g,  payload_props):
        # FK (6×1 transform) and 6×n geometric Jacobian
        J_tool_func = self._tool_Jq(n_joints, i_X_p, Si, tip_ofs, q)
        J_tool = J_tool_func(q)
        #  Payload force
        F_payload = cs.vertcat(0, 0, 0, 0, 0, payload_props[0]*g)
        tau_payload = cs.mtimes(J_tool.T, F_payload) # τ = JᵀF
        return tau_payload
    
    def get_active_complaince_tau(self, root, tip):
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        n_joints = self.get_n_joints(root, tip)
        q = cs.SX.sym("q", n_joints)
        gravity = cs.SX.sym("g")
        g_vec = cs.vertcat(0,0,gravity)
        payload_props  = cs.SX.sym("payload_props", 4) # mass, Ixx, Iyy, Izz

        tau_g = self._get_gravity_rnea(root, tip)      # G(q,g) → τ_g
        G_tau = tau_g(q, g_vec)

        i_X_p, Si, Ic, tip_ofs  = self._model_calculation(root, tip, q)
        tau_pl = self._add_payload(n_joints, i_X_p, Si, tip_ofs, q, gravity, payload_props)

        tau_comp = G_tau + tau_pl                     # ≈ “motor holds the arm”
        tau_comp_f = cs.Function("tau_comp", [q, gravity, payload_props],
                             [tau_comp], self.func_opts)
        return tau_comp_f


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
        payload_props  = cs.SX.sym("payload_props", 4) # mass, Ixx, Iyy, Izz
 
        k        = cs.SX.sym("k", n_joints, 2)          # sharpness of tanh: ↑k → closer to true sign()
        viscous  = cs.SX.sym("visc",   n_joints, 2)
        coulomb  = cs.SX.sym("coul",   n_joints, 2)
        I_Grotor = cs.SX.sym("Jm",     n_joints, 2)   # rotor inertias

        forward = cs.sign(tau) >= 0

        # directional dependent parameters
        B_vec  = cs.if_else(forward, viscous[:, 0], viscous[:, 1])
        F_vec  = cs.if_else(forward, coulomb[:, 0],  coulomb[:, 1])
        Jm_vec = cs.if_else(forward, I_Grotor[:, 0], I_Grotor[:, 1])
        k_vec  = cs.if_else(forward, k[:, 0],        k[:, 1])
               
        sgn_qdot = cs.tanh(k_vec * q_dot)
        tau_fric = cs.diag(B_vec) @ q_dot + cs.diag(F_vec) @ sgn_qdot

        i_X_p, Si, Ic, tip_ofs  = self._model_calculation(root, tip, q)
        Ic = self._add_rotor_inertia(Ic, Si, Jm_vec)
        M = self._get_M(Ic, i_X_p, Si, n_joints, q)
        M_inv = cs.solve(M, cs.SX.eye(M.size1()))

        C = self._get_C(i_X_p, Si, Ic, q, q_dot, n_joints, g_vec, f_ext)
        tau_pl = self._add_payload(n_joints, i_X_p, Si, tip_ofs, q, gravity, payload_props)

        
        q_ddot = cs.mtimes(M_inv, (tau - tau_fric - C - tau_pl))
        
        q_ddot_f = cs.Function("q_ddot", [q, q_dot, tau, gravity, k, viscous, coulomb, I_Grotor, payload_props],
                             [q_ddot], self.func_opts)
        

        return q_ddot_f
    
    def forward_simulation(self, root, tip):
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        n_joints = self.get_n_joints(root, tip)

        # ─────────────────────────────────────────────────────────────────────────────
        # 2. CasADi symbols
        # ─────────────────────────────────────────────────────────────────────────────
        q     = cs.MX.sym('q',     n_joints)        # joint positions
        q_dot = cs.MX.sym('q_dot', n_joints)        # joint velocities
        tau_cmd = cs.MX.sym('tau_cmd', n_joints)   # *controller* torque
        dt    = cs.MX.sym('dt')                     # integration step (s)
        g     = cs.MX.sym('g')                      # gravity acceleration (e.g. -9.81)

        k        = cs.MX.sym('k_',        n_joints, 2)   # tanh sharpness coefficients
        viscous  = cs.MX.sym('viscous',   n_joints, 2)   # viscous friction
        coulomb  = cs.MX.sym('coulomb',   n_joints, 2)   # Coulomb friction
        I_Grotor = cs.MX.sym('I_Grotor',  n_joints, 2)   # rotor inertias
        payload_props  = cs.MX.sym("payload_props", 4) # mass, Ixx, Iyy, Izz

        lower_joint_limit = cs.MX.sym('lower_limit', n_joints)
        upper_joint_limit = cs.MX.sym('upper_limit', n_joints)
        EPS_TORQUE = cs.MX.sym('eps_torque', n_joints)

        # ─────────────────────────────────────────────────────────────────────────────
        # 3. Forward dynamics (from CRBA model produced by urdf2casadi)
        # ─────────────────────────────────────────────────────────────────────────────
        q_ddot_fun = self.get_forward_dynamics_crba(root, tip) # acceleration
        tau_hold_fun = self.get_active_complaince_tau(root, tip)  # gravity+payload

        # ------------------------------------------------------------------
        # 4. Holding torque (independent of tau_cmd)
        # ------------------------------------------------------------------
        tau_hold = tau_hold_fun(q, g, payload_props)

        # Decision: if |tau_cmd[i]| < ε  →  use tau_hold[i]
        use_hold   = cs.fabs(tau_cmd) < EPS_TORQUE
        tau_eff    = cs.if_else(use_hold, tau_hold, tau_cmd)

        # ─────────────────────────────────────────────────────────────────────────────
        # 6. Joint‑limit saturation with recovery
        #    Freeze motion only if it is trying to go farther OUT of bounds.
        # ─────────────────────────────────────────────────────────────────────────────
        EPS = 1e-1
        tau_sat = cs.MX.zeros(n_joints)
        v_guard = cs.MX.zeros(n_joints)

        for i in range(n_joints):

            above = q[i] >= upper_joint_limit[i] - EPS
            below = q[i] <= lower_joint_limit[i] + EPS

            # Outward velocity?
            vel_out_high = cs.logic_and(above, q_dot[i] > 0)
            vel_out_low  = cs.logic_and(below, q_dot[i] < 0)

            # “Adds energy” if torque points the same way as the outward motion
            adds_energy_high = cs.logic_and(vel_out_high, tau_eff[i] >  0)
            adds_energy_low  = cs.logic_and(vel_out_low,  tau_eff[i] <  0)

            # Also block any *static* outward push (velocity zero but torque outward)
            push_out_high = cs.logic_and(above, tau_eff[i] > 0)
            push_out_low  = cs.logic_and(below, tau_eff[i] < 0)

            violate = cs.logic_or(push_out_high, push_out_low)
            violate = cs.logic_or(violate,       adds_energy_high)
            violate = cs.logic_or(violate,       adds_energy_low)

            tau_sat[i] = cs.if_else(violate, 0, tau_eff[i])

            v_guard[i] = cs.if_else(cs.logic_or(vel_out_high, vel_out_low), 0, q_dot[i])


        # ------------------------------------------------------------------
        # 5. Forward dynamics with the *selected* torque
        # ------------------------------------------------------------------
        q_ddot = q_ddot_fun(q, v_guard, tau_sat, g, k, viscous, coulomb, I_Grotor, payload_props)
            
        # ------------------------------------------------------------------
        # 7. State vector and ODE
        #     • q-dot derivative uses v_guard  ⇒ no outward drift
        #     • q_dot derivative uses q_ddot   ⇒ real braking dynamics
        # ------------------------------------------------------------------
        x    = cs.vertcat(q,  q_dot)
        xdot = cs.vertcat(v_guard, q_ddot) * dt


        # ─────────────────────────────────────────────────────────────────────────────
        # 8. Parameter vector (keep one long vector to avoid argument mismatch)
        # ─────────────────────────────────────────────────────────────────────────────
        p = cs.vertcat(
                dt,
                g,
                cs.reshape(k,       -1, 1),
                cs.reshape(viscous, -1, 1),
                cs.reshape(coulomb, -1, 1),
                cs.reshape(I_Grotor,-1, 1),
                payload_props,
                lower_joint_limit,
                upper_joint_limit,
                EPS_TORQUE)


        # # ─────────────────────────────────────────────────────────────────────────────
        # # 7. Integrator (Runge–Kutta over [0,1])
        # # ─────────────────────────────────────────────────────────────────────────────
        dae  = {'x': x, 'ode': xdot, 'p': p, 'u': tau_cmd}
        opts = {
            'simplify': True,
            'number_of_finite_elements': 100,
        }

        intg = cs.integrator('intg', 'rk', dae, 0, 1, opts)


                            
        # ─────────────────────────────────────────────────────────────────────────────
        # 8. Next‑state function and file export
        # ─────────────────────────────────────────────────────────────────────────────
        x_next = intg(x0=x, u=tau_cmd, p=p)['xf']
        p_sim =cs.vertcat(
            cs.reshape(k,   -1, 1), 
            cs.reshape(viscous,   -1, 1),
            cs.reshape(coulomb,   -1, 1),
            cs.reshape(I_Grotor,  -1, 1)
                )

        F_next = cs.Function('Mnext', [x, tau_cmd, dt, g, payload_props, p_sim , lower_joint_limit, upper_joint_limit, EPS_TORQUE], [x_next], self.func_opts)
        return F_next
