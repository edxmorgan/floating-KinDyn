"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
import casadi as cs
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import um_dynamics.utils.transformation_matrix as Transformation
import um_dynamics.utils.plucker as plucker
import um_dynamics.utils.quaternion as quatT

class RobotDynamics(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]
    func_opts = {}
    jit_func_opts = {"jit": True, "jit_options": {"flags": "-O3 -ffast-math",}}
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
        
        i_X_0s, R_symx, Fks, qFks, geo_J, body_J, anlyt_J, args = self._kinematics(root, tip, floating_base=floating_base)
        
        q, tr_n, eul, baseT_xyz, baseT_rpy = args
        base_T_sym = cs.vertcat(baseT_rpy, baseT_xyz) # transform from origin to 1st child
        p_n = cs.vertcat(tr_n, eul) 
        pose_sym = cs.vertcat(p_n, q)
        internal_fk_eval_euler = cs.Function("internal_fkeval_euler", [pose_sym, base_T_sym], [Fks[-1]])
        
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

            pose = internal_fk_eval_euler(config, base_T)

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

    def _kinematics(self, root, tip, floating_base = False):
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
        p_n = cs.vertcat(tr_n, eul) # ned total states
        coordinates = cs.vertcat(p_n, q) # state coordinates

        i_X_p, Si, Ic , tip_offsets = self._model_calculation(root, tip, q)
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

        end_i_X_0 = plucker.spatial_mtimes(tip_offsets , i_X_0)
        i_X_0s.append(end_i_X_0)
        
        R_symx, Fks, qFks, geo_J, body_J, anlyt_J = self.compute_Fk_and_jacobians(coordinates, i_X_0s)
        
        args = [q, tr_n, eul, baseT_xyz, baseT_rpy]

        return i_X_0s, R_symx, Fks, qFks, geo_J, body_J, anlyt_J, args
    
    
    def compute_Fk_and_jacobians(self, coordinates, i_X_0s):
        Fks = []  # collect forward kinematics in euler form
        qFks = []  # collect forward kinematics in quaternion form
        geo_J = []          # collect geometric J’s
        body_J = []         # collect body J’s
        anlyt_J = []          # collect analytic J’s
        R_symx = []
        npoints = len(i_X_0s)
        for i in range(npoints):
            H, R, p = plucker.spatial_to_homogeneous(i_X_0s[i])
            R_symx.append(R)  # collect rotation matrices
        
            # pose vector [p; φ θ ψ] using xyz Euler
            rpy        = plucker.rotation_matrix_to_euler(R, order='xyz')
            T_euler    = cs.vertcat(p, rpy)
            Fks.append(T_euler)  # forward kinematics
            
            qwxyz = quatT.rotation_matrix_to_quaternion(R, order='wxyz')
            T_quat = cs.vertcat(p, qwxyz)  # pose vector [p; qx qy qz qw]
            qFks.append(T_quat)  # forward kinematics
            
            # 6×n analytic Jacobian
            Ja         = cs.jacobian(T_euler, cs.vertcat(coordinates))
            anlyt_J.append(Ja)
            
            phi, theta, psi = rpy[0], rpy[1], rpy[2]

            Ts = Transformation.euler_xyz_rate_to_spatial_T(phi, theta, psi)
            Tb = Transformation.euler_xyz_rate_to_body_T(phi, theta, psi)
            
            # 6×n geometric Jacobian
            Jg         = self.analytic_to_geometric(Ja, Ts)
            Jb         = self.analytic_to_geometric(Ja, Tb)
            geo_J.append(Jg)
            body_J.append(Jb)
        return R_symx, Fks, qFks, geo_J, body_J, anlyt_J


    def analytic_to_geometric(self,Ja, T):
        """
        Convert a 6×n analytic Jacobian (XYZ Euler) to geometric.
        Ja  – analytic Jacobian  (6×n SX/DM)
        """
        # split linear / angular blocks
        Jv = Ja[0:3, :]                  # linear rows stay the same
        Jθ = Ja[3:6, :]                  # Euler‑rate rows → map
        Jω = T @ Jθ                      # angular velocity rows
        return cs.vertcat(Jv, Jω)        # 6×n geometric Jacobian


    def _model_calculation(self, root, tip, q):
        """Calculates and returns model information needed in the
        dynamics algorithms caluculations, i.e transforms, joint space
        and inertia."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        chain = self.robot_desc.get_chain(root, tip)
        spatial_inertias = []
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
                    Si = cs.SX([0, 0, 0,
                        joint.axis[0],
                        joint.axis[1],
                        joint.axis[2]])
                    q_sign = Si.T @ cs.fabs(Si)
                    XJT = plucker.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q_sign*q[i])
                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)
 
                    i_X_p.append(XJT)
                    Sis.append(Si)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    if n_actuated != 0:
                        spatial_inertias.append(spatial_inertia)
                    n_actuated += 1
                    Si = cs.SX([
                                joint.axis[0],
                                joint.axis[1],
                                joint.axis[2],
                                0,
                                0,
                                0])
                    q_sign = Si.T @ cs.fabs(Si)
                    XJT = plucker.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_sign*q[i])
                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)
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
    
    def kinetic_enegy(self):
        """Returns the kinetic energy of the system."""
        raise NotImplementedError("Kinetic energy calculation not implemented.")
    
    def potential_energy(self):
        """Returns the potential energy of the system."""
        raise NotImplementedError("Potential energy calculation not implemented.")
    
    def lagrangian(self):
        """Returns the Lagrangian of the system."""
        raise NotImplementedError("Lagrangian calculation not implemented.")
    
    def eom(self, root, tip):
        """Returns the equations of motion for the system."""
        raise NotImplementedError("Equations of motion calculation not implemented.")
    