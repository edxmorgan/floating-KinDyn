"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
import casadi as ca
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import um_dynamics.utils.transformation_matrix as Transformation
import um_dynamics.utils.plucker as plucker
import um_dynamics.utils.quaternion as quatT

def require_built_model(func):
    """Decorator that ensures the dynamics model has been built."""
    def wrapper(self, *args, **kwargs):
        required_attrs = ['K', 'P', 'L']
        missing = [name for name in required_attrs if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                f"Missing model attributes: {', '.join(missing)}. "
                "Call `build_model(root, tip, floating_base)` before accessing this property."
            )
        return func(self, *args, **kwargs)
    return wrapper

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
                        upper += [ca.inf]
                        lower += [-ca.inf]
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
        
        kinematic_dict = self._kinematics(root, tip, floating_base=floating_base)
        inertial_origins_params, m_params, I_params, g, q, q_dot, base_pose, world_pose = kinematic_dict['parameters']
        
        internal_fk_eval_euler = ca.Function("internal_fkeval_euler", [q, base_pose, world_pose], [kinematic_dict['Fks'][-1]])
        
        # Arrays to track min/max
        min_pos = np.array([np.inf, np.inf, np.inf])
        max_pos = -np.array([np.inf, np.inf, np.inf])

        # collect points for plotting
        positions = np.zeros((num_samples, 3))
        world_origin = ca.DM.zeros(6,1)

        for i in range(num_samples):
            # Sample joint angles
            q_samples = [
                np.random.uniform(low=joint_limits[j][0], 
                                high=joint_limits[j][1])
                for j in range(n_joints)
            ]
            
            pose = internal_fk_eval_euler(q_samples, base_T, world_origin)

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
        
        # baseT_xyz = ca.SX.sym('T_xyz', 3) # manipulator mount link xyz wrt floating body origin 
        # baseT_rpy = ca.SX.sym('T_rpy', 3) # manipulator mount link rpy wrt floating body origin  

        # # floaing pose in world coordinates
        # x = ca.SX.sym('x') 
        # y = ca.SX.sym('y')
        # z = ca.SX.sym('z')
        # thet = ca.SX.sym('thet')
        # phi = ca.SX.sym('phi')
        # psi = ca.SX.sym('psi')
        # tr_n = ca.vertcat(x, y, z)
        # eul = ca.vertcat(phi, thet, psi)
        
        base_xyz = ca.SX.sym('base_xyz', 3) # manipulator mount link xyz wrt floating body origin 
        base_rpy = ca.SX.sym('base_rpy', 3) # manipulator mount link rpy wrt floating body origin  
        base_pose = ca.vertcat(base_xyz, base_rpy)

        # floaing pose in world coordinates
        world_x = ca.SX.sym('world_x') 
        world_y = ca.SX.sym('world_y')
        world_z = ca.SX.sym('world_z')
        world_thet = ca.SX.sym('world_thet')
        world_phi = ca.SX.sym('world_phi')
        world_psi = ca.SX.sym('world_psi')
        world_xyz = ca.vertcat(world_x, world_y, world_z)
        world_rpy = ca.vertcat(world_phi, world_thet, world_psi)
        world_pose = ca.vertcat(world_xyz, world_rpy)

        i_X_p, tip_offset, inertial_origins_params, m_params, I_b_mats, I_params, g, q, q_dot = self._model_calculation(root, tip)
        i_X_0s = [] # transformation of joint i wrt base origin
        icom_X_0s = [] # transformation of center of mass i wrt base origin
        
        base_T = plucker.XT(base_xyz, base_rpy)
        world_T = plucker.XT(world_xyz, world_rpy)
        for i in range(0, n_joints):
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
            else:
                if floating_base:
                    p0_X_world = plucker.spatial_mtimes(base_T, world_T)
                    i_X_0 = plucker.spatial_mtimes(i_X_p[i], p0_X_world)
                else:
                    i_X_0 = plucker.spatial_mtimes(i_X_p[i], base_T)
                    
            i_com_xyz = inertial_origins_params[i][:3]
            i_com_rpy = inertial_origins_params[i][3:]
            com_X_i = plucker.XT(i_com_xyz, i_com_rpy)
            
            icom_X_0s.append(plucker.spatial_mtimes(com_X_i, i_X_0))
            i_X_0s.append(i_X_0)  # transformation of joint i wrt base origin

        end_i_X_0 = plucker.spatial_mtimes(tip_offset , i_X_0)
        i_X_0s.append(end_i_X_0)
        
        R_symx, Fks, qFks, geo_J, body_J, anlyt_J = self._compute_Fk_and_jacobians(q, i_X_0s)
        com_R_symx, com_Fks, com_qFks, com_geo_J, com_body_J, com_anlyt_J = self._compute_Fk_and_jacobians(q, icom_X_0s)
        
        dynamic_parameters = [inertial_origins_params, m_params, I_params, g, q, q_dot, base_pose, world_pose]
        
        kinematic_dict = {
            "i_X_0s": i_X_0s,
            "icom_X_0s": icom_X_0s,
            "R_symx": R_symx,
            "com_R_symx": com_R_symx,
            "Fks": Fks,
            "com_Fks": com_Fks,
            "qFks": qFks,
            "com_qFks": com_qFks,
            "geo_J": geo_J,
            "com_geo_J": com_geo_J,
            "body_J": body_J,
            "com_body_J": com_body_J,
            "anlyt_J": anlyt_J,
            "com_anlyt_J": com_anlyt_J,
            'I_b_mats': I_b_mats,
            "parameters": dynamic_parameters
        }
        return kinematic_dict
    
    
    def _compute_Fk_and_jacobians(self, q, i_X_0s):
        Fks = []  # collect forward kinematics in euler form
        qFks = []  # collect forward kinematics in quaternion form
        geo_J = []          # collect geometric J’s
        body_J = []         # collect body J’s
        anlyt_J = []          # collect analytic J’s
        R_symx = []
        for i in range(len(i_X_0s)):
            H, R, p = plucker.spatial_to_homogeneous(i_X_0s[i])
            R_symx.append(R)  # collect rotation matrices
        
            # pose vector [p; φ θ ψ] using xyz Euler
            rpy        = plucker.rotation_matrix_to_euler(R, order='xyz')
            T_euler    = ca.vertcat(p, rpy)
            Fks.append(T_euler)  # forward kinematics
            
            qwxyz = quatT.rotation_matrix_to_quaternion(R, order='wxyz')
            T_quat = ca.vertcat(p, qwxyz)  # pose vector [p; qx qy qz qw]
            qFks.append(T_quat)  # forward kinematics
            
            # 6×n analytic Jacobian
            Ja         = ca.jacobian(T_euler, q)
            anlyt_J.append(Ja)
            
            phi, theta, psi = rpy[0], rpy[1], rpy[2]

            Ts = Transformation.euler_xyz_rate_to_spatial_T(phi, theta, psi)
            Tb = Transformation.euler_xyz_rate_to_body_T(phi, theta, psi)
            
            # 6×n geometric Jacobian
            Jg         = plucker.analytic_to_geometric(Ja, Ts)
            Jb         = plucker.analytic_to_geometric(Ja, Tb)
            geo_J.append(Jg)
            body_J.append(Jb)
        return R_symx, Fks, qFks, geo_J, body_J, anlyt_J


    def links_inertial(self, n_links: int):
        """
            the mass of link i is mi and that the inertia matrix of link i,
            evaluated around a coordinate frame parallel to frame i but whose origin is at
            the center of mass. 
        """
        m_params = []
        I_tensors = []
        I_params = []
        inertial_origins_params = []
        for k in range(n_links):
            origin_xyz_rpy = ca.SX.sym(f"origin_xyz_rpy_{k}", 6)
            m = ca.SX.sym(f"m_{k}")
            Ixx = ca.SX.sym(f"Ixx_{k}")
            Iyy = ca.SX.sym(f"Iyy_{k}")
            Izz = ca.SX.sym(f"Izz_{k}")
            Ixy = ca.SX.sym(f"Ixy_{k}")
            Ixz = ca.SX.sym(f"Ixz_{k}")
            Iyz = ca.SX.sym(f"Iyz_{k}")

            I_k = ca.vertcat(
                ca.hcat([Ixx, Ixy, Ixz]),
                ca.hcat([Ixy, Iyy, Iyz]),
                ca.hcat([Ixz, Iyz, Izz])
            )
            inertial_origins_params.append(origin_xyz_rpy)
            m_params.append(m)
            I_tensors.append(I_k)
            I_params.extend([Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        return inertial_origins_params, m_params, I_tensors, I_params

    def _model_calculation(self, root, tip):
        """Calculates and returns model information needed in the
        dynamics algorithms calculations, i.e transforms, joint space
        and inertia."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        n_joints = self.get_n_joints(root, tip)
        chain = self.robot_desc.get_chain(root, tip)
        i_X_p = []   # collect transforms from child link origin to parent link origin
        tip_offset = ca.DM_eye(6)
        prev_joint = None
        i = 0
        
        inertial_origins_params, m_params, I_b_mat, I_params = self.links_inertial(n_joints)
        q = ca.SX.sym("q", n_joints)
        q_dot = ca.SX.sym("q_dot", n_joints)
        g = ca.SX.sym("g", 3)  # gravity vector

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]

                if joint.type == "fixed":
                    if prev_joint == "fixed":
                        XT_prev = ca.mtimes(
                            plucker.XT(joint.origin.xyz, joint.origin.rpy),
                            XT_prev)
                    else:
                        XT_prev = plucker.XT(
                            joint.origin.xyz,
                            joint.origin.rpy)

                elif joint.type == "prismatic":
                    Si = ca.SX([0, 0, 0,
                        joint.axis[0],
                        joint.axis[1],
                        joint.axis[2]])
                    q_sign = Si.T @ ca.fabs(Si)
                    XJT = plucker.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q_sign*q[i])
                    if prev_joint == "fixed":
                        XJT = ca.mtimes(XJT, XT_prev)
                    i_X_p.append(XJT)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    Si = ca.SX([
                                joint.axis[0],
                                joint.axis[1],
                                joint.axis[2],
                                0,
                                0,
                                0])
                    q_sign = Si.T @ ca.fabs(Si)
                    XJT = plucker.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_sign*q[i])
                    if prev_joint == "fixed":
                        XJT = ca.mtimes(XJT, XT_prev)
                    i_X_p.append(XJT)
                    i += 1

                prev_joint = joint.type
                if joint.child == tip:
                    if joint.type == "fixed":
                        tip_offset = XT_prev

        return i_X_p, tip_offset, inertial_origins_params, m_params, I_b_mat, I_params, g, q, q_dot
        
    def _kinetic_energy(self, root, tip, kinematic_dict, floating_base=False):
        """Returns the kinetic energy of the system."""
        n_joints = self.get_n_joints(root, tip)
        inertial_origins_params, m_params, I_params, g, q, q_dot, base_pose, world_pose = kinematic_dict['parameters']
        D = ca.SX.zeros((n_joints, n_joints))  # D(q) is a symmetric positive definite matrix that is in general configuration dependent. The matrix  is called the inertia matrix.
        K = 0
        for i in range(n_joints):
            Jv_i = kinematic_dict['com_geo_J'][i][0:3, :]
            Jω_i = kinematic_dict['com_geo_J'][i][3:6, :]
            R_i = kinematic_dict['com_R_symx'][i]
            Ib_mat_i = kinematic_dict['I_b_mats'][i][:, :]
            m_i = m_params[i]
            D += m_i @ (Jv_i.T @ Jv_i) + Jω_i.T @ R_i @ Ib_mat_i @ R_i.T @ Jω_i
        K = 0.5 * q_dot.T @ D @ q_dot
        return K , D

    def _potential_energy(self, root, tip, kinematic_dict, floating_base=False):
        """Returns the potential energy of the system."""
        n_joints = self.get_n_joints(root, tip)
        inertial_origins_params, m_params, I_params, g, q, q_dot, base_pose, world_pose = kinematic_dict['parameters']
        P = 0
        for i in range(n_joints):
            com_Fks = kinematic_dict['com_Fks'][i]
            p_ci = com_Fks[0:3]  # Position of the center of mass in world coordinates
            m_i = m_params[i] # Mass of the link
            P += m_i * g.T @ p_ci
        return P

    def build_model(self, root, tip, floating_base=False):
        """Builds the model of the robot dynamics."""
        self.kinematic_dict = self._kinematics(root, tip, floating_base)
        self.K, self.D = self._kinetic_energy(root, tip, self.kinematic_dict, floating_base)
        self.P  = self._potential_energy(root, tip, self.kinematic_dict, floating_base)
        self.L = self.K - self.P
        return self.kinematic_dict, self.K, self.D, self.P, self.L

    @property
    @require_built_model
    def get_kinematic_dict(self):
        """Returns the kinematic dictionary of the system."""
        return self.kinematic_dict

    @property
    @require_built_model
    def get_kinetic_energy(self):
        """Returns the kinetic energy of the system."""
        return self.K

    @property
    @require_built_model
    def get_potential_energy(self):
        """Returns the potential energy of the system."""
        return self.P

    @property
    @require_built_model
    def get_lagrangian(self):
        """Returns the Lagrangian of the system."""
        return self.L

    @property
    @require_built_model
    def get_acceleration(self):
        """Returns the equations of motion for the system."""
        raise NotImplementedError("Acceleration calculation not implemented.")

    @property
    @require_built_model
    def get_inertia_matrix(self):
        """Returns the inertia matrix of the system."""
        return self.D

    @property
    @require_built_model
    def get_coriolis_centrifugal_matrix(self):
        """Returns the Coriolis and centrifugal matrix of the system."""
        raise NotImplementedError("Coriolis and centrifugal matrix calculation not implemented.")

    @property
    @require_built_model
    def get_gravity_vector(self):
        """Returns the gravity vector of the system."""
        raise NotImplementedError("Gravity vector calculation not implemented.")