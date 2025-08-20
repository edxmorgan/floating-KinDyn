"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
import casadi as ca
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import system.utils.transformation_matrix as Transformation
import system.utils.plucker as plucker
import system.utils.quaternion as quatT

def require_built_model(func):
    """Decorator that ensures the dynamics model has been built."""
    def wrapper(self, *args, **kwargs):
        required_attrs = ['kinematic_dict', 'K', 'P', 'L', 'D', 'C', 'g', 'joint_torque', 'qdd','B']
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
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p, q, q_dot, q_dotdot, tau, base_pose, world_pose = kinematic_dict['parameters']
        
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

        i_X_p, tip_offset, c_parms, m_params, I_b_mats, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p, q, q_dot, q_dotdot, tau = self._model_calculation(root, tip)
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
                    
            c_i = c_parms[i]
            com_X_i = plucker.XT(c_i, [0,0,0])
            
            icom_X_0s.append(plucker.spatial_mtimes(com_X_i, i_X_0))
            i_X_0s.append(i_X_0)  # transformation of joint i wrt base origin

        end_i_X_0 = plucker.spatial_mtimes(tip_offset , i_X_0)
        i_X_0s.append(end_i_X_0)

        R_symx, Fks, qFks, geo_J, body_J, anlyt_J = self._compute_Fk_and_jacobians(q, i_X_0s)
        com_R_symx, com_Fks, com_qFks, com_geo_J, com_body_J, com_anlyt_J = self._compute_Fk_and_jacobians(q, icom_X_0s)
        
        dynamic_parameters = [c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose]
        
        kinematic_dict = {
            "n_joints": n_joints,
            
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
        c_parms = []
        for k in range(n_links):
            c_xyz = ca.SX.sym(f"c_{k}", 3)
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
            c_parms.append(c_xyz)
            m_params.append(m)
            I_tensors.append(I_k)
            I_params.extend([Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        return c_parms, m_params, I_tensors, I_params

    def _model_calculation(self, root, tip):
        """Calculates and returns model information needed in the
        dynamics algorithms calculations, i.e transforms, joint space
        and inertia."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        n_joints = self.get_n_joints(root, tip)
        chain = self.robot_desc.get_chain(root, tip)
        i_X_p = []   # collect transforms from child link origin to parent link origin
        axis_signs = []
        tip_offset = ca.DM.eye(6)
        prev_joint = None
        i = 0
        
        c_parms, m_params, I_b_mats, I_params = self.links_inertial(n_joints)
        q = ca.SX.sym("q", n_joints)
        q_dot = ca.SX.sym("q_dot", n_joints)
        vec_g = ca.SX.sym("vec_g", 3)  # gravity vector
        tau = ca.SX.sym("tau", n_joints) # joint torque
        q_dotdot = ca.SX.sym("q_dotdot", n_joints) # joint accelerations
        fv_coeff = ca.SX.sym('Fv', n_joints)
        fs_coeff = ca.SX.sym('Fs', n_joints)
        r_com_body = ca.SX.sym("r_com_body", 3)  # 3x1, COM in tool (end-effector) frame
        m_p = ca.SX.sym("m_p") # payload mass
        
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]

                if joint.type == "fixed":
                    if prev_joint == "fixed":
                        XT_prev = plucker.spatial_mtimes(
                            plucker.XT(joint.origin.xyz, joint.origin.rpy),
                            XT_prev)
                    else:
                        XT_prev = plucker.XT(
                            joint.origin.xyz,
                            joint.origin.rpy)

                elif joint.type == "prismatic":
                    a_parent = ca.SX(joint.axis)
                    R = plucker.rotation_rpy(joint.origin.rpy[0], joint.origin.rpy[1],joint.origin.rpy[2])  # parent→joint
                    a_true = R.T @ a_parent
                    a_used = a_parent
                    s_i = ca.sign(ca.dot(a_true, a_used))
                    axis_signs.append(s_i)

                    XJT = plucker.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        s_i*q[i])
                    if prev_joint == "fixed":
                        XJT = plucker.spatial_mtimes(XJT, XT_prev)
                    i_X_p.append(XJT)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    a_parent = ca.SX(joint.axis)         # as authored
                    R = plucker.rotation_rpy(joint.origin.rpy[0], joint.origin.rpy[1],joint.origin.rpy[2])  # parent→joint
                    a_true = R.T @ a_parent              # axis in joint frame (URDF spec)
                    a_used = a_parent                    # what your kernel effectively uses
                    s_i = ca.sign(ca.dot(a_true, a_used))
                    axis_signs.append(s_i)

                    XJT = plucker.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        s_i*q[i])
                    if prev_joint == "fixed":
                        XJT = plucker.spatial_mtimes(XJT, XT_prev)
                    i_X_p.append(XJT)
                    i += 1

                prev_joint = joint.type
                if joint.child == tip:
                    if joint.type == "fixed":
                        tip_offset = XT_prev
        self._axis_signs = ca.diag(ca.vcat(axis_signs))   # n×n diagonal
        return i_X_p, tip_offset, c_parms, m_params, I_b_mats, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p, q, q_dot, q_dotdot, tau

    def _sys_id_lump_parameters(self):
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        n_joints = self.kinematic_dict['n_joints']
        I_lump = []
        mrc_lump = []
        
        m_id_list = []
        m_rci_id_list = []
        I_id_list = []
        fs_list = []
        fv_list = []
        for i in range(n_joints):
            # define lumped parameters in dynamics model
            m_i = m_params[i]
            r_ci = c_parms[i]
            mrc_lump.append(m_i * r_ci) 
            
            Ib_mat_i = self.kinematic_dict['I_b_mats'][i]
            Sr = ca.skew(c_parms[i])
            Ibar_link = Ib_mat_i + m_params[i] * (Sr.T @ Sr)   # com axis inertia in link frame i (lumped like this for id) using parallel axis theorem
            Ici_list = ca.vertcat(Ibar_link[0,0], Ibar_link[1,1], Ibar_link[2,2],
                      Ibar_link[0,1], Ibar_link[0,2], Ibar_link[1,2])
            I_lump.append(Ici_list)
            
            # define lumped parameters for system id
            m_i_id = ca.SX.sym(f'm_i_{i}')
            m_rci_id = ca.SX.sym(f'm_r_ci_{i}', 3)
            I_i_xx_id = ca.SX.sym(f"Ixx_lumped_{i}")
            I_i_yy_id = ca.SX.sym(f"Iyy_lumped_{i}")
            I_i_zz_id = ca.SX.sym(f"Izz_lumped_{i}")
            I_i_xy_id = ca.SX.sym(f"Ixy_lumped_{i}")
            I_i_xz_id = ca.SX.sym(f"Ixz_lumped_{i}")
            I_i_yz_id = ca.SX.sym(f"Iyz_lumped_{i}")
            fs_i_id = ca.SX.sym(f"fs_{i}")
            fv_i_id = ca.SX.sym(f"fv_{i}")

            m_id_list.append(m_i_id)
            m_rci_id_list.append(m_rci_id)
            I_id_list.append([I_i_xx_id, I_i_yy_id, I_i_zz_id, I_i_xy_id, I_i_xz_id, I_i_yz_id])
            fs_list.append(fs_i_id)
            fv_list.append(fv_i_id)
            
        self._sys_id_coeff =  {
            "masses": m_params,                    # [m_i]
            "first_moments": mrc_lump,             # [m_i * r_ci] (3-vec per link)
            "inertias_vec6": I_lump,               # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about link i origin
            "fs": fs_coeff, 
            "fv": fv_coeff,
            
            "masses_id_list": m_id_list,           # symbols for m_i
            "first_moments_id_list": m_rci_id_list,# symbols for m_i * r_ci (3-vec per link)
            "inertias_id_list": I_id_list,         # symbols for inertia vec6
            "fv_list": fv_list,
            "fs_list": fs_list,
            
            "masses_id_syms_vertcat": ca.vertcat(*m_id_list),           # symbols for m_i
            "first_moments_id_vertcat": ca.vertcat(*m_rci_id_list),# symbols for m_i * r_ci (3-vec per link)
            "inertias_id_vertcat": ca.vertcat(*[s for six in I_id_list for s in six]),             # symbols for inertia vec6
            "fv_id_vertcat": ca.vertcat(*fv_list),
            "fs_id_vertcat": ca.vertcat(*fs_list),   
        }


    def _pseudo_inertia(self, m, mc, I_bar):
        """
        Construct the 4×4 pseudo-inertia (spatial inertia) matrix.

        Args
        ----
        m      : mass (scalar SX)
        c      : 3×1 centre‐of‐mass vector (SX)
        I_bar  : 3×3 inertia tensor about the same origin as `c` (SX)

        Returns
        -------
        J : 4×4 SX matrix.  J ≽ 0  ⇔  physically consistent parameters.
        """
        # upper-left 3×3 block
        J_ul = 0.5 * ca.trace(I_bar) * ca.SX.eye(3) - I_bar
        # cross blocks
        J_ur = mc
        J_ll = mc.T
        # bottom-right scalar block
        J_br = m

        J = ca.vertcat(
                ca.hcat([J_ul, J_ur]),
                ca.hcat([J_ll, J_br])
            )
        return J
    
    def _build_link_i_regressor(self):
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        n_joints = self.kinematic_dict['n_joints']
        theta_i_list = []
        Beta_K = [] #collection (11 × 1) vectors that allow the Kinetic energy to be written as a function of πi (lumped parameters).
        Beta_P = [] #collection (11 × 1) vectors that allow the Potential energy to be written as a function of πi (lumped parameters).
        D_i_s = []
        physical_plausibility_matrices = []
        for i in range(n_joints):            
            m_i_id = self._sys_id_coeff['masses_id_list'][i]
            m_rci_id = self._sys_id_coeff['first_moments_id_list'][i]
            I_id_list = self._sys_id_coeff['inertias_id_list'][i]
            fv_i_id = self._sys_id_coeff["fv_list"][i]
            fs_i_id = self._sys_id_coeff["fs_list"][i]

            I_i_xx_id, I_i_yy_id, I_i_zz_id, I_i_xy_id, I_i_xz_id, I_i_yz_id = I_id_list
            
            I_i_id = ca.vertcat(
                ca.hcat([I_i_xx_id, I_i_xy_id, I_i_xz_id]),
                ca.hcat([I_i_xy_id, I_i_yy_id, I_i_yz_id]),
                ca.hcat([I_i_xz_id, I_i_yz_id, I_i_zz_id])
            ) # inertia at center wrt origin at frame i
            
            # define lumped parameters linear in kinetic energy
            Jv_i = self.kinematic_dict['geo_J'][i][0:3, :]
            Jω_i = self.kinematic_dict['geo_J'][i][3:6, :]
            
            R_i = self.kinematic_dict['R_symx'][i]
            mrc_world = R_i @ m_rci_id
            cross    = Jv_i.T @ ca.skew(mrc_world) @ Jω_i
            
            I_i_world = R_i @ I_i_id @ R_i.T # tranform inertia from frame i to world frame
            D_i = m_i_id * (Jv_i.T @ Jv_i) + (Jω_i.T@I_i_world@Jω_i) - (cross + cross.T)
            K_i = 0.5 * q_dot.T @D_i@ q_dot
            
            # define lumped parameters linear in potential energy
            p_i = self.kinematic_dict['Fks'][i][0:3]  # Position of joint i in world coordinates
            
            Y_Pi = ca.horzcat(vec_g.T @ p_i, (R_i.T @ vec_g).T)   # shape (1, 1+3)
            P_i = Y_Pi @ ca.vertcat(m_i_id, m_rci_id)

            physical_plausibility_matrix = self._pseudo_inertia(m_i_id, m_rci_id, I_i_id)
            physical_plausibility_matrices.append(physical_plausibility_matrix)
            # collect lumped parameters into
            theta_i = [m_i_id, m_rci_id, I_i_xx_id, I_i_yy_id, I_i_zz_id, I_i_xy_id, I_i_xz_id, I_i_yz_id, fv_i_id, fs_i_id]
            theta_i_SX = ca.vertcat(*theta_i)
            theta_i_list.append(theta_i)
 
            # compute inertia and energies
            beta_K_i = ca.gradient(K_i, theta_i_SX[0:10])
            beta_P_i = ca.gradient(P_i, theta_i_SX[0:10])
            
            # collect energy regressors
            Beta_K.append(beta_K_i)
            Beta_P.append(beta_P_i)
            D_i_s.append(D_i)
            
        return D_i_s, Beta_K, Beta_P, theta_i_list, physical_plausibility_matrices
    
    def _build_sys_regressor(self, q, q_dot, q_ddot):
        K = 0
        P = 0
        theta = []
        self._sys_id_lump_parameters()
        D_i_s, Beta_K, Beta_P, theta_i_list, physical_plausibility_matrices = self._build_link_i_regressor()
        n = q.numel()
        
        theta_sizes = [int((ca.vertcat(*t)).size1()) for t in theta_i_list]
        assert len(set(theta_sizes)) == 1, f"Inconsistent theta sizes: {theta_sizes}"
        n_theta = theta_sizes[0]
        
        Y = ca.SX.zeros(n, n*n_theta)
        D = ca.SX.zeros(n, n)
        for i in range(n):
            theta_i = theta_i_list[i]
            theta.extend(theta_i)
            
            theta_i_SX = ca.vertcat(*theta_i)
            D += D_i_s[i]
            K += Beta_K[i].T@theta_i_SX[0:10]
            P += Beta_P[i].T@theta_i_SX[0:10]
            
            for j in range(n):
                #Y is block upper triangular, so we only calculate and populate blocks where j >= i.
                if j >= i:
                    BTj = Beta_K[j]  # (p_per x 1)
                    BUj = Beta_P[j]  # (p_per x 1)

                    # ∂β_Tj / ∂ q̇_i  (p_per x 1)
                    dBTj_dqdi = ca.jacobian(BTj, q_dot[i])

                    # d/dt(∂β_Tj/∂q̇_i) = (∂/∂q)·q̇ + (∂/∂q̇)·q̈
                    
                    dt_term = ca.jtimes(dBTj_dqdi, q, q_dot) + ca.jtimes(dBTj_dqdi, q_dot, q_ddot)  # (p_per x 1)

                    # − ∂β_Tj/∂q_i + ∂β_uj/∂q_i
                    minus_q_term = ca.jacobian(BTj, q[i])   # (p_per x 1)
                    plus_u_term  = ca.jacobian(BUj, q[i])   # (p_per x 1)

                    y_ij = dt_term - minus_q_term + plus_u_term   # (p_per x 1)
                    
                    # friction only on the joint's own block
                    fric_Y = ca.vertcat(q_dot[i], ca.sign(q_dot[i])) if j == i else ca.SX.zeros(2,1)
                    
                    y_ij_b = ca.vertcat(y_ij, fric_Y)  # length n_theta (=10+2)
                    
                    start_col = j * n_theta
                    end_col = (j + 1) * n_theta
                    Y[i, start_col:end_col] = y_ij_b.T
                    
        C = self._build_coriolis_centrifugal_matrix(q, q_dot, D)
        g = self._build_gravity_term(P, q)
        
        theta = ca.vertcat(*theta)
        return D, C, K, P, g, Y, theta, physical_plausibility_matrices

    def _kinetic_energy(self):
        """Returns the kinetic energy of the system."""
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p, q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        n_joints = self.kinematic_dict['n_joints']
        D = ca.SX.zeros((n_joints, n_joints))  # D(q) is a symmetric positive definite matrix that is in general configuration dependent. The matrix  is called the inertia matrix.
        K = 0
        for i in range(n_joints):
            m_i = m_params[i]
            Ib_mat_i = self.kinematic_dict['I_b_mats'][i]
            R_i      = self.kinematic_dict['R_symx'][i]      # 3×3 (rotation of link i in world)
            
            Jv_com_i = self.kinematic_dict['com_geo_J'][i][0:3, :]
            Jω_com_i = self.kinematic_dict['com_geo_J'][i][3:6, :]
            D += m_i * (Jv_com_i.T @ Jv_com_i) + Jω_com_i.T @ R_i @ Ib_mat_i @ R_i.T @ Jω_com_i
        K = 0.5 * q_dot.T @ D @ q_dot
        return K , D

    def _potential_energy(self):
        """Returns the potential energy of the system."""
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        P = 0
        n_joints = self.kinematic_dict['n_joints']
        for i in range(n_joints):
            m_i = m_params[i] # Mass of the link
            i_com_Fks = self.kinematic_dict['com_Fks'][i]
            p_ci = i_com_Fks[0:3]  # Position of the center of mass of link i in world coordinates
            P += m_i * vec_g.T @ p_ci
        return P

    def _christoﬀel_symbols_cijk(self, q, D, i, j, k):
        """Returns the christoﬀel_symbols cijk"""
        dkj_qi = ca.gradient(D[k, j], q[i])
        dki_qj = ca.gradient(D[k, i], q[j])
        dij_qk = ca.gradient(D[i, j], q[k])
        cijk = 0.5 * (dkj_qi + dki_qj - dij_qk)
        return cijk
    
    def _build_coriolis_centrifugal_matrix(self, q, q_dot, D):
        n_joints = q.size1()
        C = ca.SX.zeros(n_joints, n_joints)
        for k in range(n_joints):
            for j in range(n_joints):
                ckj = 0
                for i in range(n_joints):
                    q_dot_i = q_dot[i]
                    cijk = self._christoﬀel_symbols_cijk(q, D, i, j, k)
                    ckj += (cijk*q_dot_i)
                C[k,j] = ckj
        return C
    
    def _build_gravity_term(self, P, q):
        g_q = self._axis_signs@ca.gradient(P, q)
        return g_q
    
    def _build_friction_term(self, Fv, Fs, q_dot):
        viscous_friction = ca.diag(Fv)@q_dot
        column_friction = ca.diag(Fs)@ca.sign(q_dot)
        friction = viscous_friction + column_friction
        return friction
    
    def _build_D_dot(self, q, q_dot, D):
        n_joints = q.size1()
        D_dot = ca.SX.zeros(n_joints, n_joints)
        for k in range(n_joints):
            for j in range(n_joints):
                d_dot_kj = 0
                for i in range(n_joints):
                    qi = q[i]
                    q_dot_i = q_dot[i]
                    dkj = D[k,j]
                    d_dot_kj += ca.gradient(dkj, qi)*q_dot_i
                D_dot[k,j] = d_dot_kj
        return D_dot
          
    def _build_N(self, q, q_dot, D):
        n_joints = q.size1()
        N = ca.SX.zeros(n_joints, n_joints)
        for k in range(n_joints):
            for j in range(n_joints):
                n_kj = 0
                for i in range(n_joints):
                    qk = q[k]
                    qj = q[j]
                    dij = D[i,j]
                    dki = D[k,i]
                    q_dot_i = q_dot[i]
                    n_kj += (ca.gradient(dij, qk) - ca.gradient(dki, qj))*q_dot_i
                N[k,j] = n_kj
        return N

    def _payload_wrench_from_mass(self, m_p, r_com_body):
        # r_com_body: 3x1, COM in tool (end-effector) frame
        # Uses last link rotation to world and gravity vec_g from parameters.
        R_e = self.kinematic_dict["R_symx"][-1]    # 3x3, tool->world
        _, _, _, _, _, vec_g, *_ = self.kinematic_dict['parameters']
        f = m_p * vec_g                             # 3x1, world force
        r_w = R_e @ r_com_body                      # 3x1, COM in world
        τ = ca.cross(r_w, f)                      # 3x1, moment at tool origin
        return ca.vertcat(f, τ)                   # 6x1

    def _build_forward_dynamics(self, D, C, q_dot, g, B, tau, J_tip, F_payload):
        tau_payload = J_tip.T @ F_payload          # n×1
        tau_hat =  g + B + tau_payload #C@q_dot + g + B #+ tau_payload
        qdd = ca.solve(D, tau - tau_hat) # qdd = ca.inv(D)@(tau - tau_hat)
        return qdd

    def forward_simulation(self):
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        dt = ca.SX.sym('dt')
        rigid_p = ca.vertcat(*c_parms, *m_params, *I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p, base_pose, world_pose)
        p = ca.vertcat(rigid_p, dt)
        x    = ca.vertcat(q,  q_dot)
        xdot = ca.vertcat(q_dot, self.qdd) * dt
        dae  = {'x': x, 'ode': xdot, 'p': p, 'u': tau}
        opts = {
            'simplify': True,
            'number_of_finite_elements': 5,
            }
        intg = ca.integrator('intg', 'rk', dae, 0, 1, opts)
        x_next = intg(x0=x, u=tau, p=p)['xf']
        F_next = ca.Function('Mnext', [x, tau, dt, rigid_p], [x_next])
        return F_next
    
    def _build_inverse_dynamics(self, D, C, q_dotdot, q_dot, g, B, J_tip, F_payload):
        tau_payload = J_tip.T @ F_payload
        joint_torque = D@q_dotdot + C@q_dot + g + B + tau_payload
        return joint_torque

    def _compute_base_reaction(self):
        """
        Computes the manipulator reaction to the floating base at the base origin.
        returns the wrench from the base onto the arm.
        The floating base(AUV) feels the equal and opposite wrench.
        """
        # Get symbolic variables from the built model
        params = self.kinematic_dict['parameters']
        q, q_dot, q_ddot = params[8], params[9], params[10]
        vec_g = params[5]
        base_pose        = params[12]
        world_pose       = params[13]
        n_joints = self.kinematic_dict['n_joints']

        # External wrench [f; τ]
        F_ext = ca.SX.sym('F_ext', 6)
    
        # Base origin in WORLD
        base_T  = plucker.XT(base_pose[0:3],  base_pose[3:6])
        world_T = plucker.XT(world_pose[0:3], world_pose[3:6])
        p0_X_world     = plucker.spatial_mtimes(base_T, world_T)     # works for both fixed & floating
        _, _, p_base_W = plucker.spatial_to_homogeneous(p0_X_world)
    
        # Initialize total inertial force and torque (rate of change of momentum)
        total_inertial_force = ca.SX.zeros(3, 1)
        total_inertial_torque = ca.SX.zeros(3, 1)  # about the base origin

        # Initialize total gravity wrench
        total_gravity_force = ca.SX.zeros(3, 1)
        total_gravity_torque = ca.SX.zeros(3, 1)   # about the base origin

        # Loop through each link to aggregate its contribution
        for i in range(n_joints):
            # Get link parameters
            m_i = self.kinematic_dict['parameters'][1][i]
            I_ci = self.kinematic_dict['I_b_mats'][i] # Inertia tensor in link's CoM frame
            R_i = self.kinematic_dict['com_R_symx'][i] # Rotation of CoM frame wrt world
            p_ci = self.kinematic_dict['com_Fks'][i][0:3] # Position of CoM wrt world
            r_ci  = p_ci - p_base_W                       # moment arm about BASE origin

            # Get Jacobians for the center of mass
            J_com_i = self.kinematic_dict['com_geo_J'][i]
            Jv_ci, Jw_ci = J_com_i[0:3, :], J_com_i[3:6, :]

            # Calculate linear and angular acceleration of the CoM using jtimes for J_dot * q_dot
            a_ci = Jv_ci @ q_ddot + ca.jtimes(Jv_ci, q, q_dot)@q_dot
            alpha_i = Jw_ci @ q_ddot + ca.jtimes(Jw_ci, q, q_dot)@q_dot
            
            # Angular velocity of the link
            w_i = Jw_ci @ q_dot

            I_world_i = R_i @ I_ci @ R_i.T
            
            # Rate of change of linear momentum for link i
            f_in   = m_i * a_ci
            # Rate of change of angular momentum for link i, computed about the base origin
            # dL/dt = r x (m*a) + I_world*alpha + w x (I_world*w)
            τ_in   = ca.cross(r_ci, f_in) + I_world_i @ alpha_i + ca.cross(w_i, I_world_i @ w_i)

            total_inertial_force  += f_in
            total_inertial_torque += τ_in
        
            # Gravity force and torque contribution from link i
            f_g = m_i * vec_g
            total_gravity_force  += f_g
            total_gravity_torque += ca.cross(r_ci, f_g)

        # Wrench from external force, transformed to the base frame
        f_ext = F_ext[0:3]
        τ_ext_tip = F_ext[3:6]
        p_tip = self.kinematic_dict['Fks'][-1][0:3]
        r_tip     = p_tip - p_base_W
        τ_ext = ca.cross(r_tip, f_ext) + τ_ext_tip
        
        # From Newton-Euler: Σ F_external = dP/dt  => F_base + F_gravity + F_ext = F_inertial
        base_f  = total_inertial_force  - total_gravity_force  - f_ext
        # From Newton-Euler: Σ τ_external = dL/dt => τ_base + τ_gravity + τ_ext = τ_inertial
        base_τ = total_inertial_torque - total_gravity_torque - τ_ext
    
        # Combine into the final base wrench vector
        W_base = ca.vertcat(base_f, base_τ)

        return W_base


    def build_model(self, root, tip, floating_base=False):
        """
        Constructs the symbolic model for the robot's dynamics.

        This method calculates the key components of the Lagrangian dynamics model:
        the inertia matrix (D), the Coriolis/centrifugal matrix (C), and the
        gravity vector (g). These are stored as instance attributes for later use.

        Args:
            root (str): The name of the root link of the kinematic chain.
            tip (str): The name of the tip link of the kinematic chain.
            floating_base (bool): If True, models the base as a floating body.
        
        Returns:
            A tuple containing all the calculated dynamic components.
        """
        # Step 1: Build the fundamental kinematic model (transforms, Jacobians, etc.)
        self.kinematic_dict = self._kinematics(root, tip, floating_base)
        
        # Step 2: Extract necessary symbolic variables and parameters from the model
        c_parms, m_params, I_params, fv_coeff, fs_coeff, vec_g, r_com_body, m_p ,q, q_dot, q_dotdot, tau , base_pose, world_pose = self.kinematic_dict['parameters']
        n_joints = self.kinematic_dict['n_joints']

        # Step 3: Calculate energy components based on the Lagrangian formulation
        self.K, self.D = self._kinetic_energy()
        self.P = self._potential_energy()
        self.L = self.K - self.P

        # Step 4: Derive the dynamic matrices from the energy components
        # Gravity vector is the gradient of potential energy
        self.g = self._build_gravity_term(self.P, q)
        
        # friction effects
        self.B = self._build_friction_term(fv_coeff, fs_coeff , q_dot)
        
        # Coriolis matrix is derived from the inertia matrix and joint velocities
        self.C = self._build_coriolis_centrifugal_matrix(q, q_dot, self.D)
        self.D_dot = self._build_D_dot(q, q_dot, self.D)
        self.N = self._build_N(q, q_dot, self.D)
        self.id_D, self.id_C, self.id_K, self.id_P, self.id_g, self.Y, self.id_theta, self.physical_plausibility_matrices = self._build_sys_regressor(q, q_dot, q_dotdot)
        # total energy of the system
        self.H = self.K + self.P
        # input. power of the system
        self.H_dot = q_dot.T@tau
        
        self.base_Rct = self._compute_base_reaction()

        # Step 5: Perform assertions to ensure matrix dimensions are consistent
        assert self.D.shape == (n_joints, n_joints), f"Inertia matrix D has incorrect shape: {self.D.shape}"
        assert self.C.shape == (n_joints, n_joints), f"Coriolis matrix C has incorrect shape: {self.C.shape}"
        assert self.D_dot.shape == (n_joints, n_joints), f"matrix D_dot has incorrect shape: {self.D_dot.shape}"
        assert self.N.shape == (n_joints, n_joints), f"matrix N has incorrect shape: {self.N.shape}"
        assert self.g.shape == (n_joints, 1), f"Gravity vector g has incorrect shape: {self.g.shape}"
        assert self.base_Rct.shape == (6, 1), f"base_Rct vector has incorrect shape: {self.base_Rct.shape}"
        
        F_payload = self._payload_wrench_from_mass( m_p, r_com_body)
        J_tip = self.kinematic_dict["geo_J"][-1]   # 6×n geometric Jacobian at the tool
        self.qdd = self._build_forward_dynamics(self.D, self.C, q_dot, self.g, self.B, tau, J_tip, F_payload)
        assert self.qdd.shape == (n_joints, 1), f"Forward dynamics vector qdd has incorrect shape: {self.qdd.shape}"

        self.F_next = self.forward_simulation()
        
        self.joint_torque = self._build_inverse_dynamics(self.D, self.C, q_dotdot, q_dot, self.g, self.B, J_tip, F_payload)
        assert self.joint_torque.shape == (n_joints, 1), f"Inverse dynamics vector qdd has incorrect shape: {self.joint_torque.shape}"

        return self.kinematic_dict, self.K, self.P, self.L, self.D, self.C, self.g, self.B, self.qdd, self.joint_torque, self._sys_id_coeff, F_payload

    @property
    @require_built_model
    def get_forward_simulation(self):
        """Returns the forward dynamics of the system."""
        return self.F_next
    
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
    def get_inverse_dynamics(self):
        """Returns the inverse dynamics for the system."""
        return self.joint_torque
    
    @property
    @require_built_model
    def get_forward_dynamics(self):
        """Returns the forward dynamics for the system."""
        return self.qdd

    @property
    @require_built_model
    def get_inertia_matrix(self):
        """Returns the inertia matrix of the system."""
        return self.D

    @property
    @require_built_model
    def get_coriolis_centrifugal_matrix(self):
        """Returns the Coriolis and centrifugal matrix of the system."""
        return self.C

    @property
    @require_built_model
    def get_gravity_vector(self):
        """Returns the gravity vector of the system."""
        return self.g
    
    @property
    @require_built_model
    def get_friction_vector(self):
        """Returns the friction vector of the system."""
        return self.B
    
    @property
    @require_built_model
    def get_N(self):
        """Returns the N = D_dot-2C of the system."""
        return self.N

    @property
    @require_built_model
    def get_D_dot_2C(self):
        """Returns the N = D_dot-2C of the system."""
        return self.D_dot - 2*self.C

    @property
    @require_built_model
    def get_total_energy(self):
        """Returns the total energy of the system."""
        return self.H
  
    @property
    @require_built_model
    def get_total_power(self):
        """Returns the input power of the system."""
        return self.H_dot
    
    @property
    @require_built_model
    def get_base_Reaction(self):
        """Returns the manipulation base reactions."""
        return self.base_Rct