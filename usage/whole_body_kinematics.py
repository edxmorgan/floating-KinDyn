import os
import sys
import importlib.util
import numpy as np
import casadi as ca

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from system.robot import RobotDynamics
from system.utils.transformation_matrix import rot_from_rpy, rotation_matrix_to_quaternion


DEFAULT_CONFIG = {
    "max_iters": 40,
    "min_iters": 0,
    "tol_pos": 1e-3,
    "tol_rot": 1e-3,
    "damping": 1e-3,
    "joint_weight": 1.0,
    "base_weight_near": 50.0,
    "base_weight_far": 5.0,
    "base_rot_scale": 0.5,
    "proximity_radius": 0.2,
    "step_scale": 0.5,
    "joint_avoidance_gain": 0.0,
    "joint_avoidance_margin": 0.1,
    "base_avoidance_gain": 0.0,
    "base_avoidance_margin": 0.1,
    "base_bounds": None,
    "workspace_check": False,
    "workspace_num_samples": 2000,
    "workspace_aabb": None,
    "orientation_mode": "auto",
    "auto_quat_threshold": 1e-3,
    "orientation_weight": 1.0,
    "quat_jacobian_mode": "analytic",
    "quat_jacobian_eps": 1e-6,
}


def _load_alpha_params():
    usage_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(usage_dir, "alpha_reach.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("alpha_reach", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Params


def _wrap_to_pi(angles):
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _pose_error(target_pose, current_pose):
    err = np.asarray(target_pose).reshape((6,)) - np.asarray(current_pose).reshape((6,))
    err[3:6] = _wrap_to_pi(err[3:6])
    return err


def _normalize_quat(q):
    q = np.asarray(q, dtype=float).reshape((4,))
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def _rpy_to_quat(rpy):
    rpy_dm = ca.DM(rpy).reshape((3, 1))
    R = rot_from_rpy(rpy_dm[0], rpy_dm[1], rpy_dm[2])
    q = rotation_matrix_to_quaternion(R, order="wxyz")
    return np.array(q).reshape((4,))


def _quat_left_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _quat_rot_error(q_target, q_current):
    q_target = _normalize_quat(q_target)
    q_current = _normalize_quat(q_current)
    q_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    q_err = _quat_left_matrix(q_target) @ q_inv
    sign = 1.0
    if q_err[0] < 0.0:
        sign = -1.0
        q_err = -q_err
    rot_err = 2.0 * q_err[1:4]
    return rot_err, sign


def _apply_joint_limits(q, joint_limits):
    if joint_limits is None:
        return q
    low = np.array([lim[0] for lim in joint_limits])
    high = np.array([lim[1] for lim in joint_limits])
    return np.minimum(np.maximum(q, low), high)


def _limit_avoidance(q, limits, margin_frac):
    if limits is None or margin_frac <= 0.0:
        return np.zeros_like(q)
    low = np.array([lim[0] for lim in limits])
    high = np.array([lim[1] for lim in limits])
    span = np.maximum(high - low, 1e-9)
    margin = margin_frac * span
    bias = np.zeros_like(q)
    lower_active = q < (low + margin)
    upper_active = q > (high - margin)
    bias[lower_active] = (low[lower_active] + margin[lower_active] - q[lower_active]) / margin[lower_active]
    bias[upper_active] = -(q[upper_active] - (high[upper_active] - margin[upper_active])) / margin[upper_active]
    return bias


def _apply_bounds(x, bounds):
    if bounds is None:
        return x
    x_min = np.array(bounds["min"], dtype=float)
    x_max = np.array(bounds["max"], dtype=float)
    return np.minimum(np.maximum(x, x_min), x_max)


def _position_in_base(target_pose, base_mount, world_pose):
    def rot_from_rpy_numeric(rpy):
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Rz @ Ry @ Rx

    def T_from_xyz_rpy_numeric(xyz, rpy):
        T = np.eye(4)
        T[0:3, 0:3] = rot_from_rpy_numeric(rpy)
        T[0:3, 3] = xyz
        return T

    base_T = T_from_xyz_rpy_numeric(base_mount[0:3], base_mount[3:6])
    world_T = T_from_xyz_rpy_numeric(world_pose[0:3], world_pose[3:6])
    base_world_T = world_T @ base_T
    R = base_world_T[0:3, 0:3]
    t = base_world_T[0:3, 3]
    target = np.array(target_pose[:3], dtype=float)
    return R.T @ (target - t)


def _base_weight(pos_err_norm, proximity_radius, near_weight, far_weight):
    if proximity_radius <= 0.0:
        return near_weight
    alpha = np.clip(pos_err_norm / proximity_radius, 0.0, 1.0)
    return near_weight * (1.0 - alpha) + far_weight * alpha


class WholeBodyIK:
    def __init__(self, robot, root, tip, floating_base=True):
        self.robot = robot
        self.root = root
        self.tip = tip
        self.robot.build_model(root, tip, floating_base=floating_base)

        kdict = self.robot.kinematic_dict
        params = kdict["parameters"]
        self.q_sym = params[10]
        self.base_mount_sym = params[14]
        self.world_pose_sym = params[15]
        self.tip_offset_sym = params[16]

        fk_expr = kdict["Fks"][-1]
        qfk_expr = kdict["qFks"][-1]
        j_arm_expr = kdict["anlyt_J"][-1]
        j_base_expr = ca.jacobian(fk_expr, self.world_pose_sym)
        j_q_arm_expr = ca.jacobian(qfk_expr[3:7], self.q_sym)
        j_q_base_expr = ca.jacobian(qfk_expr[3:7], self.world_pose_sym)

        self.fk_fun = ca.Function(
            "wb_fk",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [fk_expr],
        )
        self.qfk_fun = ca.Function(
            "wb_qfk",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [qfk_expr],
        )
        self.j_arm_fun = ca.Function(
            "wb_j_arm",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [j_arm_expr],
        )
        self.j_base_fun = ca.Function(
            "wb_j_base",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [j_base_expr],
        )
        self.j_q_arm_fun = ca.Function(
            "wb_j_q_arm",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [j_q_arm_expr],
        )
        self.j_q_base_fun = ca.Function(
            "wb_j_q_base",
            [self.q_sym, self.base_mount_sym, self.world_pose_sym, self.tip_offset_sym],
            [j_q_base_expr],
        )
        self._workspace_cache = None

    def forward_kinematics(self, q, base_mount, world_pose, tip_offset):
        pose = self.fk_fun(q, base_mount, world_pose, tip_offset)
        return np.array(pose).reshape((6,))

    def jacobians(self, q, base_mount, world_pose, tip_offset):
        j_arm = np.array(self.j_arm_fun(q, base_mount, world_pose, tip_offset))
        j_base = np.array(self.j_base_fun(q, base_mount, world_pose, tip_offset))
        return j_arm, j_base

    def quaternion_kinematics(self, q, base_mount, world_pose, tip_offset):
        pose = self.qfk_fun(q, base_mount, world_pose, tip_offset)
        pose = np.array(pose).reshape((7,))
        return pose[3:7]

    def quaternion_jacobians(self, q, base_mount, world_pose, tip_offset):
        j_arm = np.array(self.j_q_arm_fun(q, base_mount, world_pose, tip_offset))
        j_base = np.array(self.j_q_base_fun(q, base_mount, world_pose, tip_offset))
        return j_arm, j_base

    def workspace_aabb(self, joint_limits, base_mount, num_samples):
        if self._workspace_cache is not None:
            return self._workspace_cache
        if joint_limits is None:
            return None
        min_pos, max_pos, _ = self.robot.approximate_workspace(
            self.root,
            self.tip,
            joint_limits,
            base_mount,
            floating_base=False,
            num_samples=num_samples,
        )
        self._workspace_cache = (min_pos, max_pos)
        return self._workspace_cache

    def solve(
        self,
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=None,
        config=None,
        return_history=False,
    ):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        q = np.asarray(q_init).astype(float).copy()
        world_pose = np.asarray(world_pose_init).astype(float).copy()
        n_joints = q.shape[0]
        history = []

        orientation_mode = cfg["orientation_mode"].lower()
        if orientation_mode not in ("rpy", "quat", "auto"):
            raise ValueError(f"Unsupported orientation_mode '{orientation_mode}'")

        for _ in range(cfg["max_iters"]):
            pose = self.forward_kinematics(q, base_mount, world_pose, tip_offset)
            mode = orientation_mode
            if orientation_mode == "auto":
                pitch_current = float(pose[4])
                pitch_target = None
                if len(target_pose) == 6:
                    pitch_target = float(target_pose[4])
                use_quat = abs(np.cos(pitch_current)) < cfg["auto_quat_threshold"]
                if pitch_target is not None:
                    use_quat = use_quat or abs(np.cos(pitch_target)) < cfg["auto_quat_threshold"]
                if use_quat:
                    mode = "quat"
                else:
                    mode = "rpy"

            if mode == "rpy":
                err = _pose_error(target_pose, pose)
                pos_err_norm = float(np.linalg.norm(err[0:3]))
                rot_err_norm = float(np.linalg.norm(err[3:6]))
            else:
                pos_err = np.asarray(target_pose[:3], dtype=float) - pose[0:3]
                pos_err_norm = float(np.linalg.norm(pos_err))

                if len(target_pose) >= 7:
                    q_target = _normalize_quat(target_pose[3:7])
                else:
                    q_target = _normalize_quat(_rpy_to_quat(target_pose[3:6]))

                q_current = self.quaternion_kinematics(q, base_mount, world_pose, tip_offset)
                rot_err, sign = _quat_rot_error(q_target, q_current)
                rot_err_norm = float(np.linalg.norm(rot_err))
                err = np.concatenate([pos_err, -rot_err])

            if pos_err_norm < cfg["tol_pos"] and rot_err_norm < cfg["tol_rot"]:
                if len(history) >= cfg["min_iters"]:
                    break

            j_arm, j_base = self.jacobians(q, base_mount, world_pose, tip_offset)

            if mode == "rpy":
                jac = np.concatenate([j_arm, j_base], axis=1)
            else:
                j_q_arm, j_q_base = self.quaternion_jacobians(
                    q, base_mount, world_pose, tip_offset
                )
                j_pos = np.concatenate([j_arm[0:3, :], j_base[0:3, :]], axis=1)
                if cfg["quat_jacobian_mode"] == "fd":
                    eps = cfg["quat_jacobian_eps"]
                    j_rot = np.zeros((3, n_joints + 6))
                    for i in range(n_joints):
                        q_p = q.copy()
                        q_m = q.copy()
                        q_p[i] += eps
                        q_m[i] -= eps
                        q_p_err, _ = _quat_rot_error(
                            q_target,
                            self.quaternion_kinematics(
                                q_p, base_mount, world_pose, tip_offset
                            ),
                        )
                        q_m_err, _ = _quat_rot_error(
                            q_target,
                            self.quaternion_kinematics(
                                q_m, base_mount, world_pose, tip_offset
                            ),
                        )
                        j_rot[:, i] = (q_p_err - q_m_err) / (2.0 * eps)
                    for i in range(6):
                        w_p = world_pose.copy()
                        w_m = world_pose.copy()
                        w_p[i] += eps
                        w_m[i] -= eps
                        w_p_err, _ = _quat_rot_error(
                            q_target,
                            self.quaternion_kinematics(
                                q, base_mount, w_p, tip_offset
                            ),
                        )
                        w_m_err, _ = _quat_rot_error(
                            q_target,
                            self.quaternion_kinematics(
                                q, base_mount, w_m, tip_offset
                            ),
                        )
                        j_rot[:, n_joints + i] = (w_p_err - w_m_err) / (2.0 * eps)
                else:
                    l_target = _quat_left_matrix(q_target)
                    d_inv = np.diag([1.0, -1.0, -1.0, -1.0])
                    j_q = np.concatenate([j_q_arm, j_q_base], axis=1)
                    j_q_err = sign * (l_target @ d_inv @ j_q)
                    j_rot = 2.0 * j_q_err[1:4, :]
                jac = np.vstack([j_pos, j_rot])

            base_w = _base_weight(
                pos_err_norm,
                cfg["proximity_radius"],
                cfg["base_weight_near"],
                cfg["base_weight_far"],
            )
            workspace_ok = True
            if cfg["workspace_check"]:
                if cfg["workspace_aabb"] is None:
                    cfg["workspace_aabb"] = self.workspace_aabb(
                        joint_limits,
                        base_mount,
                        cfg["workspace_num_samples"],
                    )
                if cfg["workspace_aabb"] is not None:
                    min_pos, max_pos = cfg["workspace_aabb"]
                    pos_base = _position_in_base(target_pose, base_mount, world_pose)
                    workspace_ok = bool(np.all(pos_base >= min_pos) and np.all(pos_base <= max_pos))
                    if not workspace_ok:
                        base_w = min(base_w, cfg["base_weight_far"])

            base_rot_w = base_w * cfg["base_rot_scale"]
            weights = np.concatenate(
                [
                    np.full(n_joints, cfg["joint_weight"]),
                    np.full(3, base_w),
                    np.full(3, base_rot_w),
                ]
            )
            weight_mat = np.diag(weights)

            err[3:6] *= cfg["orientation_weight"]
            jac[3:6, :] *= cfg["orientation_weight"]

            lhs = jac.T @ jac + (cfg["damping"] ** 2) * weight_mat
            rhs = jac.T @ err
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            delta *= cfg["step_scale"]
            joint_bias = _limit_avoidance(q, joint_limits, cfg["joint_avoidance_margin"])
            if cfg["joint_avoidance_gain"] != 0.0:
                delta[:n_joints] += cfg["joint_avoidance_gain"] * joint_bias

            if cfg["base_bounds"] is not None and cfg["base_avoidance_gain"] != 0.0:
                base_bias = _limit_avoidance(
                    world_pose,
                    list(zip(cfg["base_bounds"]["min"], cfg["base_bounds"]["max"])),
                    cfg["base_avoidance_margin"],
                )
                delta[n_joints:] += cfg["base_avoidance_gain"] * base_bias

            q = _apply_joint_limits(q + delta[:n_joints], joint_limits)
            world_pose = world_pose + delta[n_joints:]
            world_pose = _apply_bounds(world_pose, cfg["base_bounds"])
            world_pose[3:6] = _wrap_to_pi(world_pose[3:6])

            if return_history:
                history.append(
                    {
                        "pos_err": pos_err_norm,
                        "rot_err": rot_err_norm,
                        "step_arm": float(np.linalg.norm(delta[:n_joints])),
                        "step_base": float(np.linalg.norm(delta[n_joints:])),
                        "base_weight": float(base_w),
                        "workspace_ok": bool(workspace_ok),
                        "joint_bias_norm": float(np.linalg.norm(joint_bias)),
                        "orientation_mode": mode,
                    }
                )

        info = {
            "pos_err": pos_err_norm,
            "rot_err": rot_err_norm,
            "iters": len(history) if return_history else None,
        }
        return q, world_pose, info, history


def _default_demo():
    params = _load_alpha_params()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if params:
        root = params.root
        tip = params.tip
        base_mount = np.array(params.base_T0_new, dtype=float)
        joint_limits = params.joint_limits
        urdf_path = os.path.join(repo_root, params.relative_urdf_path.lstrip("/"))
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(repo_root, "usage/urdf/reach_alpha_5/alpha_5_robot.urdf")
    else:
        root = "base_link"
        tip = "alpha_standard_jaws_base_link"
        base_mount = np.zeros(6)
        joint_limits = None
        urdf_path = os.path.join(repo_root, "resources/urdf/alpha_5_robot.urdf")

    robot = RobotDynamics(use_jit=False)
    robot.from_file(urdf_path)

    wbik = WholeBodyIK(robot, root, tip, floating_base=True)
    tip_offset = np.zeros(6)
    world_pose_init = np.zeros(6)

    if joint_limits:
        joint_min = np.array([lim[0] for lim in joint_limits])
        joint_max = np.array([lim[1] for lim in joint_limits])
        q_init = 0.5 * (joint_min + joint_max)
    else:
        q_init = np.zeros(wbik.q_sym.shape[0])

    target_pose = np.array([0.35, 0.0, -0.1, np.pi, 0.0, 0.0])
    q_sol, base_sol, info, history = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        return_history=True,
    )

    print("final pos err:", info["pos_err"])
    print("final rot err:", info["rot_err"])
    print("q_sol:", q_sol)
    print("base_world_sol:", base_sol)
    if history:
        print("last step arm/base:", history[-1]["step_arm"], history[-1]["step_base"])


if __name__ == "__main__":
    _default_demo()
