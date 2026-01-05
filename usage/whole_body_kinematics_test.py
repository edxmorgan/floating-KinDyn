import os
import sys
import json
import csv
import numpy as np
import casadi as ca

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from system.robot import RobotDynamics
from usage.whole_body_kinematics import (
    WholeBodyIK,
    _load_alpha_params,
    _pose_error,
    _normalize_quat,
    _rpy_to_quat,
    _quat_left_matrix,
)
def _rot_from_rpy_numeric(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _T_from_xyz_rpy_numeric(xyz, rpy):
    T = np.eye(4)
    T[0:3, 0:3] = _rot_from_rpy_numeric(rpy)
    T[0:3, 3] = xyz
    return T


def _snapshot_dir():
    path = os.path.join(REPO_ROOT, "usage/whole_body_snapshots")
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _build_robot():
    params = _load_alpha_params()
    if params:
        root = params.root
        tip = params.tip
        base_mount = np.array(params.base_T0_new, dtype=float)
        joint_limits = params.joint_limits
        urdf_path = os.path.join(REPO_ROOT, params.relative_urdf_path.lstrip("/"))
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(REPO_ROOT, "usage/urdf/reach_alpha_5/alpha_5_robot.urdf")
    else:
        root = "base_link"
        tip = "alpha_standard_jaws_base_link"
        base_mount = np.zeros(6)
        joint_limits = None
        urdf_path = os.path.join(REPO_ROOT, "resources/urdf/alpha_5_robot.urdf")

    robot = RobotDynamics(use_jit=False)
    robot.from_file(urdf_path)
    robot.build_model(root, tip, floating_base=True)
    return robot, root, tip, base_mount, joint_limits


def _build_internal_fk(robot):
    kdict = robot.kinematic_dict
    params = kdict["parameters"]
    q_sym = params[10]
    base_pose_sym = params[14]
    world_pose_sym = params[15]
    tip_offset_sym = params[16]
    fk_expr = kdict["Fks"][-1]
    fk_fun = ca.Function(
        "internal_fk_eval_euler",
        [q_sym, base_pose_sym, world_pose_sym, tip_offset_sym],
        [fk_expr],
    )
    return fk_fun


def _sample_q(joint_limits, size):
    if joint_limits is None:
        return np.zeros(size)
    low = np.array([lim[0] for lim in joint_limits])
    high = np.array([lim[1] for lim in joint_limits])
    return np.random.uniform(low, high)


def _finite_difference(fun, x, eps=1e-6):
    x = np.asarray(x, dtype=float).copy()
    f0 = np.asarray(fun(x)).reshape((6,))
    jac = np.zeros((6, x.shape[0]))
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += eps
        x_m[i] -= eps
        f_p = np.asarray(fun(x_p)).reshape((6,))
        f_m = np.asarray(fun(x_m)).reshape((6,))
        jac[:, i] = (f_p - f_m) / (2.0 * eps)
    return f0, jac


def test_fk_consistency(wbik, fk_internal, q, base_mount, world_pose, tip_offset):
    pose_internal = np.array(fk_internal(q, base_mount, world_pose, tip_offset)).reshape((6,))
    pose_wbik = wbik.forward_kinematics(q, base_mount, world_pose, tip_offset)
    max_err = float(np.max(np.abs(pose_internal - pose_wbik)))
    print("Milestone 1: FK max error:", max_err)
    assert max_err < 1e-8


def test_jacobians_fd(wbik, q, base_mount, world_pose, tip_offset):
    j_arm, j_base = wbik.jacobians(q, base_mount, world_pose, tip_offset)

    def fk_q(x):
        return wbik.forward_kinematics(x, base_mount, world_pose, tip_offset)

    def fk_base(x):
        return wbik.forward_kinematics(q, base_mount, x, tip_offset)

    _, j_arm_fd = _finite_difference(fk_q, q)
    _, j_base_fd = _finite_difference(fk_base, world_pose)

    err_arm = float(np.max(np.abs(j_arm - j_arm_fd)))
    err_base = float(np.max(np.abs(j_base - j_base_fd)))
    print("Milestone 1: J_arm max error:", err_arm)
    print("Milestone 1: J_base max error:", err_base)
    assert err_arm < 5e-4
    assert err_base < 5e-4


def test_ik_error_decreases(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    target_pose = pose_init + np.array([0.05, 0.0, -0.03, 0.0, 0.0, 0.0])
    err0 = np.linalg.norm(_pose_error(target_pose, pose_init))

    q_sol, base_sol, info, history = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        return_history=True,
    )
    pose_final = wbik.forward_kinematics(q_sol, base_mount, base_sol, tip_offset)
    err_final = np.linalg.norm(_pose_error(target_pose, pose_final))
    print("Milestone 2: initial error:", float(err0))
    print("Milestone 2: final error:", float(err_final))
    assert err_final < err0


def test_arm_first(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    near_target = pose_init + np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    far_target = pose_init + np.array([0.5, 0.0, -0.3, 0.0, 0.0, 0.0])

    cfg = {
        "max_iters": 1,
        "base_weight_near": 50.0,
        "base_weight_far": 1.0,
        "proximity_radius": 0.2,
        "step_scale": 1.0,
    }

    _, _, _, hist_near = wbik.solve(
        near_target,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    _, _, _, hist_far = wbik.solve(
        far_target,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )

    near_ratio = hist_near[-1]["step_base"] / max(hist_near[-1]["step_arm"], 1e-9)
    far_ratio = hist_far[-1]["step_base"] / max(hist_far[-1]["step_arm"], 1e-9)
    print("Milestone 3: near base/arm ratio:", float(near_ratio))
    print("Milestone 3: far base/arm ratio:", float(far_ratio))
    assert near_ratio < far_ratio


def test_joint_limit_avoidance(wbik, base_mount, world_pose_init, tip_offset, joint_limits):
    if joint_limits is None:
        print("Milestone 4: joint limit avoidance skipped (no limits).")
        return

    low = np.array([lim[0] for lim in joint_limits])
    high = np.array([lim[1] for lim in joint_limits])
    span = high - low
    q_init = high - 0.05 * span
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)

    cfg = {
        "max_iters": 1,
        "min_iters": 1,
        "joint_avoidance_gain": 0.2,
        "joint_avoidance_margin": 0.2,
        "step_scale": 1.0,
    }
    q_sol, _, _, _ = wbik.solve(
        pose_init,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    moved = np.all(q_sol < q_init)
    print("Milestone 4: joint avoidance moved away:", bool(moved))
    assert moved


def test_base_bounds(wbik, q_init, base_mount, tip_offset, joint_limits):
    world_pose_init = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    base_bounds = {
        "min": [-0.2, -0.2, -0.2, -np.pi / 2, -np.pi / 2, -np.pi / 2],
        "max": [0.2, 0.2, 0.2, np.pi / 2, np.pi / 2, np.pi / 2],
    }
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)

    cfg = {
        "max_iters": 1,
        "min_iters": 1,
        "base_bounds": base_bounds,
        "step_scale": 1.0,
    }
    _, base_sol, _, _ = wbik.solve(
        pose_init,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    base_sol = np.asarray(base_sol)
    min_ok = np.all(base_sol >= np.array(base_bounds["min"]) - 1e-9)
    max_ok = np.all(base_sol <= np.array(base_bounds["max"]) + 1e-9)
    print("Milestone 4: base bounds enforced:", bool(min_ok and max_ok))
    assert min_ok and max_ok


def test_workspace_check(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    if joint_limits is None:
        print("Milestone 5: workspace check skipped (no limits).")
        return

    aabb = wbik.workspace_aabb(joint_limits, base_mount, num_samples=1000)
    if aabb is None:
        print("Milestone 5: workspace check skipped (no AABB).")
        return

    min_pos, max_pos = aabb
    target_base = max_pos + np.array([0.4, 0.0, 0.0])
    base_T = _T_from_xyz_rpy_numeric(base_mount[0:3], base_mount[3:6])
    world_T = _T_from_xyz_rpy_numeric(world_pose_init[0:3], world_pose_init[3:6])
    base_world_T = world_T @ base_T
    target_h = np.array([target_base[0], target_base[1], target_base[2], 1.0])
    target_world = (base_world_T @ target_h).reshape((4,))

    target_pose = np.array([target_world[0], target_world[1], target_world[2], 0.0, 0.0, 0.0])
    cfg = {
        "max_iters": 1,
        "min_iters": 1,
        "workspace_check": True,
        "workspace_num_samples": 1000,
        "base_weight_near": 50.0,
        "base_weight_far": 1.0,
        "step_scale": 1.0,
    }
    _, _, _, hist = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    workspace_ok = hist[-1]["workspace_ok"]
    base_weight = hist[-1]["base_weight"]
    print("Milestone 5: workspace ok:", bool(workspace_ok))
    print("Milestone 5: base weight:", float(base_weight))
    assert not workspace_ok
    assert base_weight <= cfg["base_weight_far"] + 1e-9


def test_workspace_inside(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    if joint_limits is None:
        print("Milestone 5b: workspace inside skipped (no limits).")
        return

    aabb = wbik.workspace_aabb(joint_limits, base_mount, num_samples=1000)
    if aabb is None:
        print("Milestone 5b: workspace inside skipped (no AABB).")
        return

    min_pos, max_pos = aabb
    target_base = 0.5 * (min_pos + max_pos)
    base_T = _T_from_xyz_rpy_numeric(base_mount[0:3], base_mount[3:6])
    world_T = _T_from_xyz_rpy_numeric(world_pose_init[0:3], world_pose_init[3:6])
    base_world_T = world_T @ base_T
    target_h = np.array([target_base[0], target_base[1], target_base[2], 1.0])
    target_world = (base_world_T @ target_h).reshape((4,))
    target_pose = np.array([target_world[0], target_world[1], target_world[2], 0.0, 0.0, 0.0])

    cfg = {
        "max_iters": 1,
        "min_iters": 1,
        "workspace_check": True,
        "workspace_num_samples": 1000,
        "base_weight_near": 50.0,
        "base_weight_far": 1.0,
        "step_scale": 1.0,
    }
    _, _, _, hist = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    workspace_ok = hist[-1]["workspace_ok"]
    print("Milestone 5b: workspace ok:", bool(workspace_ok))
    assert workspace_ok


def test_quaternion_mode(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    rpy_target = pose_init[3:6] + np.array([0.0, 0.0, 0.1])
    q_target = _normalize_quat(_rpy_to_quat(rpy_target))
    target_pose = np.concatenate([pose_init[0:3], q_target])

    cfg = {
        "max_iters": 60,
        "orientation_mode": "quat",
        "orientation_weight": 2.0,
        "quat_jacobian_mode": "analytic",
        "step_scale": 0.7,
    }

    q_init_quat = _normalize_quat(
        wbik.quaternion_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    )
    q_inv_init = np.array(
        [q_init_quat[0], -q_init_quat[1], -q_init_quat[2], -q_init_quat[3]]
    )
    q_err_init = _quat_left_matrix(q_target) @ q_inv_init
    if q_err_init[0] < 0.0:
        q_err_init = -q_err_init
    rot_err_init = 2.0 * q_err_init[1:4]
    err_init = np.linalg.norm(rot_err_init)

    q_sol, base_sol, info, _ = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    pose_final = wbik.forward_kinematics(q_sol, base_mount, base_sol, tip_offset)
    q_final = _normalize_quat(wbik.quaternion_kinematics(q_sol, base_mount, base_sol, tip_offset))
    q_inv = np.array([q_final[0], -q_final[1], -q_final[2], -q_final[3]])
    q_err = _quat_left_matrix(q_target) @ q_inv
    if q_err[0] < 0.0:
        q_err = -q_err
    rot_err = 2.0 * q_err[1:4]
    err_final = np.linalg.norm(rot_err)
    print("Milestone 6: quat initial error:", float(err_init))
    print("Milestone 6: quat final error:", float(err_final))
    assert err_final < err_init


def test_quat_near_singularity(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    rpy_target = np.array([pose_init[3], 1.55, pose_init[5]])
    q_target = _normalize_quat(_rpy_to_quat(rpy_target))
    target_pose = np.concatenate([pose_init[0:3], q_target])

    cfg = {
        "max_iters": 50,
        "orientation_mode": "quat",
        "orientation_weight": 2.0,
        "quat_jacobian_mode": "analytic",
        "step_scale": 0.6,
    }
    q_init_quat = _normalize_quat(
        wbik.quaternion_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    )
    q_inv_init = np.array(
        [q_init_quat[0], -q_init_quat[1], -q_init_quat[2], -q_init_quat[3]]
    )
    q_err_init = _quat_left_matrix(q_target) @ q_inv_init
    if q_err_init[0] < 0.0:
        q_err_init = -q_err_init
    rot_err_init = 2.0 * q_err_init[1:4]
    err_init = np.linalg.norm(rot_err_init)

    q_sol, base_sol, _, _ = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    q_final = _normalize_quat(wbik.quaternion_kinematics(q_sol, base_mount, base_sol, tip_offset))
    q_inv = np.array([q_final[0], -q_final[1], -q_final[2], -q_final[3]])
    q_err = _quat_left_matrix(q_target) @ q_inv
    if q_err[0] < 0.0:
        q_err = -q_err
    rot_err = 2.0 * q_err[1:4]
    err_final = np.linalg.norm(rot_err)
    print("Milestone 6b: quat near-sing initial:", float(err_init))
    print("Milestone 6b: quat near-sing final:", float(err_final))
    assert err_final < err_init


def test_auto_mode_switch(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    target_pose = np.array([pose_init[0], pose_init[1], pose_init[2], 0.0, 1.55, 0.0])
    cfg = {
        "max_iters": 2,
        "min_iters": 2,
        "orientation_mode": "auto",
        "auto_quat_threshold": 0.05,
    }
    _, _, _, hist = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        config=cfg,
        return_history=True,
    )
    mode = hist[-1].get("orientation_mode")
    print("Milestone 6c: auto mode:", mode)
    assert mode == "quat"


def test_random_targets_convergence(wbik, base_mount, world_pose_init, tip_offset, joint_limits):
    np.random.seed(11)
    total = 8
    success = 0

    cfg = {
        "max_iters": 40,
        "step_scale": 0.5,
        "damping": 1e-3,
    }

    for _ in range(total):
        q_init = _sample_q(joint_limits, wbik.q_sym.shape[0])
        pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
        delta_pos = np.random.uniform([-0.08, -0.08, -0.08], [0.08, 0.08, 0.08])
        delta_rpy = np.random.uniform([-0.15, -0.15, -0.15], [0.15, 0.15, 0.15])
        target_pose = pose_init + np.concatenate([delta_pos, delta_rpy])
        err0 = np.linalg.norm(_pose_error(target_pose, pose_init))

        q_sol, base_sol, _, _ = wbik.solve(
            target_pose,
            q_init,
            base_mount,
            world_pose_init,
            tip_offset,
            joint_limits=joint_limits,
            config=cfg,
            return_history=False,
        )
        pose_final = wbik.forward_kinematics(q_sol, base_mount, base_sol, tip_offset)
        err_final = np.linalg.norm(_pose_error(target_pose, pose_final))
        if err_final < err0 and err_final < 0.005:
            success += 1

    rate = success / total
    print("Milestone 7: random convergence rate:", float(rate))
    assert rate >= 0.8


def test_iteration_progress(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits):
    pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
    target_pose = pose_init + np.array([0.06, -0.02, 0.04, 0.0, 0.0, 0.1])
    q_sol, base_sol, _, hist = wbik.solve(
        target_pose,
        q_init,
        base_mount,
        world_pose_init,
        tip_offset,
        joint_limits=joint_limits,
        return_history=True,
    )
    if not hist:
        print("Milestone 7b: iteration progress skipped (no history).")
        return
    errs = [np.hypot(h["pos_err"], h["rot_err"]) for h in hist]
    print("Milestone 7b: iter start/end:", float(errs[0]), float(errs[-1]))
    assert errs[-1] < errs[0]

    snap_dir = _snapshot_dir()
    rows = [
        [i, h["pos_err"], h["rot_err"], h["step_arm"], h["step_base"], h["base_weight"]]
        for i, h in enumerate(hist)
    ]
    _write_csv(
        os.path.join(snap_dir, "iteration_history.csv"),
        rows,
        ["iter", "pos_err", "rot_err", "step_arm", "step_base", "base_weight"],
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot([r[0] for r in rows], [r[1] for r in rows], label="pos_err")
        ax[0].plot([r[0] for r in rows], [r[2] for r in rows], label="rot_err")
        ax[0].set_title("IK Errors")
        ax[0].set_xlabel("iter")
        ax[0].legend()

        ax[1].plot([r[0] for r in rows], [r[3] for r in rows], label="arm_step")
        ax[1].plot([r[0] for r in rows], [r[4] for r in rows], label="base_step")
        ax[1].set_title("Step Norms")
        ax[1].set_xlabel("iter")
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(snap_dir, "iteration_history.png"))
        plt.close(fig)
        print("Milestone 7b: saved iteration_history.png")
    except Exception as exc:
        print("Milestone 7b: plot skipped:", str(exc))


def test_random_targets_convergence_large(wbik, base_mount, world_pose_init, tip_offset, joint_limits):
    np.random.seed(23)
    total = 50000
    success = 0
    rows = []

    cfg = {
        "max_iters": 45,
        "step_scale": 0.5,
        "damping": 1e-3,
    }

    for i in range(total):
        q_init = _sample_q(joint_limits, wbik.q_sym.shape[0])
        pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
        delta_pos = np.random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
        delta_rpy = np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
        target_pose = pose_init + np.concatenate([delta_pos, delta_rpy])
        err0 = np.linalg.norm(_pose_error(target_pose, pose_init))

        q_sol, base_sol, _, hist = wbik.solve(
            target_pose,
            q_init,
            base_mount,
            world_pose_init,
            tip_offset,
            joint_limits=joint_limits,
            config=cfg,
            return_history=True,
        )
        pose_final = wbik.forward_kinematics(q_sol, base_mount, base_sol, tip_offset)
        err_final = np.linalg.norm(_pose_error(target_pose, pose_final))
        iters = len(hist) if hist else 0
        ok = err_final < err0 and err_final < 0.005
        success += int(ok)
        rows.append([i, err0, err_final, iters, int(ok)])

    rate = success / total
    stats = {
        "total": total,
        "success": success,
        "success_rate": rate,
        "mean_final": float(np.mean([r[2] for r in rows])),
        "median_final": float(np.median([r[2] for r in rows])),
        "p95_final": float(np.percentile([r[2] for r in rows], 95)),
        "mean_iters": float(np.mean([r[3] for r in rows])),
    }
    print("Milestone 8: large MC success rate:", float(rate))

    snap_dir = _snapshot_dir()
    _write_csv(
        os.path.join(snap_dir, "monte_carlo_50000.csv"),
        rows,
        ["case", "err_init", "err_final", "iters", "success"],
    )
    _write_json(os.path.join(snap_dir, "monte_carlo_50000.json"), stats)
    assert rate >= 0.9


def test_stress_bounds_and_margins(wbik, base_mount, world_pose_init, tip_offset, joint_limits):
    if joint_limits is None:
        print("Milestone 9: stress test skipped (no limits).")
        return

    low = np.array([lim[0] for lim in joint_limits])
    high = np.array([lim[1] for lim in joint_limits])
    span = high - low
    base_sizes = [0.1, 0.2, 0.3]
    margins = [0.05, 0.1, 0.2]
    total = 0
    violations = 0
    rows = []

    for base_size in base_sizes:
        base_bounds = {
            "min": [-base_size, -base_size, -base_size, -np.pi / 2, -np.pi / 2, -np.pi / 2],
            "max": [base_size, base_size, base_size, np.pi / 2, np.pi / 2, np.pi / 2],
        }
        for margin in margins:
            q_init = high - 0.02 * span
            pose_init = wbik.forward_kinematics(q_init, base_mount, world_pose_init, tip_offset)
            cfg = {
                "max_iters": 5,
                "min_iters": 5,
                "joint_avoidance_gain": 0.2,
                "joint_avoidance_margin": margin,
                "base_bounds": base_bounds,
                "base_avoidance_gain": 0.2,
                "base_avoidance_margin": 0.2,
                "step_scale": 0.7,
            }
            q_sol, base_sol, _, _ = wbik.solve(
                pose_init,
                q_init,
                base_mount,
                world_pose_init,
                tip_offset,
                joint_limits=joint_limits,
                config=cfg,
                return_history=False,
            )
            total += 1
            joint_ok = np.all(q_sol <= high + 1e-9) and np.all(q_sol >= low - 1e-9)
            base_ok = np.all(base_sol >= np.array(base_bounds["min"]) - 1e-9) and np.all(
                base_sol <= np.array(base_bounds["max"]) + 1e-9
            )
            ok = joint_ok and base_ok
            violations += int(not ok)
            rows.append([base_size, margin, int(joint_ok), int(base_ok)])

    print("Milestone 9: stress violations:", int(violations))
    snap_dir = _snapshot_dir()
    _write_csv(
        os.path.join(snap_dir, "stress_bounds_margins.csv"),
        rows,
        ["base_size", "margin", "joint_ok", "base_ok"],
    )
    assert violations == 0


def main():
    np.random.seed(4)
    robot, root, tip, base_mount, joint_limits = _build_robot()
    wbik = WholeBodyIK(robot, root, tip, floating_base=True)
    fk_internal = _build_internal_fk(robot)

    tip_offset = np.zeros(6)
    world_pose = np.zeros(6)
    q = _sample_q(joint_limits, wbik.q_sym.shape[0])

    test_fk_consistency(wbik, fk_internal, q, base_mount, world_pose, tip_offset)
    test_jacobians_fd(wbik, q, base_mount, world_pose, tip_offset)

    q_init = q
    world_pose_init = world_pose
    test_ik_error_decreases(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_arm_first(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_joint_limit_avoidance(wbik, base_mount, world_pose_init, tip_offset, joint_limits)
    test_base_bounds(wbik, q_init, base_mount, tip_offset, joint_limits)
    test_workspace_check(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_workspace_inside(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_quaternion_mode(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_quat_near_singularity(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_auto_mode_switch(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_random_targets_convergence(wbik, base_mount, world_pose_init, tip_offset, joint_limits)
    test_iteration_progress(wbik, q_init, base_mount, world_pose_init, tip_offset, joint_limits)
    test_random_targets_convergence_large(wbik, base_mount, world_pose_init, tip_offset, joint_limits)
    test_stress_bounds_and_margins(wbik, base_mount, world_pose_init, tip_offset, joint_limits)

    print("All milestone tests passed.")


if __name__ == "__main__":
    main()
