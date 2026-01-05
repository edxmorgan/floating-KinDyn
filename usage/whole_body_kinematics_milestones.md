# Whole-Body Kinematics Project (Milestones)

Goal: build a whole-body kinematics solver where the only command is the
end-effector pose, and the system solves for arm joints plus floating-base
xyz/rpy while preferring arm motion when the target is close.

## Milestone 1: Whole-body kinematics plumbing
- Build FK and Jacobians for the end-effector with respect to joints and base
  world pose (xyz/rpy).
- Expose numerical functions in `usage/whole_body_kinematics.py`.
- Acceptance: FK returns a 6D pose and Jacobians have shapes (6 x n) and (6 x 6).

## Milestone 2: Weighted whole-body IK loop
- Implement a damped least-squares IK loop for delta [q, base_xyzrpy].
- Include damping, step scaling, and optional joint-limit clamping.
- Acceptance: pose error decreases from a nominal initial guess.

## Milestone 3: Arm-first coordination
- Add dynamic base weights based on proximity to the target.
- Heavier base penalties when error is small; lighter penalties when far.
- Acceptance: base stays near zero when target is close, base moves when far.

## Milestone 4: Validation and examples
- Provide reachable and unreachable target cases.
- Log per-iteration pose error and base vs arm step norms.
- Acceptance: coordination behavior is visible in logs or simple plots.

## Milestone 5: Extensions (optional)
- Joint-limit avoidance (soft costs) and base velocity bounds.
- Workspace-based reach checks using `approximate_workspace`.
- Optional orientation weighting or quaternion error.
