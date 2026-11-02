"""Export endpoint IK NLPs. No plant/controller tuning or embedded joint limits.

Bounds and measured/desired states are runtime inputs. FK is the same exported
function used by simlab to validate end-effector poses. Solutions are endpoints,
not collision-checked trajectories.
"""
import argparse
from pathlib import Path
import casadi as ca


def build_endpoint_ik(fk_path, plugin='ipopt'):
    fk = ca.Function.load(str(fk_path))
    z = ca.MX.sym('configuration', 11)  # world xyz, unit quaternion wxyz, joints
    p = ca.MX.sym('task', 31)  # measured(11), target(7), base(6), tip(6), alignment weight
    link_poses = fk(z[7:11], p[18:24], ca.DM.zeros(6), p[24:30])
    local = link_poses[-1]
    # Original preference: first arm-link +X points along vehicle -X.
    # Evaluate in the vehicle frame; common world rotation cancels. A directed
    # dot-product cost avoids the cross-product's false zero at opposite alignment.
    # alpha_axis_e is the first revolute joint, with identity origin rotation
    # and local Z axis (alpha_5_robot.urdf). Use its rotation directly: the
    # legacy FK matrix-to-quaternion chart has singular derivatives here.
    roll, pitch, yaw = p[21], p[22], p[23]
    cx = ca.cos(yaw)*ca.cos(pitch)
    cy = ca.cos(yaw)*ca.sin(pitch)*ca.sin(roll)-ca.sin(yaw)*ca.cos(roll)
    alignment_cost = 1 + cx*ca.cos(z[7]) + cy*ca.sin(z[7])
    qb = z[3:7]
    w, v = qb[0], qb[1:]
    # Quaternion rotation, constrained to SO(3) by unit norm below.
    rotation = (w*w-ca.dot(v,v))*ca.MX.eye(3) + 2*ca.mtimes(v,v.T) + 2*w*ca.skew(v)
    position = z[:3] + rotation @ local[:3]
    ql = local[3:7] / ca.norm_2(local[3:7])
    actual_q = ca.vertcat(w*ql[0]-ca.dot(v,ql[1:]),
                         w*ql[1:]+ql[0]*v+ca.cross(v,ql[1:]))
    target_q = p[14:18] / ca.norm_2(p[14:18])
    rotation_error = (target_q[0]*actual_q[1:] - actual_q[0]*target_q[1:]
                      - ca.cross(target_q[1:], actual_q[1:]))
    constraints = ca.vertcat(position-p[11:14], rotation_error, ca.dot(qb,qb)-1)
    # Both quaternion signs describe the same rotation: objective is sign invariant.
    objective = (ca.sumsqr(z[:3]-p[:3]) + .02*ca.sumsqr(z[7:]-p[7:11])
                 + .08*(1-ca.dot(qb,p[3:7])**2) + (1-rotation[2,2])
                 + p[30]*alignment_cost)
    # Parameter sensitivities are not used by IK; avoid differentiating the
    # legacy FK quaternion chart with respect to fixed mounting parameters.
    common = {'print_time': False, 'error_on_fail': False, 'calc_lam_p': False}
    if plugin == 'ipopt':
        common.update({'ipopt.print_level':0, 'ipopt.sb':'yes',
                       'ipopt.max_iter':200, 'ipopt.max_cpu_time':.2,
                       'ipopt.tol':1e-7, 'ipopt.constr_viol_tol':1e-7,
                       'ipopt.hessian_approximation':'limited-memory', 'ipopt.bound_relax_factor':0.})
    else:
        common.update({'max_iter':100, 'print_header':False, 'print_iteration':False,
                       'print_status':False, 'qpsol':'qrqp',
                       'qpsol_options':{'print_header':False,'print_iter':False,'error_on_fail':False},
                       'hessian_approximation':'limited-memory',
                       'tol_pr':1e-7,'tol_du':1e-7})
    return ca.nlpsol('endpoint_ik_'+plugin,plugin,{'x':z,'p':p,'f':objective,'g':constraints},common)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fk',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    for plugin in ('ipopt','sqpmethod'):
        output=args.output_dir/f'endpoint_ik_{plugin}.casadi'
        build_endpoint_ik(args.fk,plugin).save(str(output))
        print(output)
