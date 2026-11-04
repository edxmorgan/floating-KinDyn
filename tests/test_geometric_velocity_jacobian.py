"""Finite-difference rigid motion and virtual-work checks, independent of Euler rates."""
import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from system.robot import RobotDynamics
from system.utils.transformation_matrix import T_from_xyz_rpy


@pytest.mark.parametrize('rpy', [[.4, -.6, .8], [.2, np.pi/2, -.4], [0., 0., 0.]])
def test_world_twist_jacobian_and_wrench(rpy):
    q = ca.SX.sym('q', 2)
    base = ca.SX.sym('base', 6)
    local = T_from_xyz_rpy([.2, -.1, .3], [0., q[0], 0.]) @ T_from_xyz_rpy([.4+q[1], 0., .1], [0., 0., 0.])
    world = T_from_xyz_rpy(base[:3], base[3:]) @ local
    outputs = RobotDynamics(use_jit=False)._compute_Fk_and_jacobians(q, [world], base)
    evaluate = ca.Function('evaluate', [q, base], [outputs[9][-1]])
    local_fk = ca.Function('local_fk', [q], [local])
    angles = np.array([.3, .1])
    xyz = np.array([.7, -.3, 1.2])
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    J = np.array(evaluate(angles, np.r_[xyz, rpy]))
    nu = np.array([.2, -.1, .3, .4, .2, -.3, .5, -.2])
    def pose(t):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(t*nu[3:6]).as_matrix() @ rotation
        transform[:3, 3] = xyz+t*nu[:3]
        return transform @ np.array(local_fk(angles+t*nu[6:]))
    h = 1e-6
    plus, minus, now = pose(h), pose(-h), pose(0.)
    velocity = (plus[:3, 3]-minus[:3, 3])/(2*h)
    skew = ((plus[:3, :3]-minus[:3, :3])/(2*h)) @ now[:3, :3].T
    twist = np.r_[velocity, skew[2,1], skew[0,2], skew[1,0]]
    np.testing.assert_allclose(J @ nu, twist, atol=1e-8)
    wrench = np.array([1., -2., .7, .3, -.4, .5])
    assert (J.T @ wrench) @ nu == pytest.approx(wrench @ twist, abs=1e-8)
    # A pure force creates exactly the expected base-origin moment.
    force = wrench[:3]
    mapped = J.T @ np.r_[force, [0.]*3]
    np.testing.assert_allclose(mapped[:3], force, atol=1e-10)
    np.testing.assert_allclose(mapped[3:6], np.cross(now[:3,3]-xyz, force), atol=1e-10)


def test_notebook_world_angular_rate_map():
    import json
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[1]/'usage/robot_inverse_kinematics_analysis.ipynb').read_text())
    base = ca.SX.sym('base', 6)
    namespace = dict(ca=ca, world_pose=base, omega_base_W=ca.SX.sym('omega', 3))
    exec(''.join(notebook['cells'][5]['source']), namespace)
    evaluate = ca.Function('rate_map', [base], [namespace['Ts']])
    rpy = np.array([.4, -.6, .8])
    rate = np.array([.2, -.3, .5])
    h = 1e-6
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    derivative = (Rotation.from_euler('xyz', rpy+h*rate).as_matrix()
                  - Rotation.from_euler('xyz', rpy-h*rate).as_matrix())/(2*h)
    skew = derivative @ rotation.T
    np.testing.assert_allclose(np.array(evaluate(np.r_[np.zeros(3), rpy])) @ rate,
                               [skew[2,1], skew[0,2], skew[1,0]], atol=1e-9)


def test_actual_arm_urdf_all_jacobian_columns():
    from pathlib import Path
    robot = RobotDynamics(use_jit=False)
    robot.from_file(str(Path(__file__).parents[1]/'usage/urdf/reach_alpha_5/alpha_5_robot.urdf'))
    model = robot._kinematics('base_link', 'alpha_standard_jaws_base_link', floating_base=True)
    parameters = model['parameters']
    q, mount, world, tip = parameters[10], parameters[14], parameters[15], parameters[16]
    Jfun = ca.Function('actual_J', [q, mount, world, tip], [model['geo_J_full'][-1]])
    local = ca.Function('actual_local', [q, mount, world, tip], [model['i_X_0s'][-1]])
    mount_value = [.19, 0., -.12, np.pi, 0., 0.]
    tip_value = [0., 0., .04, 0., 0., 0.]
    rng = np.random.default_rng(714)
    max_error = 0.
    for case in range(20):
        joints = rng.uniform(.2, 2., 4)
        pose = rng.uniform(-1., 1., 6)
        if case == 0:
            pose[4] = np.pi/2
        rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
        J = np.array(Jfun(joints, mount_value, pose, tip_value))
        assert J.shape == (6, 10)
        for column in range(10):
            nu = np.eye(10)[column]
            def moved(t):
                T = np.eye(4)
                T[:3,:3] = Rotation.from_rotvec(t*nu[3:6]).as_matrix() @ rotation
                T[:3,3] = pose[:3]+t*nu[:3]
                return T @ np.array(local(joints+t*nu[6:], mount_value, np.zeros(6), tip_value))
            h = 1e-6
            plus, minus, now = moved(h), moved(-h), moved(0.)
            derivative = (plus-minus)/(2*h)
            skew = derivative[:3,:3] @ now[:3,:3].T
            numerical = np.r_[derivative[:3,3], skew[2,1], skew[0,2], skew[1,0]]
            max_error = max(max_error, float(np.max(np.abs(J[:,column]-numerical))))
            np.testing.assert_allclose(J[:,column], numerical, atol=1e-8)
    print(f'Actual URDF: 20 poses × 10 columns; maximum absolute error {max_error:.3g}')
