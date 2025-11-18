# Manipulator Dynamics 🦾

Symbolic modelling toolkit for floating- or fixed-base manipulators. The library ingests a URDF chain, constructs forward/ inverse kinematics, dynamics, energy terms, and identification regressors as CasADi graphs, and can JIT compile them into C for fast controllers, estimators, or trajectory optimisers.

## Key capabilities

- **Full Lagrangian pipeline** – generates inertia (`D`), Coriolis (`C`), gravity (`g`), friction, energy, power, and payload reaction terms via CasADi symbols.
- **Workspace analysis** – sampling-based AABB and point-cloud reach studies via `RobotDynamics.approximate_workspace` plus precomputed `usage/workspace.npy`.
- **System ID helpers** – exported regressors (`id2sim_params.casadi`, `arm_id_Y.casadi`) and parameter lumping utilities for energy-based identification.
- **Controllers & filters** – CasADi PID builder with gravity feedforward (`system/controllers.py`) and EKF/Sensor fusion examples under `usage/`.
- **Code generation** – produces standalone C (`fk_eval_.c`, `Mnext.c`, etc.) and shared libraries (`libFK.so`, `libmEKF_next.so`) for embedded execution.
- **JIT-ready** – optional Clang-backed CasADi JIT (see `RobotDynamics.jit_func_opts`) for rapid prototyping.

## Repository layout

- `system/robot.py` – main `RobotDynamics` class that parses URDF, builds kinematics/dynamics, computes regressors, and simulates forward dynamics.
- `system/controllers.py` – PID builder and helper utilities that wrap the generated model.
- `system/utils/` – transformation matrices, Jacobian helpers, and casadi-compatible math utilities.
- `usage/` – reference notebooks (`robot_dynamics.ipynb`, `controllers.ipynb`, `joint_ekf.ipynb`), controllers, compiled artifacts, and URDF resources.
- `resources/` – meshes, URDF files, and documentation assets (e.g., `resources/dory.jpg`).
- `usage/urdf/` – manipulators used in the notebooks (`alpha_5_robot.urdf`, etc.).

## Requirements

- Python ≥ 3.10
- [CasADi](https://web.casadi.org/) with optional Clang support for JIT mode
- `numpy`, `scipy`, `matplotlib`
- `urdf-parser-py` (and its `lxml` dependency) for loading URDF chains
- `pyquaternion` / `transforms3d` only if extending rotation helpers
- A C/C++ compiler in `PATH` (Clang recommended when using JIT)

### Example environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install casadi numpy scipy matplotlib urdf-parser-py transforms3d
```

CasADi JIT expects Clang on Linux/macOS (`CC=clang`). Update `RobotDynamics.jit_func_opts` if a different compiler is preferred.

## Quickstart

```python
from system.robot import RobotDynamics

robot = RobotDynamics(use_jit=True)
robot.from_file("resources/urdf/alpha_5_robot.urdf")
robot.build_model("base_link", "alpha_standard_jaws_base_link", floating_base=False)

# Access generated functions
fk = robot.get_kinematic_dict["Fks"][-1]        # terminal FK pose (symbolic)
M = robot.get_inertia_matrix                      # D(q)
C = robot.get_coriolis_centrifugal_matrix         # C(q, qdot)
G = robot.get_gravity_vector                      # g(q)
qdd = robot.get_forward_dynamics                  # forward dynamics with friction
Y = robot.kinematic_dict["Y"]                     # identification regressor

# Evaluate FK numerically
fk_func = robot.kinematic_dict["fk_func"]         # casadi.Function built during _kinematics
pose = fk_func(q_sample, base_pose, world_pose)
```

`RobotDynamics.build_model` also assembles friction (`B`), Baumgarte constraints, payload wrench models, and workspace helpers. Forward and inverse dynamics functions are available via the documented properties.

## Controllers & observers

`system/controllers.py` exposes `RobotControllers.build_arm_pid()`, which wraps the generated feedforward torque (`id_g`) with elementwise PID gains, saturation limits, and integral buffers:

```python
from system.robot import RobotDynamics
from system.controllers import RobotControllers

robot = RobotDynamics()
robot.from_file("resources/urdf/alpha_5_robot.urdf")
robot.build_model("base_link", "alpha_standard_jaws_base_link")

controllers = RobotControllers(robot.get_n_joints("base_link", "alpha_standard_jaws_base_link"), robot)
pid = controllers.build_arm_pid()
u_cmd, err, sum_e_next = pid(q, q_dot, q_ref, Kp, Ki, Kd, sum_e, dt, u_max, u_min, controllers.sim_params)
```

Extended Kalman Filter helpers (`libmEKF_next.so`, `joint_ekf.ipynb`) demonstrate how to fuse joint sensors with the generated dynamics. The `usage/controllers.ipynb` notebook ties PID control together with the UVMS base pose alignment parameters defined in `usage/alpha_reach.py`.

## Workspace analysis & reach studies

`RobotDynamics.approximate_workspace` draws uniform joint samples within specified limits, evaluates FK, and returns both an AABB and the sampled point cloud. Reference limits for the 4-DOF Alpha manipulator live in `usage/alpha_reach.py`, and precomputed reach data is stored at `usage/workspace.npy` for quick plotting.

## Identification & code generation assets

- `usage/arm_id_Y.casadi`, `usage/id2sim_params.casadi` – exported regressors and mappings used alongside `system._build_sys_regressor`.
- `usage/fk_eval_.c`, `usage/fk_com_eval.c`, `usage/Mnext.c`, `usage/EKFnext.c` – generated C for FK, COM FK, mass matrix propagation, and EKF updates. Corresponding `.so` files in `usage/` are ready for use in external simulators.

## Implementation status

- [x] Forward kinematics (Euler & quaternion forms)
- [ ] Inverse kinematics (analytic solver placeholder)
- [x] Workspace analysis (sampling-based AABB + cloud)
- [x] Forward dynamics (with payload and friction terms)
- [x] Inverse dynamics (Lagrange–Euler)
- [x] Energy & Lagrangian terms
- [x] Identification helpers (energy-based regressors)
- [x] Kalman filter helpers
- [x] JIT support
- [ ] GPU acceleration
- [x] C/C++ code generation
- [ ] Floating-base navigation & coordination primitives

## References

- Roy Featherstone. *Robot Dynamics Algorithms*. Kluwer, 1987.
- M. W. Spong, S. Hutchinson, M. Vidyasagar. *Robot Modeling and Control*. Wiley, 2006.
- Bruno Siciliano et al. *Robotics: Modelling, Planning and Control*. Springer, 2010.

## Caution ⚠️

The toolkit is experimental and has only been validated on a limited set of underwater manipulators. Review generated models, especially friction and payload terms, before deploying in safety-critical control loops. Contributions, validation reports, and bug fixes are encouraged.
