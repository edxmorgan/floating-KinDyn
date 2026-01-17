import cvxpy as cp
import numpy as np

class SerialLinksDynamicEstimator:
    def __init__(self, dof, horizon_steps, n_params, theta_prev,
                 fixed_blocks=None):
        self.theta_prev = theta_prev
        self.n = horizon_steps * dof
        self.p = n_params
        self.link_group_len = 13
        self.masses = [0.194, 0.429, 0.115, 0.333]

        self.n_joints = self.p // self.link_group_len
        if dof != self.n_joints or self.p % self.link_group_len != 0:
            raise ValueError("p is not divisible by {self.link_group_len}, check regressor layout")

        def block_slice(j):
            s = j * self.link_group_len
            return slice(s, s + self.link_group_len)
        self.block_slice = block_slice

        self.mass_idx = [j*self.link_group_len + 0 for j in range(self.n_joints)]
        self.fv_idx   = [j*self.link_group_len + 10 for j in range(self.n_joints)]
        self.fc_idx   = [j*self.link_group_len + 11 for j in range(self.n_joints)]
        self.fs_idx   = [j*self.link_group_len + 12 for j in range(self.n_joints)]

        # which joints are fixed as full blocks
        self._fixed_blocks = {}
        fixed_joint_set = set()
        if fixed_blocks:
            for j, vec in fixed_blocks.items():
                v = np.asarray(vec, dtype=float).reshape(self.link_group_len)
                self._fixed_blocks[j] = v
                fixed_joint_set.add(int(j))

        self.x = cp.Variable(shape=(self.p,))
        self.w = cp.Variable(shape=(self.p,))
        self.v = cp.Variable(shape=(self.n,))

        self.A = cp.Parameter(shape=(self.n, self.p))
        self.b = cp.Parameter(shape=(self.n,))
        self.x_hat_prev = cp.Parameter(shape=(self.p,))

        # tau = 2.0
        # obj = cp.Minimize(cp.sum_squares(self.v))
        obj = cp.Minimize(cp.norm1(self.v))
        constr = []

        # friction bounds only for joints that are not fixed
        fv_idx_free = [self.fv_idx[j] for j in range(self.n_joints) if j not in fixed_joint_set]
        fc_idx_free = [self.fc_idx[j] for j in range(self.n_joints) if j not in fixed_joint_set]
        fs_idx_free = [self.fs_idx[j] for j in range(self.n_joints) if j not in fixed_joint_set]
        if fv_idx_free:
            constr += [self.x[fv_idx_free] >= 0.0]
        if fc_idx_free:
            constr += [self.x[fc_idx_free] == 0.0]
        if fs_idx_free:
            constr += [self.x[fs_idx_free] == 0.0]
        # per joint constraints, skip all prior constraints for fixed joints
        for j in range(self.n_joints):
            s = j * self.link_group_len
            bs = self.block_slice(j)

            if j in fixed_joint_set:
                # hard pin entire block, no other constraints on this block
                constr += [self.x[bs] == self._fixed_blocks[j]]
                continue

            # prior constraints for free joints only
            m   = self.x[s + 0]
            mcx = self.x[s + 1]
            mcy = self.x[s + 2]
            mcz = self.x[s + 3]

            r = 0.1
            constr += [
                -r*m <= mcx, mcx <= r*m,
                -r*m <= mcy, mcy <= r*m,
                -r*m <= mcz, mcz <= r*m,
                m == self.masses[j]
            ]

            Jj, Jj1 = self.J_from_block(self.x, s)
            psd_eps = 1e-6
            constr += [Jj >> psd_eps * np.eye(4)]
            constr += Jj1

        # model equations
        constr += [
            # self.x == self.x_hat_prev + self.w,
            self.b == self.A @ self.x + self.v
        ]

        self.prob = cp.Problem(obj, constr)

    def J_from_block(self, th, s):
        m   = th[s + 0]
        mcx = th[s + 1];  mcy = th[s + 2];  mcz = th[s + 3]
        Ixx = th[s + 4];  Iyy = th[s + 5];  Izz = th[s + 6]
        Ixy = th[s + 7];  Ixz = th[s + 8];  Iyz = th[s + 9]
        I_bar = cp.bmat([[Ixx, Ixy, Ixz],
                         [Ixy, Iyy, Iyz],
                         [Ixz, Iyz, Izz]])
        mc = cp.vstack([mcx, mcy, mcz])
        J_ul = 0.5 * cp.trace(I_bar) * np.eye(3) - I_bar
        J = cp.bmat([[J_ul, mc],
                     [mc.T, cp.reshape(m, (1, 1), order="F")]])

        J1 = [Ixx <= Iyy + Izz, Iyy <= Ixx + Izz, Izz <= Ixx + Iyy]
        return J, J1

    def estimate_link_physical_parameters(self, Y_big, tau_big, warm_start):
        self.A.value = Y_big
        self.b.value = tau_big
        self.x_hat_prev.value = self.theta_prev
        self.prob.solve(solver=cp.MOSEK, verbose=False, warm_start=warm_start, ignore_dpp=True)
        x = np.array(self.x.value)
        self.theta_prev = x
        w = np.array(self.w.value)
        v = np.array(self.v.value)
        solve_time = self.prob.solver_stats.solve_time
        return x, v, w, solve_time