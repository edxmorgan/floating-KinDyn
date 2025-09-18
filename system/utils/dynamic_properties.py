import casadi as ca
import numpy as np

def is_symmetric(A, tol=1e-12):
    return A.size1() == A.size2() and float(ca.norm_inf(A - A.T)) <= tol

def is_spd_strict(A, sym_tol=1e-12):
    """
    Test the original matrix A for SPD:
    - Require near-symmetry (no symmetrisation step)
    - Cholesky on A (no jitter)
    """
    if not is_symmetric(A, sym_tol):
        return False
    try:
        ca.chol(A)   # will fail unless A is SPD
        return True
    except RuntimeError:
        return False

def is_spd_sylvester(A, sym_tol=1e-12, det_tol=0.0):
    if not is_symmetric(A, sym_tol):
        return False
    n = A.size1()
    for k in range(1, n+1):
        if float(ca.det(A[:k, :k])) <= det_tol:
            return False
    return True

def min_eigval(A):
    M = np.array(A)
    w = np.linalg.eigvalsh(M) # symmetric eigensolver
    return float(w.min())

def is_spd_eigs(A, sym_tol=1e-12, eig_tol=0.0):
    if not is_symmetric(A, sym_tol):
        return False
    return min_eigval(A) > eig_tol

def is_skew_symmetric(A, tol=1e-9):
    """Numerical skew‑symmetry test: ‖A + Aᵀ‖∞ < tol."""
    return float(ca.norm_inf(A + A.T)) < tol

def lock_mask_from_indices(n_joints, locked_idx):
    """
    locked_idx, iterable of joint indices to lock, zero based.
    Returns an SX column vector of length n_joints with 1 for locked.
    """
    m = np.zeros((n_joints, 1))
    for i in locked_idx:
        m[i, 0] = 1.0
    return ca.DM(m)  # DM is fine to pass at call time