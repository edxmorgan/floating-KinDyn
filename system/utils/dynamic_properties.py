import casadi as ca

def is_symmetric(A, tol=1e-9):
    """Numerical symmetry test."""
    return float(ca.norm_inf(A - A.T)) < tol        # ‖A‑Aᵀ‖∞ < tol

def is_spd(A, sym_tol=1e-9, pd_tol=1e-12):
    """
    Symmetric‑positive‑definite test.
    1. Symmetrises small numerical noise:   A ← ½(A+Aᵀ)
    2. Tries a Cholesky factorisation.
    """
    A_sym = 0.5*(A + A.T)                         # cheap symmetrisation
    if not is_symmetric(A_sym, tol=sym_tol):
        return False

    try:
        ca.chol(A_sym + pd_tol*ca.DM.eye(A_sym.size1()))  # jitter protects near‑singular
        return True
    except RuntimeError:
        return False