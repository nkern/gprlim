"""
Linear solvers for Gaussian-process inference: a complex preconditioned conjugate-gradient
solver, batched Kronecker(-Woodbury / -Cholesky) routines, and the general batched dense
Wiener solvers.

Throughout, ``Ct`` is the time-axis kernel (an ``(Ntimes, Ntimes)`` Hermitian covariance,
possibly complex) and ``Cf`` is the frequency-axis kernel (``(Nfreqs, Nfreqs)``), and the
joint signal covariance is the Kronecker product ``Ct (x) Cf``.

The structured Kronecker routines never densify the unraveled
``(Ntimes*Nfreqs) x (Ntimes*Nfreqs)`` covariance:

* :func:`pcg` -- preconditioned CG for Hermitian PD systems, real or complex, through a
  matmul-only interface, with pluggable dense / structured preconditioners
  (:func:`kron_eigen_preconditioner`, :func:`dense_preconditioner`, ...). ``linear_operator``'s
  own CG is real-only (it compares a complex residual norm with ``<``), so this is what
  the complex CarrierKernel covariances need. Accuracy is set by the CG tolerance, not a
  rank cutoff -- the right tool when the kernel's dynamic range is too high to truncate.
* :func:`kron_woodbury_predict` / :func:`kron_woodbury_inpaint` -- a *direct* batched
  Wiener solve via the Woodbury identity on the low-rank Kronecker factor; never iterates,
  but its accuracy is capped by the low-rank cutoff ``rcond`` (a win when the kernels are
  low rank).
* :func:`kron_cholesky` -- the implicit Cholesky factor of a Kronecker product (used by
  :class:`gprlim.kernels.KroneckerKernel` for prior draws / whitening).

The batched dense solvers instead take a single (Nsamples x Nsamples) signal covariance
``C`` shared across a batch of per-row noise diagonals -- the per-axis ('freq'/'time'-mode)
workhorse, and the dense dual of the Kronecker routines above:

* :func:`gpr_invert` (dispatching to :func:`woodbury_batched` / :func:`cholesky_batched`)
  -- the Wiener-mean ``B (C + diag(N_b))^-1 y_b`` per row.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.operators import (
    TriangularLinearOperator,
    KroneckerProductTriangularLinearOperator,
)


# --------------------------------------------------------------------------------------
# complex-data helpers (shared by the dense and Kronecker Wiener solves)
# --------------------------------------------------------------------------------------
# A real covariance acts identically on the real and imaginary parts of complex data, so
# the parts can be stacked and solved as two real systems (cheaper) and recombined; a
# complex covariance couples them and must be solved directly (a real one then promoted).
def stack_ri(x):
    """Stack complex ``x``'s real & imag parts along the leading (row) axis; real ``x`` is
    returned unchanged. (Nrows, ...) complex -> (2*Nrows, ...) real."""
    return torch.cat([x.real, x.imag], 0) if x.is_complex() else x


def unstack_ri(x):
    """Inverse of :func:`stack_ri`: fold a leading [real, imag] stack back into complex.
    (2*Nrows, ...) real -> (Nrows, ...) complex."""
    h = x.shape[0] // 2
    return torch.complex(x[:h], x[h:])


def promote_like(a, ref):
    """Promote real ``a`` to ``ref``'s (complex) dtype so a real covariance can solve a
    complex system directly; a no-op if ``ref`` is real or ``a`` is already complex."""
    return a.to(ref.dtype) if (ref.is_complex() and not a.is_complex()) else a


def truncate(C, rcond):
    r"""
    Low-rank-truncate a Hermitian-PSD covariance via its eigendecomposition.

    Reconstructs ``C`` from only the eigenmodes whose eigenvalue exceeds ``rcond * lambda_max``,
    dropping the rest to exactly zero (a truncated eigendecomposition / pinv-style cutoff):

        C_trunc = Q diag(lambda_i * [lambda_i >= rcond * lambda_max]) Q^H.

    ``rcond`` is the usual relative cutoff of ``numpy``/``torch`` ``pinv``. This is the
    *opposite* of shrinking toward the diagonal: rather than lifting the small eigenvalues to a
    floor (which would give them nonzero Wiener gain), it removes them. In the 2D joint solve,
    applied to the time factor ``Ct`` this keeps only the leading (smooth) time modes -- which
    both *lowers* high-delay leakage (the dropped fast-time modes were the ones coupling into
    the frequency kernel's out-of-band ripple) and *shrinks* the CG problem (fewer Kronecker
    modes above the noise -> faster convergence). See [[joint-eta-delay-confinement]].

    Parameters
    ----------
    C : tensor or LinearOperator
        Hermitian-PSD covariance of shape (..., n, n); a ``LinearOperator`` is densified for the
        eigendecomposition (``C`` is a small per-axis factor). Returns a dense tensor.
    rcond : float or None
        Relative eigenvalue cutoff in (0, 1]: keep modes with ``lambda >= rcond * lambda_max``.
        ``0`` / ``None`` is a no-op (returns ``C`` unchanged); ``rcond = 1`` keeps only the
        leading mode (maximal truncation).

    Returns
    -------
    tensor
        The truncated covariance ``C_trunc`` (rank = number of kept modes).
    """
    if not rcond:                                 # 0 / 0.0 / None -> unchanged
        return C
    if not 0.0 < rcond <= 1.0:
        raise ValueError(f"rcond must be in (0, 1], got {rcond}")
    dense = C.to_dense() if isinstance(C, LinearOperator) else C
    w, Q = torch.linalg.eigh(dense)               # ascending real eigenvalues; w[..., -1] = lam_max
    keep = w >= rcond * w[..., -1:]               # drop modes (incl. numerical negatives) below cutoff
    wk = torch.where(keep, w, torch.zeros_like(w))
    return (Q * wk.unsqueeze(-2)) @ Q.conj().transpose(-1, -2)   # Q diag(wk) Q^H


def shrink(C, rcond):
    r"""
    Shrink a Hermitian-PSD covariance's *spectrum toward flat* (a scaled identity) via its
    eigendecomposition -- the mirror image of :func:`truncate`.

    Instead of dropping the small eigenvalues, it *lifts* them to a floor ``rcond * lambda_max``
    and renormalizes so the trace (hence the marginal variance) is preserved::

        C_shrink = c * Q diag(max(lambda_i, rcond * lambda_max)) Q^H,   c = tr(C) / tr(floored).

    Lifting the near-zero (fast) modes back above the noise gives them nonzero Wiener gain. In the
    2D joint solve, applied to the time factor ``Ct`` this interpolates from the full joint model
    (``rcond`` -> 0) toward the per-``x2``-independent 1D solve: at ``rcond = 1`` the spectrum is
    flattened to ``lambda_max`` everywhere, so (trace-preserved) ``Ct`` becomes ``mean(diag(Ct))*I``
    -- ``Ct (x) Cf = I (x) Cf`` up to amplitude -- which is exactly the 1D inpaint along ``x2``
    (per-``x1`` independent). It is the *opposite direction* from :func:`truncate` (which confines
    toward rank-1). Lifting is inherently front-loaded, so tune ``rcond`` on a log scale (like a
    ``pinv`` cutoff). See [[joint-eta-delay-confinement]].

    Parameters
    ----------
    C : tensor or LinearOperator
        Hermitian-PSD covariance of shape (..., n, n); a ``LinearOperator`` is densified for the
        eigendecomposition. Returns a dense tensor.
    rcond : float or None
        Relative eigenvalue floor in (0, 1]. ``0`` / ``None`` is a no-op (returns ``C``
        unchanged); ``rcond = 1`` fully flattens the spectrum (``C`` proportional to the identity).

    Returns
    -------
    tensor
        The spectrum-shrunk covariance ``C_shrink`` (full rank, trace preserved).
    """
    if not rcond:                                 # 0 / 0.0 / None -> unchanged
        return C
    if not 0.0 < rcond <= 1.0:
        raise ValueError(f"rcond must be in (0, 1], got {rcond}")
    dense = C.to_dense() if isinstance(C, LinearOperator) else C
    w, Q = torch.linalg.eigh(dense)               # ascending real eigenvalues; w[..., -1] = lam_max
    wf = torch.clamp_min(w, rcond * w[..., -1:])  # lift modes below the floor up to it
    wf = wf * (w.sum(-1, keepdim=True) / wf.sum(-1, keepdim=True))   # renormalize trace (preserve variance)
    return (Q * wf.unsqueeze(-2)) @ Q.conj().transpose(-1, -2)   # Q diag(wf) Q^H


# --------------------------------------------------------------------------------------
# preconditioned conjugate gradients
# --------------------------------------------------------------------------------------
def _rdot(a, b):
    """Real part of ``<a, b> = a^H b`` over the last axis, keepdim -> (..., 1)."""
    return (a.conj() * b).sum(-1, keepdim=True).real


def _rnorm(a):
    """Euclidean norm over the last (system) axis, shape (..., 1)."""
    return _rdot(a, a).clamp_min(0).sqrt()


def pcg(A, b, M=None, tol=1e-6, max_iter=1000, x0=None, weight=None):
    """
    Preconditioned conjugate gradients for a Hermitian positive-definite system
    ``A x = b`` (real or complex), operating only through matrix-vector products.

    Parameters
    ----------
    A : callable or tensor or LinearOperator
        The Hermitian PD operator. A callable is used directly as the matvec
        ``v -> A @ v`` (the no-densify path, e.g. :func:`kron_matvec`); a dense tensor or
        linear operator is wrapped automatically. ``A`` may be batched (a different
        diagonal per row of ``b``).
    b : tensor
        Right-hand side(s) of shape (..., n): the last axis is the system, any leading
        axes are independent batched solves.
    M : callable, optional
        Preconditioner applying ``r -> M^-1 r`` (e.g. :func:`kron_eigen_preconditioner`,
        :func:`dense_preconditioner`, :func:`diag_preconditioner`). Default: identity.
    tol : float
        Relative-residual tolerance ``||b - A x|| / ||b||``; stop when every row meets it.
    max_iter : int
        Maximum iterations.
    x0 : tensor, optional
        Initial guess (default zeros).
    weight : tensor, optional
        Per-element weights (..., n) for the residual *norm only* (not the iteration): the stop
        test uses ``sqrt(sum w |r|^2) / sqrt(sum w |b|^2)``. Pass the inverse noise variance
        (``1/noise``) so high-variance (flagged) entries drop out of the convergence test --
        otherwise their unfittable contribution to ``||b||`` makes ``tol`` data-dependent.
        Default: unweighted (equivalent to ``weight`` constant).

    Returns
    -------
    x : tensor
        Solution, same shape as ``b``.
    info : dict
        ``iters`` (iterations run) and ``resid`` (max relative residual over the batch).
    """
    # accept A/M as callables (the no-densify path, e.g. kron_matvec) or as a matrix /
    # linear operator (wrapped into a matvec acting on the last axis of v)
    matvec = A if callable(A) else (lambda v: (A @ v.unsqueeze(-1)).squeeze(-1))
    apply_M = (lambda r: r) if M is None else M
    tiny = torch.finfo(b.real.dtype).tiny
    # residual norm for the stop test: plain Euclidean, or inverse-variance-weighted (so
    # high-noise / flagged entries drop out and `tol` is independent of the flag-var/noise ratio)
    rnorm = _rnorm if weight is None else (
        lambda v: (weight * (v.conj() * v).real).sum(-1, keepdim=True).clamp_min(0).sqrt())

    # r = residual b - A x ; z = preconditioned residual M^-1 r ; p = search direction.
    # rz = <r, z> is the quantity PCG drives to zero (real for Hermitian PD A and M).
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    bnorm = rnorm(b).clamp_min(tiny)                       # denominator of the relative residual
    z = apply_M(r)
    p = z.clone()
    rz = _rdot(r, z)

    k, resid = 0, float((rnorm(r) / bnorm).max())
    if resid <= tol:                                       # x0 already within tolerance
        return x, {'iters': 0, 'resid': resid}
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        # alpha: exact line-minimization step along p (denominator p^H A p is real & > 0)
        alpha = rz / _rdot(p, Ap).clamp_min(tiny)
        x = x + alpha * p                                  # advance the solution
        r = r - alpha * Ap                                 # ... and the residual
        resid = float((rnorm(r) / bnorm).max())            # worst row across the batch
        if resid <= tol:
            break
        z = apply_M(r)
        rz_new = _rdot(r, z)
        # beta = rz_new / rz makes the next direction A-conjugate to the previous ones
        p = z + (rz_new / rz) * p
        rz = rz_new

    return x, {'iters': k, 'resid': resid}


# --------------------------------------------------------------------------------------
# preconditioners  (each returns a callable  r -> M^-1 r)
# --------------------------------------------------------------------------------------
def diag_preconditioner(diag):
    """Jacobi preconditioner ``M^-1 r = r / diag`` for a diagonal ``M`` of shape (..., n)
    (e.g. ``diag = A.diagonal()``): cheap, and tames a flag-heavy (high-variance) noise diagonal."""
    return lambda r: r / diag


def dense_preconditioner(M):
    """Exact dense preconditioner: ``M^-1 r`` via a Cholesky factor of the Hermitian PD
    matrix ``M`` of shape (..., n, n). Factored once, applied with ``cholesky_solve``."""
    L = torch.linalg.cholesky(M)
    return lambda r: torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)


def operator_preconditioner(linop):
    """Preconditioner from any object exposing ``.solve`` (e.g. a ``linear_operator``):
    ``M^-1 r = linop.solve(r)``."""
    return lambda r: linop.solve(r.unsqueeze(-1)).squeeze(-1)


def kron_eigen_preconditioner(Ct, Cf, shift=0.0):
    r"""
    Structured (no-densify) preconditioner for ``M = Ct (x) Cf + shift * I``, via the
    per-axis eigendecompositions -- the natural preconditioner for a Kronecker signal
    covariance plus a (scalar) noise floor. With ``shift`` set to the good-pixel noise
    variance, ``M`` captures the smooth covariance *and its full dynamic range* exactly,
    so CG only has to resolve the (low-rank) flagged correction.

    Uses ``M^-1 vec(R) = vec( Q_t [ (Q_t^H R Q_f^*) / (lam_t lam_f^T + shift) ] Q_f^T )``
    (row-major ``vec``), all via small (Ntimes, Nfreqs) matmuls -- never the
    ``(Ntimes*Nfreqs)^2`` matrix.

    Parameters
    ----------
    Ct : tensor
        Time-axis kernel, Hermitian PSD of shape (Ntimes, Ntimes), real or complex.
    Cf : tensor
        Frequency-axis kernel, Hermitian PSD of shape (Nfreqs, Nfreqs), real or complex.
        ``Ct`` and ``Cf`` are promoted to a common dtype so a complex kernel and a real
        kernel compose.
    shift : float
        Diagonal shift (e.g. the good-pixel noise variance ``sigma^2``).

    Returns
    -------
    callable
        ``r -> M^-1 r`` on vectors of shape (..., Ntimes*Nfreqs).
    """
    # eigendecompose each kernel ONCE: Ct = Qt diag(lam_t) Qt^H, Cf = Qf diag(lam_f) Qf^H.
    # Then  Ct (x) Cf = (Qt (x) Qf) diag(lam_t (x) lam_f) (Qt (x) Qf)^H, so M = Ct (x) Cf +
    # shift*I shares those eigenvectors and has eigenvalues lam_t (x) lam_f + shift.
    dt = torch.promote_types(Ct.dtype, Cf.dtype)
    Ct, Cf = Ct.to(dt), Cf.to(dt)
    lam_t, Qt = torch.linalg.eigh(Ct)
    lam_f, Qf = torch.linalg.eigh(Cf)
    Ntimes, Nfreqs = Ct.shape[-1], Cf.shape[-1]
    denom = lam_t[:, None] * lam_f[None, :] + shift            # M's eigenvalues, (Ntimes, Nfreqs) real
    QtH, Qfc, QfT = Qt.conj().transpose(-1, -2), Qf.conj(), Qf.transpose(-1, -2)

    def apply_Minv(r):
        # M^-1 r in three cheap (Ntimes, Nfreqs) matmuls -- never forms the full matrix:
        R = r.reshape(*r.shape[:-1], Ntimes, Nfreqs)               # unravel to the (t, f) grid
        Rt = (QtH @ R @ Qfc) / denom                               # rotate into eigenbasis, divide by eigenvalues
        return (Qt @ Rt @ QfT).reshape(*r.shape[:-1], Ntimes * Nfreqs)   # rotate back, re-ravel
    return apply_Minv


def kron_blockdiag_preconditioner(Ct, Cf, Df):
    r"""
    Structured (no-densify) preconditioner for ``M = Ct (x) Cf + I (x) diag(Df)`` -- the natural
    preconditioner when the noise is *separable across the outer (time) axis*: whole inner-axis
    (frequency) channels flagged the same way at every time, so ``diag(noise) = I (x) diag(Df)``
    with ``Df`` the per-channel noise (large on fully-flagged channels).

    Unlike the scalar-shift :func:`kron_eigen_preconditioner`, this carries the *full-channel
    flag structure exactly*, so preconditioned CG converges in a handful of iterations -- and in
    a single iteration (it is then the exact inverse) when the noise is exactly separable.

    Mechanism: ``M`` block-diagonalizes in the eigenbasis of ``Ct = Qt diag(lam_t) Qt^H``,
    ``(Qt^H (x) I) M (Qt (x) I) = blockdiag_i( lam_t[i] Cf + diag(Df) )``. Rather than factor each
    of the ``Ntimes`` blocks separately (that costs ``Ntimes`` ``Nfreqs x Nfreqs`` solves *per
    apply* -- which for large ``Nfreqs`` can dominate and make CG net-slower than the scalar-shift
    preconditioner despite fewer iterations), whiten once: ``Ctil = Df^-1/2 Cf Df^-1/2 =
    V diag(nu) V^H`` shares its eigenvectors across every block, so ``B_i^-1 = G diag(1 /
    (lam_t[i] nu + 1)) G^H`` with ``G = Df^-1/2 V``. Then ``M^-1 r`` is a rotate by ``Qt^H`` (outer
    axis) and ``G^H`` (inner axis), an elementwise divide by ``lam_t (x) nu + 1``, and the two
    rotates back -- FOUR small matmuls, the same per-apply cost as :func:`kron_eigen_preconditioner`.
    Every block is Hermitian PD (``Df > 0``, ``lam_t`` clamped at 0 so ``nu >= 0``), so the divisor
    is ``>= 1`` and it never breaks down.

    Parameters
    ----------
    Ct : tensor
        Outer/time-axis kernel, Hermitian PSD (Ntimes, Ntimes), real or complex.
    Cf : tensor
        Inner/frequency-axis kernel, Hermitian PSD (Nfreqs, Nfreqs).
    Df : tensor
        Per-inner-channel noise variance (Nfreqs,), strictly positive -- the time-independent
        part of the noise (fully-flagged channels carry the large value).

    Returns
    -------
    callable
        ``r -> M^-1 r`` on vectors of shape (..., Ntimes*Nfreqs).
    """
    dt = torch.promote_types(Ct.dtype, Cf.dtype)
    Ct, Cf = Ct.to(dt), Cf.to(dt)
    Ntimes, Nfreqs = Ct.shape[-1], Cf.shape[-1]
    lam_t, Qt = torch.linalg.eigh(Ct)
    lam_t = lam_t.clamp_min(0)                                     # PSD blocks even for tiny negative eig
    # whiten Cf by the per-channel noise and diagonalize ONCE: Ctil = Df^-1/2 Cf Df^-1/2 = V nu V^H.
    # every block  B_i = lam_t[i] Cf + diag(Df)  then shares eigenvectors G = Df^-1/2 V, with
    # B_i^-1 = G diag(1 / (lam_t[i] nu + 1)) G^H -- so M^-1 is four matmuls, not Ntimes block solves.
    dinv = Df.rsqrt().to(dt)                                       # Df^-1/2
    nu, V = torch.linalg.eigh(dinv[:, None] * Cf * dinv[None, :])
    G = dinv[:, None] * V                                          # Df^-1/2 V, (Nfreqs, Nfreqs)
    denom = lam_t[:, None] * nu.clamp_min(0)[None, :] + 1.0        # blocks' eigenvalues, (Ntimes, Nfreqs)
    QtH, Gc, GT = Qt.conj().transpose(-1, -2), G.conj(), G.transpose(-1, -2)

    def apply_Minv(r):
        R = r.reshape(*r.shape[:-1], Ntimes, Nfreqs)
        Rt = QtH @ R                                              # (Qt^H (x) I) r -- rotate outer axis
        Xt = ((Rt @ Gc) / denom) @ GT                            # whiten inner axis, divide, un-whiten
        return (Qt @ Xt).reshape(*r.shape[:-1], Ntimes * Nfreqs)  # rotate back
    return apply_Minv


# --------------------------------------------------------------------------------------
# structured Kronecker matvec + high-level preconditioned-CG Wiener solve
# --------------------------------------------------------------------------------------
def kron_matvec(Ct, Cf, diag=None):
    r"""
    Matvec for ``A = Ct (x) Cf [+ diag]`` via the row-major Kronecker identity
    ``(Ct (x) Cf) vec(X) = vec(Ct X Cf^T)`` -- never forms the ``(Ntimes*Nfreqs)^2`` matrix.

    Parameters
    ----------
    Ct, Cf : tensor
        Time / frequency kernels of shape (Ntimes, Ntimes) / (Nfreqs, Nfreqs), real or
        complex (promoted to a common dtype).
    diag : tensor, optional
        Per-element diagonal to add (e.g. the noise variance), shape (..., Ntimes*Nfreqs).

    Returns
    -------
    callable
        ``v -> A @ v`` on vectors of shape (..., Ntimes*Nfreqs).
    """
    dt = torch.promote_types(Ct.dtype, Cf.dtype)
    Ct, Cf = Ct.to(dt), Cf.to(dt)
    Ntimes, Nfreqs, CfT = Ct.shape[-1], Cf.shape[-1], Cf.transpose(-1, -2)

    def mv(v):
        X = v.reshape(*v.shape[:-1], Ntimes, Nfreqs)
        out = (Ct @ X @ CfT).reshape(*v.shape[:-1], Ntimes * Nfreqs)
        return out if diag is None else out + diag * v
    return mv


def kron_wiener_cg(Ct, Cf, noise, y, shift=None, tol=1e-6, max_iter=1000, x0=None, n_threads=1,
                   return_alpha=False, precond='eigen'):
    r"""
    Complex Wiener posterior mean  ``m = (Ct (x) Cf) (Ct (x) Cf + diag(noise))^-1 y``  via
    preconditioned CG: structured throughout (never densifies the ``(Ntimes*Nfreqs)^2``
    covariance), Kronecker-eigen preconditioned, and accurate to ``tol`` -- no low-rank
    truncation (cf. :func:`kron_woodbury_predict`, whose accuracy is capped by ``rcond``).

    Parameters
    ----------
    Ct, Cf : tensor
        Time / frequency kernels of shape (Ntimes, Ntimes) / (Nfreqs, Nfreqs), Hermitian
        PSD (``Ct`` may be complex).
    noise : tensor
        Per-pixel noise variance, shape (..., Ntimes, Nfreqs); flagged pixels carry a
        large value. A leading axis of 1 (shared across the batch) broadcasts.
    y : tensor
        Data, shape (..., Ntimes, Nfreqs).
    shift : float, optional
        Preconditioner diagonal shift; default the smallest noise value (the good-pixel
        variance), which makes the preconditioner capture the bulk of ``A``.
    precond : str, optional
        CG preconditioner: 'eigen' (default) is the scalar-shift Kronecker-eigen preconditioner
        :func:`kron_eigen_preconditioner`. 'blockdiag' is the block-diagonal
        :func:`kron_blockdiag_preconditioner` -- exact when the noise is separable across the
        outer axis (whole inner-axis channels flagged the same at every time), cutting CG to a
        handful of iterations. It builds the per-channel noise ``Df = min`` over the batch/outer
        axis and *reverts to 'eigen' automatically* when no channel is fully flagged, so it never
        does more iterations than 'eigen'. Preconditioner choice does not change the solution
        (both solve the true ``A`` to ``tol``), only the iteration count.
    tol, max_iter, x0
        Forwarded to :func:`pcg`. The stop test uses the inverse-variance-weighted relative
        residual (``weight = 1/noise``), so high-variance flagged pixels drop out of it and
        ``tol`` is a data-independent accuracy (e.g. ~1e-6) rather than something tied to the
        flag-variance / good-pixel-noise ratio.
    n_threads : int, optional
        Threads used to parallelize the CG over the leading (batch / baseline) axis. The
        per-axis eigendecompositions and the preconditioner are built once and shared; only
        the independent right-hand sides are split across threads (PyTorch releases the GIL
        during the BLAS matvecs). When ``n_threads > 1`` the BLAS intra-op thread count is pinned
        to 1 for the duration (restored after) -- the per-axis matrices are small, so the
        baseline axis is the efficient place to parallelize and this stops PyTorch's own BLAS
        threading from oversubscribing the cores against the pool. 1 = serial (default), k = k
        threads, -1 = one per CPU. Capped at the batch size.
    return_alpha : bool, optional
        If True, return the solved coefficients ``alpha = (Ks + diag(noise))^-1 y`` instead of
        the mean ``Ks alpha`` (so the caller can apply a cross-covariance to predict at new
        points). Same shape as ``y``.

    Returns
    -------
    m : tensor
        Posterior mean, shape (..., Ntimes, Nfreqs) (or ``alpha`` if ``return_alpha``).
    info : dict
        CG diagnostics; for a parallel run, the worst (max) ``iters`` / ``resid`` over chunks.
    """
    # unravel the (Ntimes, Nfreqs) grids to length-(Ntimes*Nfreqs) vectors (time slow, freq fast)
    Ntimes, Nfreqs = Ct.shape[-1], Cf.shape[-1]
    noise_f = noise.reshape(*noise.shape[:-2], Ntimes * Nfreqs)
    y_f = y.reshape(*y.shape[:-2], Ntimes * Nfreqs)

    # shift = good-pixel noise variance, so the preconditioner M = Ct (x) Cf + sigma^2 I
    # matches A on the (bulk) unflagged pixels -> CG only has to resolve the flagged ones.
    # Built ONCE (one eigh per axis) and shared across all baselines / threads.
    if shift is None:
        shift = float(noise_f.real.amin())
    if precond == 'eigen':
        M = kron_eigen_preconditioner(Ct, Cf, shift=shift)
    elif precond == 'blockdiag':
        # per-channel (inner-axis) noise = min over the batch & outer (time) axis: the
        # time-independent part, large on fully-flagged channels. If no channel is fully flagged
        # (Df uniform) M IS the scalar-shift preconditioner, so fall back to the cheaper eigen
        # apply -- identical iterations, no per-iteration overhead.
        Df = noise.reshape(-1, Nfreqs).real.amin(dim=0)
        M = (kron_blockdiag_preconditioner(Ct, Cf, Df) if bool((Df > Df.amin()).any())
             else kron_eigen_preconditioner(Ct, Cf, shift=shift))
    else:
        raise ValueError(f"precond must be 'eigen' or 'blockdiag', got {precond!r}")

    def solve(yf, nf, xf):
        # alpha = (Ct (x) Cf + diag(noise))^-1 yf via preconditioned CG (A never densified),
        # then the Wiener mean m = (Ct (x) Cf) alpha (one more structured matvec). The stop
        # test is inverse-variance-weighted (weight=1/noise) so flagged pixels drop out of it
        # and `tol` is independent of the flag-var/noise ratio.
        a, inf = pcg(kron_matvec(Ct, Cf, diag=nf), yf, M=M, tol=tol, max_iter=max_iter, x0=xf,
                     weight=nf.reciprocal())
        # return alpha = (Ks+N)^-1 y (for prediction at new points) or the mean Ks alpha
        return (a if return_alpha else kron_matvec(Ct, Cf)(a)), inf

    # number of independent right-hand sides (e.g. baselines) on the leading axis
    B = y_f.shape[0] if y_f.dim() > 1 else 1
    nw = min(B, (os.cpu_count() or 1) if n_threads in (-1, None) else max(1, n_threads))
    if nw <= 1:
        m, info = solve(y_f, noise_f, x0)
    else:
        # split the batch into nw chunks; M is shared, the matvec's noise diagonal is sliced
        # per chunk (or shared when noise has a leading axis of 1). GIL is released in the BLAS.
        per_bl_noise = noise_f.shape[0] == B
        def work(c):
            return solve(y_f[c], noise_f[c] if per_bl_noise else noise_f,
                         x0[c] if x0 is not None else None)
        # the baseline pool IS the parallelism: pin BLAS to one thread so PyTorch's own
        # (small-matrix, per-axis) threading doesn't oversubscribe the cores against the pool.
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with ThreadPoolExecutor(nw) as ex:
                parts = list(ex.map(work, torch.arange(B).chunk(nw)))
        finally:
            torch.set_num_threads(old_threads)
        m = torch.cat([p[0] for p in parts], 0)
        info = {'iters': max(p[1]['iters'] for p in parts),
                'resid': max(p[1]['resid'] for p in parts)}
    return m.reshape(y.shape), info


# --------------------------------------------------------------------------------------
# batched Kronecker-Woodbury direct solve
# --------------------------------------------------------------------------------------
def kron_woodbury_predict(Ct, Cf, noise, y, rcond=1e-12, return_alpha=False):
    r"""
    Batched Wiener posterior mean ``m_b = Ks (Ks + diag(noise_b))^-1 y_b`` for a
    Kronecker signal covariance ``Ks = Ct (x) Cf``, without ever forming the
    ``(Ntimes*Nfreqs) x (Ntimes*Nfreqs)`` matrix.

    Solves via the Woodbury identity on the low-rank factor ``Ks = U U^H`` with
    ``U = U_t (x) U_f`` (the per-axis eigendecompositions truncated at ``rcond``): only
    a ``(k_t*k_f) x (k_t*k_f)`` capacitance is inverted, so this is a *direct*, high-
    dynamic-range-safe solve -- a Cholesky of that small capacitance, no CG / no iteration
    -- and a large flagged-pixel variance enters benignly as its reciprocal. It is the dual
    of the structured per-baseline solve of :func:`gprlim.models.posterior_mean_2d`
    (the joint 2D path): a win exactly when the kernels are low rank (the smooth-kernel
    regime); for full-rank kernels the capacitance is ``N x N`` and a dense / CG solve is
    preferable.

    Accuracy is set by the rank cutoff ``rcond``: a high-dynamic-range kernel must keep
    all modes down to ~``rcond * lambda_max``, so at high SNR set ``rcond`` small. See
    :func:`kron_wiener_cg` for a no-truncation, preconditioned-CG alternative.

    Parameters
    ----------
    Ct : tensor
        Time-axis kernel, Hermitian PSD of shape (Ntimes, Ntimes). May be complex (e.g. a
        ``CarrierKernel`` time axis -- an improper, asymmetric-PSD signal); ``noise``/``y``
        are promoted to the common dtype, and a complex ``Ct`` yields the strictly-linear
        (Hermitian) Wiener mean.
    Cf : tensor
        Frequency-axis kernel, Hermitian PSD of shape (Nfreqs, Nfreqs).
    noise, y : tensor
        Per-element noise variance and right-hand sides, shape ``(b, Ntimes, Nfreqs)`` (the
        leading axis batches over baselines); flagged pixels carry a large variance.
    rcond : float
        Relative eigenvalue cutoff for each kernel's low-rank factor.
    return_alpha : bool
        If True, return the solved coefficients ``alpha = (Ks + diag(noise))^-1 y`` instead of
        the mean ``Ks alpha`` -- so the caller can apply a different output operator (e.g. a
        cross-covariance to predict at new points). Same shape as ``y``.

    Returns
    -------
    tensor
        Posterior mean ``(b, Ntimes, Nfreqs)`` (or ``alpha`` if ``return_alpha``).
    """
    # common dtype so a complex kernel (CarrierKernel time axis -> improper / asymmetric-
    # PSD signal) and a real kernel (real freq axis) compose; a complex Hermitian Ct then
    # gives the strictly-linear (Hermitian) Wiener mean
    Ct = torch.as_tensor(Ct); Cf = torch.as_tensor(Cf)
    dtype = torch.promote_types(Ct.dtype, Cf.dtype)
    noise = torch.as_tensor(noise).to(dtype)
    y = torch.as_tensor(y).to(dtype)

    # low-rank factors: Ks = U U^H with U = U_t (x) U_f
    wt, Qt = torch.linalg.eigh(Ct)
    wf, Qf = torch.linalg.eigh(Cf)
    keep_t = wt > wt[-1] * rcond
    keep_f = wf > wf[-1] * rcond
    Ut = (Qt[:, keep_t] * wt[keep_t].clamp_min(0).sqrt()).to(dtype)   # (Ntimes, k_t)
    Uf = (Qf[:, keep_f] * wf[keep_f].clamp_min(0).sqrt()).to(dtype)   # (Nfreqs, k_f)
    rt, rf = Ut.shape[1], Uf.shape[1]

    d = 1.0 / noise                                                  # (b, Ntimes, Nfreqs)

    # capacitance  M_b = I + U^H diag(d_b) U   (b, k, k),  k = k_t*k_f
    # (the noise is the one thing that doesn't factor, so it enters here as a
    #  noise-weighted contraction over (t, f))
    W = torch.einsum('fj,btf,fl->btjl', Uf.conj(), d, Uf)            # (b, Ntimes, k_f, k_f)
    M = torch.einsum('ti,tk,btjl->bijkl', Ut.conj(), Ut, W).reshape(-1, rt * rf, rt * rf)
    M = M + torch.eye(rt * rf, dtype=M.dtype, device=M.device)
    L = torch.linalg.cholesky(M)

    # alpha_b = (Ks + D_b)^-1 y_b  via Woodbury:  D^-1 y - D^-1 U M^-1 (U^H D^-1 y)
    Dy = d * y
    r = torch.einsum('ti,btf,fj->bij', Ut.conj(), Dy, Uf.conj()).reshape(-1, rt * rf, 1)
    z = torch.cholesky_solve(r, L).reshape(-1, rt, rf)
    Uz = torch.einsum('ti,bij,fj->btf', Ut, z, Uf)
    alpha = d * (y - Uz)
    if return_alpha:                                                 # for prediction at new points
        return alpha

    # posterior mean  m_b = Ks alpha_b = U (U^H alpha_b)
    rr = torch.einsum('ti,btf,fj->bij', Ut.conj(), alpha, Uf.conj())
    m = torch.einsum('ti,bij,fj->btf', Ut, rr, Uf)
    return m


def kron_woodbury_inpaint(data, flags, Ct, Cf, noise, rcond=1e-12, huge=1e12):
    r"""
    Joint (time x frequency) inpaint via the batched Kronecker-Woodbury solver
    (:func:`kron_woodbury_predict`) -- a no-densify alternative to
    :func:`gprlim.models.inpaint_2d` when the kernels are low rank. Flagged
    pixels are given a large noise variance (``huge``) and replaced by the Wiener
    posterior mean; good pixels are returned untouched.

    Complex data is handled two ways depending on the kernels: with *real* kernels the
    real and imaginary parts are stacked along the batch axis and solved independently (a
    real covariance acts identically on each, which is cheaper); with a *complex*
    (Hermitian) kernel -- which couples them -- the complex system is solved directly.

    Parameters
    ----------
    data : tensor
        Visibilities, shape ``(Ntimes, Nfreqs)`` or ``(Nbls, Ntimes, Nfreqs)``, real or
        complex.
    flags : tensor
        Boolean flags broadcastable to ``data`` (True = flagged).
    Ct, Cf : tensor
        The (already-fit) time and frequency kernels, (Ntimes, Ntimes) / (Nfreqs, Nfreqs).
    noise : tensor
        Good-pixel noise variance broadcastable to ``data``; flagged pixels are
        overwritten with ``huge``.
    rcond : float
        Low-rank cutoff forwarded to :func:`kron_woodbury_predict`.
    huge : float
        Noise variance assigned to flagged pixels (effectively excludes them).

    Returns
    -------
    tensor
        Inpainted data, same shape as ``data`` (flagged pixels filled, good kept).
    """
    data = torch.as_tensor(data)
    flags = torch.as_tensor(flags).bool()
    twod = data.dim() == 2
    d3 = data[None] if twod else data
    f3 = flags[None] if twod else flags
    Nbls, Ntimes, Nfreqs = d3.shape

    noise = torch.as_tensor(noise)
    if noise.dim() == 2:
        noise = noise[None]
    n3 = noise.expand(Nbls, Ntimes, Nfreqs).clone()
    n3[f3] = huge

    kernel_complex = torch.as_tensor(Ct).is_complex() or torch.as_tensor(Cf).is_complex()
    if data.is_complex() and not kernel_complex:
        # real kernel -> split real/imag into the batch (cheaper), then recombine
        y = torch.cat([d3.real, d3.imag], 0)
        nzb = torch.cat([n3, n3], 0)
        m = kron_woodbury_predict(Ct, Cf, nzb, y, rcond=rcond)
        m = torch.complex(m[:Nbls], m[Nbls:])
    else:
        # real data, or a complex Hermitian kernel (couples real/imag) -> solve directly
        m = kron_woodbury_predict(Ct, Cf, n3, d3, rcond=rcond)

    out = torch.where(f3, m.to(d3.dtype), d3)
    return out[0] if twod else out


# --------------------------------------------------------------------------------------
# Kronecker Cholesky factor (used by KroneckerKernel for prior draws / whitening)
# --------------------------------------------------------------------------------------
def kron_cholesky(mats, jitter=1e-10):
    r"""
    Lower-triangular Cholesky factor of ``K = (x)_i mats[i]``, as a
    ``KroneckerProductTriangularLinearOperator`` via ``chol(A (x) B) = chol(A) (x)
    chol(B)`` -- the full ``(prod N_i)`` factor is never densified.

    Each (dense, Hermitian-PSD) per-axis kernel is factorized with ``torch.linalg.cholesky``
    (which handles a complex-Hermitian kernel, unlike ``linear_operator``'s
    ``psd_safe_cholesky`` whose jitter lookup rejects complex dtypes), after a small
    per-axis diagonal jitter for positive-definiteness. This is the **noiseless**
    prior-draw / whitening factor; ``K + noise`` is not Kronecker and cannot be factored
    this way.

    Parameters
    ----------
    mats : list of tensor
        Per-axis dense Hermitian-PSD kernels (e.g. ``KroneckerKernel._factor_mats()``).
    jitter : float
        Per-axis diagonal jitter as a fraction of each kernel's mean diagonal (0 disables).

    Returns
    -------
    KroneckerProductTriangularLinearOperator
    """
    chols = []
    for m in mats:
        if jitter:
            n = m.shape[-1]
            eps = jitter * m.diagonal(dim1=-2, dim2=-1).real.mean()
            m = m + eps * torch.eye(n, dtype=m.dtype, device=m.device)
        chols.append(TriangularLinearOperator(torch.linalg.cholesky(m), upper=False))
    return KroneckerProductTriangularLinearOperator(*chols)


# --------------------------------------------------------------------------------------
# batched dense (C + diag(N))^-1 solvers  (shared C, per-row noise diagonal)
# --------------------------------------------------------------------------------------
def _eigh_solve(A, rhs, rcond):
    """
    Truncated-eigendecomposition solve A^+ @ rhs for a batch of Hermitian A,
    robust to singular / indefinite A (the pinv-equivalent fallback). Modes
    with eigenvalue <= rcond * lambda_max are dropped.

    Parameters
    ----------
    A : tensor
        Hermitian matrices of shape (..., n, n)
    rhs : tensor
        Right-hand sides of shape (..., n, c)
    rcond : float
        Relative eigenvalue cutoff.

    Returns
    -------
    tensor of shape (..., n, c)
    """
    w, V = torch.linalg.eigh(A)
    winv = torch.where(w > rcond * w[..., -1:].clamp_min(0), 1.0 / w, torch.zeros_like(w))
    return V @ (winv.unsqueeze(-1) * (V.transpose(-1, -2) @ rhs))


def woodbury_batched(C, N, B=None, y=None, rcond=1e-15):
    """
    Batched Woodbury solve of (C + diag(N_b))^-1 over a batch of noise
    diagonals N sharing a single low-rank covariance C. Fast when the
    effective rank k = rank(C) (modes above rcond) is < n. Requires N > 0
    (it uses 1/N); for possibly singular / indefinite systems use
    `cholesky_batched`.

    Computes, per batch element, B @ (C + diag(N_b))^-1 @ y_b with B and/or y
    optional (see `gpr_invert`). The y-path never forms the (n, n) inverse.

    Parameters
    ----------
    C : tensor
        Shared signal covariance (n, n), PSD.
    N : tensor
        Per-batch noise variance diagonal (Nbatch, n), > 0.
    B : tensor, optional
        Left matrix (N*, n), e.g. cross-covariance K(x*, X).
    y : tensor, optional
        Observations (Nbatch, n).
    rcond : float
        Relative eigenvalue cutoff for the rank of C.

    Returns
    -------
    tensor
    """
    # low-rank factor of the signal covariance, keeping the modes above the
    # rcond cutoff: C ~= U U^H, with k = effective rank (U is complex if C is)
    evals, evecs = torch.linalg.eigh(C)
    keep = evals > evals[-1] * rcond
    U = evecs[:, keep] * evals[keep].clamp_min(0).sqrt()  # (n, k)

    # noise precision diagonal (Woodbury requires N > 0); Nc carries U's dtype so
    # the matmul-backed einsums don't mix real & complex (a no-op when C is real)
    Ninv = 1.0 / N  # (b, n)
    Nc = Ninv.to(U.dtype)

    # capacitance matrix M_b = I_k + U^H diag(1/N_b) U, and its Cholesky
    M = torch.einsum('nk,bn,nl->bkl', U.conj(), Nc, U)  # (b, k, k)
    M.diagonal(dim1=-2, dim2=-1).add_(1.0)
    L = torch.linalg.cholesky(M)

    # y-path: apply (C + diag(N_b))^-1 to y_b via the Woodbury identity,
    # without ever forming the (n, n) inverse
    if y is not None:

        # whiten y by the noise, then project into the low-rank subspace
        Dy = Ninv * y  # (b, n)
        rhs = torch.einsum('nk,bn->bk', U.conj(), Dy).unsqueeze(-1)

        # solve the small (k, k) capacitance system
        z = torch.cholesky_solve(rhs, L).squeeze(-1)  # (b, k)

        # reconstruct alpha_b = (C + diag(N_b))^-1 y_b
        alpha = Dy - Ninv * torch.einsum('nk,bk->bn', U, z)  # (b, n)

        # optionally pre-multiply by B (e.g. the cross-covariance K(x*, X))
        return alpha if B is None else torch.einsum('mn,bn->bm', B, alpha)

    # no-y path: build the full inverse per batch element via
    # (D + U U^H)^-1 = Dinv - Dinv U M^-1 U^H Dinv
    b = N.shape[0]
    MiUt = torch.cholesky_solve(U.conj().t().expand(b, -1, -1).contiguous(), L)  # (b, k, n) = M^-1 U^H
    W = torch.einsum('nk,bkm->bnm', U, MiUt)  # (b, n, n) = U M^-1 U^H
    inv = torch.diag_embed(Nc) - Ninv.unsqueeze(-1) * W * Ninv.unsqueeze(-2)

    # optionally pre-multiply by B
    return inv if B is None else torch.einsum('rp,bpq->brq', B, inv)


def cholesky_batched(C, N, B=None, y=None, rcond=1e-15):
    """
    Batched Cholesky solve of (C + diag(N_b))^-1 over a batch of noise
    diagonals N sharing a covariance C. General-purpose default: no low-rank
    assumption on C, and robust to a non-PSD (C + diag(N_b)) -- any batch
    element that is not positive-definite falls back to a truncated
    eigendecomposition (pinv-equivalent), so no jitter is needed.

    Same input / output contract as `woodbury_batched`.

    Parameters
    ----------
    C : tensor
        Shared signal covariance (n, n).
    N : tensor
        Per-batch noise variance diagonal (Nbatch, n).
    B : tensor, optional
        Left matrix (N*, n), e.g. cross-covariance K(x*, X).
    y : tensor, optional
        Observations (Nbatch, n).
    rcond : float
        Relative eigenvalue cutoff for the non-PD fallback.

    Returns
    -------
    tensor
    """
    n = C.shape[-1]

    # build the batched system A_b = C + diag(N_b) and attempt a Cholesky;
    # cholesky_ex flags (info > 0) any element that is not positive-definite
    A = C.unsqueeze(0) + torch.diag_embed(N)  # (b, n, n)
    L, info = torch.linalg.cholesky_ex(A)
    bad = info > 0
    good = ~bad

    # solve A_b X_b = rhs_b: Cholesky for the PD elements, with a truncated
    # eigendecomposition (pinv-equivalent) fallback for any non-PD ones
    def _solve(rhs):  # rhs: (b, n, c)
        sol = torch.empty_like(rhs)
        if good.any():
            sol[good] = torch.cholesky_solve(rhs[good], L[good])
        if bad.any():
            sol[bad] = _eigh_solve(A[bad], rhs[bad], rcond)
        return sol

    # y-path: solve against y_b directly, optionally pre-multiply by B
    if y is not None:
        alpha = _solve(y.unsqueeze(-1)).squeeze(-1)  # (b, n)
        return alpha if B is None else torch.einsum('mn,bn->bm', B, alpha)

    # no-y path: solve against the identity to get the full inverse per batch
    eye = torch.eye(n, dtype=C.dtype, device=C.device).expand(N.shape[0], n, n)
    inv = _solve(eye)  # (b, n, n)

    # optionally pre-multiply by B
    return inv if B is None else torch.einsum('rp,bpq->brq', B, inv)


def gpr_invert(C, N, B=None, y=None, rcond=1e-15, method='cholesky', chunk=None):
    """
    Perform (C + N)^-1 where C is low-rank
    and N is diagonal using Woodbury identity.
    Optionally, compute the matrix product
    B @ (C + N)^-1 @ y, if B and/or y is provided.

    Parameters
    ----------
    C : tensor
        Covariance matrix of signal (Nsamples, Nsamples)
    N : tensor
        Diagonal of noise variance ([Nbatch], Nsamples)
    B : tensor
        Matrix to dot into output (N*, Nsamples)
    y : tensor
        Observations of shape ([Nbatch], Nsamples)
    rcond : float
        Relative condition / eigenvalue cutoff.
    method : str
        Batched solver for the shared-C, batched-N case: 'cholesky'
        (default, robust, no low-rank assumption) or 'woodbury'
        (faster when rank(C) < Nsamples; requires N > 0).
    chunk : int, optional
        Process the batch in chunks of this size to bound memory.

    Returns
    -------
    tensor
    """
    if N.ndim == 1:
        # single system: pinv (robust to non-PSD)
        out = torch.linalg.pinv(C + N.diag(), hermitian=True, rcond=rcond)
        if y is not None:
            out = out @ y
        if B is not None:
            out = B @ out

    else:
        assert y is None or y.ndim > 1, "If N has batch dim then y must too"
        if C.ndim > 2:
            # C also has a batch dimension (not standard): per-batch pinv
            assert B is None or B.ndim > 2, "If C has batch dim then B must too"
            out = torch.stack([
                gpr_invert(C[i], N[i], B=None if B is None else B[i],
                           y=None if y is None else y[i], rcond=rcond)
                for i in range(len(N))
            ])

        else:
            # C shared, N batched: vectorized solve (the main driver)
            solver = {'woodbury': woodbury_batched, 'cholesky': cholesky_batched}[method]
            if chunk is None:
                out = solver(C, N, B=B, y=y, rcond=rcond)
            else:
                out = torch.cat([
                    solver(C, N[i:i + chunk], B=B,
                           y=None if y is None else y[i:i + chunk], rcond=rcond)
                    for i in range(0, len(N), chunk)
                ], dim=0)

    return out

