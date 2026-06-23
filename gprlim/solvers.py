"""
Linear solvers for structured Gaussian-process inference: a complex, preconditioned
conjugate-gradient solver and batched Kronecker(-Woodbury / -Cholesky) routines.

Throughout, ``Ct`` is the time-axis kernel (an ``(Ntimes, Ntimes)`` Hermitian covariance,
possibly complex) and ``Cf`` is the frequency-axis kernel (``(Nfreqs, Nfreqs)``), and the
joint signal covariance is the Kronecker product ``Ct (x) Cf``.

These all avoid densifying the unraveled ``(Ntimes*Nfreqs) x (Ntimes*Nfreqs)`` covariance:

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
"""
import torch
from linear_operator import to_linear_operator
from linear_operator.operators import (
    TriangularLinearOperator,
    KroneckerProductTriangularLinearOperator,
)


# --------------------------------------------------------------------------------------
# preconditioned conjugate gradients
# --------------------------------------------------------------------------------------
def _rdot(a, b):
    """Real part of ``<a, b> = a^H b`` over the last axis, keepdim -> (..., 1)."""
    return (a.conj() * b).sum(-1, keepdim=True).real


def _rnorm(a):
    """Euclidean norm over the last (system) axis, shape (..., 1)."""
    return _rdot(a, a).clamp_min(0).sqrt()


def pcg(A, b, M=None, tol=1e-6, max_iter=1000, x0=None):
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

    # r = residual b - A x ; z = preconditioned residual M^-1 r ; p = search direction.
    # rz = <r, z> is the quantity PCG drives to zero (real for Hermitian PD A and M).
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    bnorm = _rnorm(b).clamp_min(tiny)                      # denominator of the relative residual
    z = apply_M(r)
    p = z.clone()
    rz = _rdot(r, z)

    k, resid = 0, float((_rnorm(r) / bnorm).max())
    if resid <= tol:                                       # x0 already within tolerance
        return x, {'iters': 0, 'resid': resid}
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        # alpha: exact line-minimization step along p (denominator p^H A p is real & > 0)
        alpha = rz / _rdot(p, Ap).clamp_min(tiny)
        x = x + alpha * p                                  # advance the solution
        r = r - alpha * Ap                                 # ... and the residual
        resid = float((_rnorm(r) / bnorm).max())           # worst row across the batch
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
    (e.g. ``diag = A.diagonal()``): cheap, and tames a FLAG_VAR-heavy noise diagonal."""
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


def kron_wiener_cg(Ct, Cf, noise, y, shift=None, tol=1e-6, max_iter=1000, x0=None):
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
        large value.
    y : tensor
        Data, shape (..., Ntimes, Nfreqs).
    shift : float, optional
        Preconditioner diagonal shift; default the smallest noise value (the good-pixel
        variance), which makes the preconditioner capture the bulk of ``A``.
    tol, max_iter, x0
        Forwarded to :func:`pcg`.

    Returns
    -------
    m : tensor
        Posterior mean, shape (..., Ntimes, Nfreqs).
    info : dict
        CG diagnostics from :func:`pcg`.
    """
    # unravel the (Ntimes, Nfreqs) grids to length-(Ntimes*Nfreqs) vectors (time slow, freq fast)
    Ntimes, Nfreqs = Ct.shape[-1], Cf.shape[-1]
    noise_f = noise.reshape(*noise.shape[:-2], Ntimes * Nfreqs)
    y_f = y.reshape(*y.shape[:-2], Ntimes * Nfreqs)

    # shift = good-pixel noise variance, so the preconditioner M = Ct (x) Cf + sigma^2 I
    # matches A on the (bulk) unflagged pixels -> CG only has to resolve the flagged ones
    if shift is None:
        shift = float(noise_f.real.amin())

    # solve alpha = (Ct (x) Cf + diag(noise))^-1 y by preconditioned CG, with A applied as a
    # structured matvec (the full covariance is never densified) ...
    A = kron_matvec(Ct, Cf, diag=noise_f)
    M = kron_eigen_preconditioner(Ct, Cf, shift=shift)
    alpha, info = pcg(A, y_f, M=M, tol=tol, max_iter=max_iter, x0=x0)

    # ... then the Wiener posterior mean is m = (Ct (x) Cf) alpha (one more Kron matvec)
    m = kron_matvec(Ct, Cf)(alpha)
    return m.reshape(y.shape), info


# --------------------------------------------------------------------------------------
# batched Kronecker-Woodbury direct solve
# --------------------------------------------------------------------------------------
def kron_woodbury_predict(Ct, Cf, noise, y, rcond=1e-12):
    r"""
    Batched Wiener posterior mean ``m_b = Ks (Ks + diag(noise_b))^-1 y_b`` for a
    Kronecker signal covariance ``Ks = Ct (x) Cf``, without ever forming the
    ``(Ntimes*Nfreqs) x (Ntimes*Nfreqs)`` matrix.

    Solves via the Woodbury identity on the low-rank factor ``Ks = U U^H`` with
    ``U = U_t (x) U_f`` (the per-axis eigendecompositions truncated at ``rcond``): only
    a ``(k_t*k_f) x (k_t*k_f)`` capacitance is inverted, so this is a *direct*, high-
    dynamic-range-safe solve -- a Cholesky of that small capacitance, no CG / no iteration
    -- and a flagged-pixel ``FLAG_VAR`` enters benignly as ``1/FLAG_VAR``. It is the dual
    of the structured per-baseline solve of :func:`gprlim.workflows.inpaint`
    (``mode='joint'``): a win exactly when the kernels are low rank (the smooth-kernel
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

    Returns
    -------
    tensor
        Posterior mean, shape ``(b, Ntimes, Nfreqs)``.
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

    # posterior mean  m_b = Ks alpha_b = U (U^H alpha_b)
    rr = torch.einsum('ti,btf,fj->bij', Ut.conj(), alpha, Uf.conj())
    m = torch.einsum('ti,bij,fj->btf', Ut, rr, Uf)
    return m


def kron_woodbury_inpaint(data, flags, Ct, Cf, noise, rcond=1e-12, huge=1e12):
    r"""
    Joint (time x frequency) inpaint via the batched Kronecker-Woodbury solver
    (:func:`kron_woodbury_predict`) -- a no-densify alternative to
    ``gprlim.workflows.inpaint(mode='joint')`` when the kernels are low rank. Flagged
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
