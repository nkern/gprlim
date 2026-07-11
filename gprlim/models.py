import math

import torch

from .solvers import (stack_ri, unstack_ri, promote_like, shrink, kron_cholesky,
                      gpr_invert, cholesky_batched,
                      kron_woodbury_predict, kron_wiener_cg)


def mean_center(y, noise, dim=-1):
    """
    Inverse-noise-weighted mean of ``y`` along dim (a per-batch constant offset),
    for optional centering before a GP solve. For polynomial or other trends, use a real
    mean function instead.

    Parameters
    ----------
    y : tensor
        Data of shape (Nbatch, ..., Nx), real or complex.
    noise : tensor
        Noise variance, same shape as ``y`` (the inverse weights).
    dim : int or tuple of int
        Axis (or axes) of ``y`` to average over.

    Returns
    -------
    tensor
        The weighted mean of shape (Nbatch, ..., 1); subtract it from ``y`` to center.
    """
    w = noise.pow(-1)
    return (y * w).sum(dim, keepdim=True) / w.sum(dim, keepdim=True)


def posterior_mean_1d(kernel, x, y, noise, pred_x=None, mu=None, dim=-1, detrend=False,
                      rcond=1e-15, method='cholesky', chunk=None, pred_kernel=None,
                      cg_tol=1e-6, cg_max_iter=1000):
    """
    GP posterior mean (Wiener filter) ``Cp (Cs + diag(noise))^-1 (y - mu) + mu`` along one
    axis of ``y``, batched over all the others.

    The GP/sample axis of ``y`` is ``dim`` (length ``len(x)``); every other axis is flattened
    into a batch and solved together (each row shares ``kernel`` over ``x``). If the data are
    complex and the covariance is real, the real and imaginary parts are stacked and solved
    as two real systems (cheaper, identical result); a complex covariance is solved directly.

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over the 1-D grid.
    x : tensor
        Training grid, shape (Nx,).
    y : tensor
        Data, shape (..., Nx, ...), with the sample axis at ``dim``.
    noise : tensor
        Per-element noise variance, must broadcast with ``y``.
    pred_x : tensor, optional
        Prediction grid (Npred,); the cross-covariance K(pred_x, x) is used. Default predicts
        at the training points (Npred = Nx).
    pred_kernel : gpytorch.kernels.Kernel, optional
        Separate prediction covariance ``Cp`` for the output operator in the Wiener filter
        ``Cp (Cs + diag(noise))^-1 y`` -- e.g. to reconstruct a single signal component (``Cs``
        is still the full signal-plus-other covariance used in the solve). Default None reuses
        ``kernel`` as ``Cp``. Evaluated at ``pred_x`` (as ``pred_kernel(pred_x, x)``) if given.
    mu : callable, optional
        Mean function ``mu(x[:, None]) -> (..., Nx)``; default zero mean.
    dim : int, optional
        The sample axis of ``y`` (must not be 0, the batch axis). Default -1.
    detrend : bool, optional
        If True, do another global mean-subtraction after subtracting mu from data.
    rcond, method, chunk
        Forwarded to the batched solver (see :func:`gprlim.solvers.gpr_invert`). ``method='cg'``
        is a preconditioned-CG solve (:func:`gprlim.solvers.cg_batched`) that is cheapest when the
        covariance is high-rank (e.g. a ``DeltaKernel``) and the flags are near-shared across the
        batch (fully-flagged / vertical channels, as in the ``2d_1d`` final frequency stage); it
        matches 'woodbury' to ``cg_tol`` and reports ``cg_iters``.
    cg_tol, cg_max_iter : float, int
        CG tolerance / iteration cap for ``method='cg'`` (see :func:`gprlim.solvers.cg_batched`).

    Returns
    -------
    tensor
        Posterior mean: ``y`` with its ``dim`` axis replaced by length Npred.
    info : dict
        Solver diagnostics; empty except for ``method='cg'``, which reports ``cg_iters`` /
        ``resid`` (as :func:`posterior_mean_2d` does).
    """
    if dim == 0:
        raise ValueError("dim must not be 0 (the batch axis); use a non-zero / negative dim.")
    info = {}
    x = torch.as_tensor(x).reshape(-1)
    Nx = x.shape[-1]
    # move the sample axis to last and flatten the remaining axes into one batch; noise is
    # broadcast to y's shape first (so a shared (1, ...) / lower-rank noise lines up per row)
    noise = torch.broadcast_to(noise, y.shape)
    yp = y.movedim(dim, -1)
    lead = yp.shape[:-1]
    yf = yp.reshape(-1, Nx)
    nf = noise.movedim(dim, -1).reshape(-1, Nx)

    with torch.no_grad():
        # get dense kernel
        gx = x if pred_x is None else torch.as_tensor(pred_x).reshape(-1)
        pk = kernel if pred_kernel is None else pred_kernel       # Cp kernel (defaults to Cs's)
        Cs = kernel(x[:, None]).to_dense()
        Cp = Cs if (pred_x is None and pred_kernel is None) else pk(gx[:, None], x[:, None]).to_dense()

        # get mean
        mu_x = 0.0 if mu is None else mu(x[:, None])
        mu_pred = mu_x if pred_x is None else (0.0 if mu is None else mu(gx[:, None]))
        yc = yf - mu_x

        trend = 0.0
        if detrend:
            trend = mean_center(yc, nf, dim=-1)          # nf/-1: yc is moved to (..., Nx)
            yc = yc - trend

        # real covariance + complex data -> stack real/imag, two real solves, recombine;
        # otherwise solve directly (promoting a real cov to complex for complex data).
        if yc.is_complex() and not Cs.is_complex():
            yc, nf = stack_ri(yc), torch.cat([nf, nf], 0)
            pred = unstack_ri(gpr_invert(Cs, nf, B=Cp, y=yc, rcond=rcond, method=method, chunk=chunk,
                                         cg_tol=cg_tol, cg_max_iter=cg_max_iter, info=info))
        else:
            Cs, Cp = promote_like(Cs, yc), promote_like(Cp, yc)
            pred = gpr_invert(Cs, nf, B=Cp, y=yc, rcond=rcond, method=method, chunk=chunk,
                              cg_tol=cg_tol, cg_max_iter=cg_max_iter, info=info)
        pred = pred + mu_pred + trend

    return pred.reshape(*lead, pred.shape[-1]).movedim(-1, dim), info


def posterior_mean_2d(kernel1, kernel2, x1, x2, y, noise, pred_x1=None, pred_x2=None,
                      C1_rcond=None, method='woodbury', mu=None, dims=(-2, -1), detrend=False,
                      rcond=1e-12, cg_tol=1e-4, cg_max_iter=5000, n_threads=1,
                      pred_kernel1=None, pred_kernel2=None, precond='separable', sparse_rcond=1e-12):
    """
    GP posterior mean for a separable 2D covariance ``K = C1 (x) C2`` (``C1 = kernel1`` over
    ``x1``, the slow/outer axis; ``C2 = kernel2`` over ``x2``), batched over the leading axis
    of ``y``.

    ``y`` has shape (Nbatch, Nx1, Nx2); returns the full posterior mean on the (x1, x2) grid,
    same shape. The ``(Nx1*Nx2)^2`` covariance is never formed ('woodbury'/'cg' work on the
    dense factors; 'cholesky' densifies only ``C1 (x) C2``). A real covariance with complex
    data is split into real/imag (two real solves); a complex covariance is solved directly.

    Parameters
    ----------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` (outer) and ``x2`` (inner). ``kernel1`` may be complex.
    x1, x2 : tensor
        1D grids, shapes (Nx1,), (Nx2,).
    y : tensor
        Data, shape (Nbatch, Nx1, Nx2).
    noise : tensor
        Per-pixel noise variance, must broadcast with ``y``; flags should be a large value.
    pred_x1, pred_x2 : tensor, optional
        Prediction grids (Npred1,), (Npred2,); the cross-covariances ``C1(pred_x1, x1)`` and
        ``C2(pred_x2, x2)`` are applied to the solved coefficients to predict at new points
        (either axis independently; default predicts on the (x1, x2) grid). Not supported
        together with ``C1_rcond`` (the shrinkage applies to the training block only).
    pred_kernel1, pred_kernel2 : gpytorch.kernels.Kernel, optional
        Separate prediction factor kernels for the output operator ``Kp = Cp1 (x) Cp2`` in the
        Wiener filter ``Kp (Ks + diag(noise))^-1 y`` -- e.g. to reconstruct a single signal
        component (``Ks = C1 (x) C2`` is still the full covariance used in the solve). Default
        None reuses ``kernel1`` / ``kernel2``. Compatible with ``C1_rcond`` (which shrinks only
        the signal block ``Ks``); evaluated at ``pred_x1`` / ``pred_x2`` if those are given.
    C1_rcond : float, optional
        Relative eigenvalue *floor* in (0, 1] that shrinks the outer factor ``C1``'s spectrum
        toward flat (lift modes to ``C1_rcond * lambda_max``, trace-preserved; see
        :func:`gprlim.solvers.shrink`). Interpolates from the full 2D joint model (None / 0)
        toward the per-``x1``-independent 1D solve: ``C1_rcond = 1`` flattens ``C1`` to a scaled
        identity, so ``C1 (x) C2 = I (x) C2`` -- the 1D inpaint along ``x2``. Tune on a log scale.
        Default None (full joint).
    method : str
        'woodbury' (default; low-rank, rank-truncated at ``rcond``), 'cg' (preconditioned CG,
        to ``cg_tol``, parallelized over the batch by ``n_threads``), or 'cholesky' (densify
        ``C1 (x) C2`` -- small grids only).
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)``; default zero mean.
    dims : tuple, optional
        Dimensions to perform 2D posterior mean. Maps dims to (-2, -1) of working tensor.
    detrend : bool, optional
        If True, do another global mean-subtraction after subtracting mu from data.
    rcond, cg_tol, cg_max_iter, n_threads
        Forwarded to the chosen structured solver.
    precond : str, optional
        CG preconditioner (``method='cg'`` only): 'scalar' (scalar-shift) and 'separable'
        (default; separable-noise, exact for full-channel AND full-time flags -- far fewer CG iterations
        when whole frequency channels or whole time integrations are flagged, cascading to per-channel
        then to 'scalar' as those flags are absent) are pure preconditioners that do NOT change
        the result. 'sparse_separable' additionally applies the
        signal *operator* in low-rank form (both the preconditioner and the matvec truncated at
        ``sparse_rcond``) -- a much cheaper per-iteration CG for low-rank kernels, but a low-rank
        (approximate) solve like ``method='woodbury'`` rather than an exact one (accuracy set by
        ``sparse_rcond``, exact as it ``-> 0``). See
        :func:`gprlim.solvers.kron_heteroscedastic_preconditioner` /
        :func:`gprlim.solvers.kron_sparse_heteroscedastic_preconditioner` /
        :func:`gprlim.solvers.kron_lowrank_matvec`.
    sparse_rcond : float, optional
        Relative eigenvalue cutoff for ``precond='sparse_separable'`` (ignored otherwise): the
        low-rank operator and preconditioner keep modes above ``sparse_rcond * lambda_max`` per axis.
        Smaller keeps more modes (more accurate, stronger preconditioner, fewer iterations, costlier
        apply; ``-> 0`` recovers the exact 'separable' solve); larger truncates more (cheaper apply,
        less accurate, possibly more iterations). The default tracks 'separable' in both answer and
        iteration count; loosen it only while the result and ``info['cg_iters']`` stay acceptable.
        Default 1e-12.

    Returns
    -------
    tensor
        Posterior mean, shape (Nbatch, Nx1, Nx2) (or (Nbatch, Npred1, Npred2) for pred grids).
    info : dict
        Solver diagnostics. For ``method='cg'`` holds ``cg_iters`` -- the number of CG
        iterations (the worst over batch chunks for a parallel ``n_threads`` run). Empty
        for the 'woodbury' / 'cholesky' methods.
    """
    new_points = pred_x1 is not None or pred_x2 is not None
    if new_points and C1_rcond:
        raise NotImplementedError("pred_x1/pred_x2 with C1_rcond shrinkage is not supported "
                                  "(the shrinkage applies only to the training block).")
    x1 = torch.as_tensor(x1).reshape(-1)
    x2 = torch.as_tensor(x2).reshape(-1)
    Nx1, Nx2 = x1.shape[-1], x2.shape[-1]
    gx1 = x1 if pred_x1 is None else torch.as_tensor(pred_x1).reshape(-1)
    gx2 = x2 if pred_x2 is None else torch.as_tensor(pred_x2).reshape(-1)

    # move the two sample axes to the last two and flatten the rest into one batch; noise is
    # broadcast to y's shape first (so a shared (1, ...) / lower-rank noise lines up per row)
    noise = torch.broadcast_to(noise, y.shape)
    yp = y.movedim(dims, (-2, -1))
    lead = yp.shape[:-2]
    yf = yp.reshape(-1, Nx1, Nx2)
    nf = noise.movedim(dims, (-2, -1)).reshape(-1, Nx1, Nx2)

    with torch.no_grad():
        # mean function: subtract at the training grid, add back at the prediction grid; then
        # optional inverse-noise-weighted detrend (a per-row offset, restored after the solve).
        mu_x = 0.0 if mu is None else mu(x1[:, None], x2[:, None])
        mu_pred = mu_x if not new_points else (0.0 if mu is None else mu(gx1[:, None], gx2[:, None]))
        yc = yf - mu_x
        trend = 0.0
        if detrend:
            trend = mean_center(yc, nf, dim=(-2, -1))
            yc = yc - trend

        # dense per-axis training factors (the (Nx1*Nx2)^2 covariance is never formed); spectrum-
        # shrink C1 toward flat (no-op if C1_rcond is None) -- interpolates 2D->1D. Bp1/Bp2 are
        # the output operators: training covariances (predict on the grid) or cross-covariances.
        C1 = shrink(kernel1(x1[:, None]).to_dense().detach(), C1_rcond)
        C2 = kernel2(x2[:, None]).to_dense().detach()
        pk1 = kernel1 if pred_kernel1 is None else pred_kernel1   # Cp factors (default to Cs's)
        pk2 = kernel2 if pred_kernel2 is None else pred_kernel2
        Bp1 = C1 if (pred_x1 is None and pred_kernel1 is None) else pk1(gx1[:, None], x1[:, None]).to_dense().detach()
        Bp2 = C2 if (pred_x2 is None and pred_kernel2 is None) else pk2(gx2[:, None], x2[:, None]).to_dense().detach()

        # complex strategy: split real/imag for a real covariance, else solve directly
        # (promoting a real cov to complex for complex data).
        cov_dtype = torch.promote_types(C1.dtype, C2.dtype)
        split = yc.is_complex() and not cov_dtype.is_complex
        if yc.is_complex() and not split:
            cov_dtype = torch.promote_types(cov_dtype, yc.dtype)
        C1, C2, Bp1, Bp2 = (t.to(cov_dtype) for t in (C1, C2, Bp1, Bp2))
        if split:
            ys, nz = torch.cat([yc.real, yc.imag], 0), torch.cat([nf, nf], 0)
        else:
            ys, nz = yc, nf

        # solve for alpha = (Ks + diag(noise))^-1 ys  (Ks = C1 (x) C2), structured per method
        info = {}
        if method == 'woodbury':
            alpha = kron_woodbury_predict(C1, C2, nz, ys, rcond=rcond, return_alpha=True)
        elif method == 'cg':
            alpha, cg_info = kron_wiener_cg(C1, C2, nz, ys, tol=cg_tol, max_iter=cg_max_iter,
                                            n_threads=n_threads, return_alpha=True, precond=precond,
                                            sparse_rcond=sparse_rcond)
            info['cg_iters'] = cg_info['iters']
        elif method == 'cholesky':
            nb = ys.shape[0]
            alpha = gpr_invert(torch.kron(C1, C2), nz.reshape(nb, -1), y=ys.reshape(nb, -1),
                               method='cholesky').reshape(nb, Nx1, Nx2)
        else:
            raise ValueError(f"method must be 'woodbury', 'cg' or 'cholesky', got {method!r}")

        # apply the output operator  m = (Bp1 (x) Bp2) alpha  (rectangular for new points),
        # recombine real/imag, then restore the mean function + detrend offset
        mm = torch.einsum('pt,rtf,qf->rpq', Bp1, alpha, Bp2)
        pred = (unstack_ri(mm) if split else mm) + mu_pred + trend

    return pred.reshape(*lead, *pred.shape[-2:]).movedim((-2, -1), dims), info


def inpaint_1d(kernel, x, y, noise, flags, mu=None, dim=-1, detrend=False,
               rcond=1e-15, method='cholesky', chunk=None, pred_kernel=None):
    """
    Inpaint ``y`` at flagged pixels with the 1D GP posterior mean.

    Computes the posterior mean (:func:`posterior_mean_1d`) over the whole grid and writes it
    into the flagged pixels, leaving good pixels untouched. ``flags`` only marks where to
    insert the model -- the down-weighting of flagged pixels must already be encoded in
    ``noise`` (a large variance there), so the parametric kernel fills even fully-flagged
    rows/columns (which a data-driven covariance could not).

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over the 1-D grid.
    x : tensor
        Training grid, shape (Nx,).
    y : tensor
        Data, shape (Nbatch, ...), with the sample axis at ``dim`` (real or complex).
    noise : tensor
        Per-element noise variance, must broadcast with ``y``; flagged pixels assumed already
        down-weighted (large variance).
    flags : tensor
        Boolean mask, same shape as ``y`` (True where flagged); marks where the model is
        written back into the data.
    mu : callable, optional
        Mean function ``mu(x[:, None]) -> (..., Nx)``; forwarded to :func:`posterior_mean_1d`.
    dim : int, optional
        The sample axis of ``y`` (must not be 0, the batch axis). Default -1.
    detrend : bool, optional
        Subtract a per-row inverse-noise-weighted mean before the solve and add it back (see
        :func:`posterior_mean_1d`).
    pred_kernel : gpytorch.kernels.Kernel, optional
        Separate prediction covariance ``Cp`` for the model written into the gaps -- e.g. to
        inpaint a single signal component; default None reuses ``kernel``. See
        :func:`posterior_mean_1d`.
    rcond, method, chunk
        Forwarded to :func:`posterior_mean_1d` (the batched solver).

    Returns
    -------
    inp : tensor
        ``y`` with flagged pixels replaced by the posterior mean; good pixels untouched.
    mdl : tensor
        The full posterior-mean model over the grid (same shape/dtype as ``y``).
    """
    mdl, _ = posterior_mean_1d(kernel, x, y, noise, mu=mu, dim=dim, detrend=detrend,
                               rcond=rcond, method=method, chunk=chunk, pred_kernel=pred_kernel)
    return torch.where(flags, mdl, y), mdl


def inpaint_2d(kernel1, kernel2, x1, x2, y, noise, flags, C1_rcond=None, mu=None, detrend=False,
               dims=(-2, -1), method='woodbury', rcond=1e-12, cg_tol=1e-4, cg_max_iter=5000,
               n_threads=1, pred_kernel1=None, pred_kernel2=None, precond='separable', sparse_rcond=1e-12):
    """
    Inpaint ``y`` at flagged pixels with the 2D GP posterior mean (separable ``C1 (x) C2``).

    Computes the joint posterior mean (:func:`posterior_mean_2d`) on the (x1, x2) grid and
    writes it into the flagged pixels, leaving good pixels untouched. ``flags`` only marks
    where to insert -- the flagged-pixel down-weighting must already be in ``noise`` (a large
    variance there). The ``(Nx1*Nx2)^2`` covariance is never formed.

    Parameters
    ----------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` (outer/slow axis) and ``x2`` (inner/fast). ``kernel1`` may
        be complex.
    x1, x2 : tensor
        1D grids, shapes (Nx1,), (Nx2,).
    y : tensor
        Data, shape (..., Nx1, Nx2) with the two GP axes at ``dims`` (real or complex).
    noise : tensor
        Per-pixel noise variance, must broadcast with ``y``; flagged pixels assumed already
        down-weighted (large variance).
    flags : tensor
        Boolean mask, broadcastable to ``y`` (True where flagged).
    C1_rcond : float, optional
        Relative eigenvalue floor in (0, 1] that shrinks the outer factor ``C1``'s spectrum toward
        flat, interpolating the 2D joint solve toward the per-``x1``-independent 1D inpaint along
        ``x2`` (``C1_rcond = 1`` recovers the 1D solve; see
        :func:`gprlim.solvers.shrink` / :func:`posterior_mean_2d`).
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)``.
    detrend : bool, optional
        Subtract an inverse-noise-weighted mean over the (x1, x2) axes before the solve and
        add it back (see :func:`posterior_mean_2d`).
    pred_kernel1, pred_kernel2 : gpytorch.kernels.Kernel, optional
        Separate prediction factor kernels for the model written into the gaps -- e.g. to inpaint
        a single signal component; default None reuses ``kernel1`` / ``kernel2``. See
        :func:`posterior_mean_2d`.
    dims, method, rcond, cg_tol, cg_max_iter, n_threads, precond, sparse_rcond
        Forwarded to :func:`posterior_mean_2d` (``dims`` = the two GP axes of ``y``). The default
        ``precond='separable'`` gives a large CG speedup when whole frequency channels are
        flagged; use ``precond='sparse_separable'`` (tuned by ``sparse_rcond``) for a cheaper
        per-iteration apply when the kernels are low rank, or ``precond='scalar'`` for the
        plain scalar-shift preconditioner.

    Returns
    -------
    inp : tensor
        ``y`` with flagged pixels replaced by the posterior mean; good pixels untouched.
    mdl : tensor
        The full posterior-mean model on the (x1, x2) grid (same shape/dtype as ``y``).
    """
    mdl, _ = posterior_mean_2d(kernel1, kernel2, x1, x2, y, noise, C1_rcond=C1_rcond, mu=mu,
                               detrend=detrend, dims=dims, method=method, rcond=rcond, cg_tol=cg_tol,
                               cg_max_iter=cg_max_iter, n_threads=n_threads,
                               pred_kernel1=pred_kernel1, pred_kernel2=pred_kernel2, precond=precond,
                               sparse_rcond=sparse_rcond)
    return torch.where(flags, mdl, y), mdl


def _chol_jit(K, jitter=1e-10):
    """Lower Cholesky of a Hermitian-PSD ``K`` with a small relative diagonal jitter for
    positive-definiteness (``torch.linalg.cholesky`` handles a complex-Hermitian ``K``)."""
    d = K.diagonal(dim1=-2, dim2=-1).real.mean()
    eye = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    return torch.linalg.cholesky(K + (jitter * d) * eye)


def _draws_from_chol(L, batch, generator=None):
    """``batch + (n,)`` draws ``f = L z``: ``z`` real standard normal, or (if ``L`` is for a
    complex-Hermitian covariance) a *circular* complex standard normal ``z = (a + i b)/sqrt(2)``
    with ``a, b ~ N(0, 1)`` so ``E[z z^H] = I`` and hence ``E[f f^H] = L L^H``. ``L`` may be a
    dense tensor or a ``LinearOperator``."""
    n = L.shape[-1]
    rdtype = torch.empty((), dtype=L.dtype).real.dtype     # real-component dtype (real or complex L)
    a = torch.randn(*batch, n, 1, dtype=rdtype, device=L.device, generator=generator)
    if L.dtype.is_complex:
        b = torch.randn(*batch, n, 1, dtype=rdtype, device=L.device, generator=generator)
        z = torch.complex(a, b) / 2.0 ** 0.5
    else:
        z = a
    return (L @ z).squeeze(-1)


def prior_draws_1d(kernel, x, mu=None, size=1, jitter=1e-10, generator=None):
    """
    Draw samples from the 1D GP prior with covariance ``kernel(x)``.

    Each draw is ``f = mu + L z`` with ``L = chol(kernel(x))`` and ``z`` standard normal. A
    real kernel gives real draws; a complex-Hermitian kernel (e.g. a CarrierKernel) gives
    circular complex draws (``z`` a circular standard normal), so ``E[f f^H] = K``.

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over the 1-D grid.
    x : tensor
        Grid to draw on, shape (Nx,).
    mu : callable, optional
        Mean function ``mu(x[:, None]) -> (..., Nx)`` added to each draw; default zero mean.
    size : int, optional
        Number of draws. Default 1.
    jitter : float, optional
        Relative diagonal jitter added to ``K`` for a positive-definite Cholesky.
    generator : torch.Generator, optional
        RNG for reproducible draws.

    Returns
    -------
    tensor
        Draws of shape (size, Nx) (complex if ``kernel`` is complex-Hermitian).
    """
    x = torch.as_tensor(x).reshape(-1)
    with torch.no_grad():
        K = kernel(x[:, None]).to_dense()
        f = _draws_from_chol(_chol_jit(K, jitter), (size,), generator)
        return f if mu is None else f + mu(x[:, None])


def prior_draws_2d(kernel1, kernel2, x1, x2, mu=None, size=1, jitter=1e-10, generator=None):
    """
    Draw samples from the separable 2D GP prior ``K = C1 (x) C2``.

    Each draw is ``f = mu + L z`` with ``L`` the structured Kronecker Cholesky
    (:meth:`gprlim.kernels.KroneckerKernel.cholesky`), so the full ``(Nx1*Nx2)`` factor is
    never densified. Circular complex draws if ``C1`` (or ``C2``) is complex-Hermitian, so
    ``E[f f^H] = C1 (x) C2``.

    Parameters
    ----------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` (outer) and ``x2`` (inner). Either may be complex.
    x1, x2 : tensor
        1D grids, shapes (Nx1,), (Nx2,).
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)`` added to each
        draw; default zero mean.
    size : int, optional
        Number of draws. Default 1.
    jitter : float, optional
        Relative diagonal jitter on each factor for a positive-definite Cholesky.
    generator : torch.Generator, optional
        RNG for reproducible draws.

    Returns
    -------
    tensor
        Draws of shape (size, Nx1, Nx2) (complex if either kernel is complex-Hermitian).
    """
    from .kernels import KroneckerKernel
    x1 = torch.as_tensor(x1).reshape(-1)
    x2 = torch.as_tensor(x2).reshape(-1)
    with torch.no_grad():
        L = KroneckerKernel(kernel1, kernel2, x1, x2).cholesky(jitter=jitter)
        f = _draws_from_chol(L, (size,), generator).reshape(size, x1.shape[-1], x2.shape[-1])
        return f if mu is None else f + mu(x1[:, None], x2[:, None])


def _conv_normal(shape, data_complex, cov_complex, dtype, device, generator=None):
    """Standard normal matching the Wiener-solve complex convention: real (real data);
    circular complex ``z = (a + i b)/sqrt(2)`` with ``E[|z|^2] = 1`` (complex data + complex
    covariance, real/imag coupled); or 'biparte' complex ``z = a + i b`` with real & imag each
    ``N(0,1)`` (complex data + real covariance, where the solve treats each part identically)."""
    a = torch.randn(*shape, dtype=dtype, device=device, generator=generator)
    if not data_complex:
        return a
    z = torch.complex(a, torch.randn(*shape, dtype=dtype, device=device, generator=generator))
    return z / 2.0 ** 0.5 if cov_complex else z


def _chol_matmul(L, z):
    """Apply a Cholesky factor: ``L @ z`` with ``z`` shaped (..., n) -> (..., n). For the
    real-covariance, complex-data ('biparte') case the real factor is applied to the real and
    imaginary parts separately, avoiding a real-operator @ complex-rhs dtype clash (works for a
    dense tensor or a Kronecker ``LinearOperator``)."""
    if z.is_complex() and not L.dtype.is_complex:
        re = (L @ z.real.unsqueeze(-1)).squeeze(-1)
        im = (L @ z.imag.unsqueeze(-1)).squeeze(-1)
        return torch.complex(re, im)
    return (L @ z.unsqueeze(-1)).squeeze(-1)


def posterior_draws_1d(kernel, x, y, noise, mu=None, size=1, dim=-1, jitter=1e-10,
                       rcond=1e-15, method='cholesky', chunk=None, generator=None):
    """
    Draw samples from the 1D GP posterior given data ``y`` via Matheron's rule.

    Pathwise (Matheron) conditioning, which is *exact* -- the samples have the true posterior
    mean and covariance:

        f_post = f_prior + posterior_mean(y - f_prior - eps),   eps ~ N(0, noise)

    with a fresh prior draw ``f_prior`` and an independent noise draw ``eps`` per sample; the
    correction reuses :func:`posterior_mean_1d`, so no posterior covariance is ever formed.
    Complex data is handled exactly as there (a real covariance draws the real and imaginary
    parts with equal variance; a complex covariance draws circular).

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over the 1-D grid.
    x : tensor
        Grid, shape (Nx,); the data grid, and the grid the draws are returned on.
    y : tensor
        Data, shape (Nbatch, ...), with the sample axis at ``dim``.
    noise : tensor
        Per-element noise variance, must broadcast with ``y``.
    mu : callable, optional
        Mean function ``mu(x[:, None]) -> (..., Nx)`` carried by the prior draw; default zero.
    size : int, optional
        Number of posterior draws. Default 1.
    dim : int, optional
        The sample axis of ``y`` (must not be 0). Default -1.
    jitter : float, optional
        Relative diagonal jitter for the prior-draw Cholesky.
    rcond, method, chunk
        Forwarded to :func:`posterior_mean_1d` for the Matheron correction.
    generator : torch.Generator, optional
        RNG for reproducible draws.

    Returns
    -------
    tensor
        Posterior draws of shape (size, *y.shape).
    """
    x = torch.as_tensor(x).reshape(-1)
    n = x.shape[-1]
    with torch.no_grad():
        K = kernel(x[:, None]).to_dense()
        L = _chol_jit(K, jitter)
        cov_c = K.is_complex()
        yp = y.movedim(dim, -1)                                  # sample axis last
        lead = yp.shape[:-1]
        np_ = noise.movedim(dim, -1)
        f_prior = _chol_matmul(L, _conv_normal((size, *lead, n), y.is_complex(), cov_c, y.real.dtype, y.device, generator))
        if mu is not None:
            f_prior = f_prior + mu(x[:, None])                   # prior carries the mean
        eps = np_.sqrt() * _conv_normal((size, *lead, n), y.is_complex(), cov_c, y.real.dtype, y.device, generator)
        resid = yp.unsqueeze(0) - f_prior - eps                  # (size, *lead, n)
        corr, _ = posterior_mean_1d(kernel, x, resid, np_.unsqueeze(0).expand(size, *np_.shape),
                                    dim=-1, rcond=rcond, method=method, chunk=chunk)
        f_post = f_prior + corr
    return f_post.movedim(-1, dim + 1 if dim >= 0 else dim)      # restore y's axis order (+ size)


def posterior_draws_2d(kernel1, kernel2, x1, x2, y, noise, mu=None, size=1, C1_rcond=None,
                       dims=(-2, -1), jitter=1e-10, method='woodbury', rcond=1e-12, cg_tol=1e-4,
                       cg_max_iter=5000, n_threads=1, generator=None):
    """
    Draw samples from the separable 2D GP posterior given data ``y`` via Matheron's rule.

    Pathwise (Matheron) conditioning, which is *exact*:

        f_post = f_prior + posterior_mean(y - f_prior - eps),   eps ~ N(0, noise)

    The prior draw uses the structured Kronecker Cholesky of the (``C1_rcond``-shrunk)
    ``C1 (x) C2`` and the correction reuses :func:`posterior_mean_2d`, so the ``(Nx1*Nx2)^2``
    covariance is never formed. Complex data is handled as in :func:`posterior_mean_2d`. A mean
    function ``mu`` is carried by the prior draw (and so appears in the posterior samples).

    Parameters
    ----------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` (outer/slow axis) and ``x2`` (inner/fast). ``kernel1`` may
        be complex.
    x1, x2 : tensor
        1D grids, shapes (Nx1,), (Nx2,).
    y : tensor
        Data, shape (..., Nx1, Nx2) with the two GP axes at ``dims``.
    noise : tensor
        Per-pixel noise variance, must broadcast with ``y``.
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)`` carried by the
        prior draw; default zero mean.
    size : int, optional
        Number of posterior draws. Default 1.
    C1_rcond : float, optional
        Relative eigenvalue floor in (0, 1] that shrinks the outer factor ``C1``'s spectrum toward
        flat (lift modes to ``C1_rcond * lambda_max``, trace-preserved; ``C1_rcond = 1`` -> the 1D
        solve; see :func:`gprlim.solvers.shrink`); applied to both the prior draw and the correction.
    dims : tuple, optional
        The two GP axes of ``y`` (mapped to the last two internally; default (-2, -1)).
    jitter : float, optional
        Relative diagonal jitter for the prior-draw Kronecker Cholesky.
    method, rcond, cg_tol, cg_max_iter, n_threads
        Forwarded to :func:`posterior_mean_2d` for the Matheron correction.
    generator : torch.Generator, optional
        RNG for reproducible draws.

    Returns
    -------
    tensor
        Posterior draws of shape (size, *y.shape).
    """
    x1 = torch.as_tensor(x1).reshape(-1)
    x2 = torch.as_tensor(x2).reshape(-1)
    n1, n2 = x1.shape[-1], x2.shape[-1]
    # move the two GP axes to the last two and flatten the rest into one batch (noise broadcast
    # to y first); reverse the move on return -- the prepended size axis shifts non-negative dims
    yf = y.movedim(dims, (-2, -1))
    lead = yf.shape[:-2]
    yf = yf.reshape(-1, n1, n2)
    nf = torch.broadcast_to(noise, y.shape).movedim(dims, (-2, -1)).reshape(-1, n1, n2)
    Nb = yf.shape[0]
    with torch.no_grad():
        C1 = shrink(kernel1(x1[:, None]).to_dense(), C1_rcond)
        C2 = kernel2(x2[:, None]).to_dense()
        dt = torch.promote_types(C1.dtype, C2.dtype)             # uniform factor dtype
        cov_c = dt.is_complex
        L = kron_cholesky([C1.to(dt), C2.to(dt)], jitter=jitter)  # chol of shrink(C1) (x) C2
        z = _conv_normal((size, Nb, n1 * n2), y.is_complex(), cov_c, y.real.dtype, y.device, generator)
        f_prior = _chol_matmul(L, z).reshape(size, Nb, n1, n2)
        if mu is not None:
            f_prior = f_prior + mu(x1[:, None], x2[:, None])     # prior carries the mean
        eps = nf.sqrt() * _conv_normal((size, Nb, n1, n2), y.is_complex(), cov_c, y.real.dtype, y.device, generator)
        resid = (yf.unsqueeze(0) - f_prior - eps).reshape(size * Nb, n1, n2)
        nr = nf.unsqueeze(0).expand(size, Nb, n1, n2).reshape(size * Nb, n1, n2)
        corr, _ = posterior_mean_2d(kernel1, kernel2, x1, x2, resid, nr, C1_rcond=C1_rcond, method=method,
                                    rcond=rcond, cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                                    n_threads=n_threads)
        corr = corr.reshape(size, Nb, n1, n2)
        out = (f_prior + corr).reshape(size, *lead, n1, n2)
    dest = tuple(d + 1 if d >= 0 else d for d in dims)            # +1 for the prepended size axis
    return out.movedim((-2, -1), dest)


def fit_axis_kernel(data, flags, noise, x, kernel, dim=-1, nsamp=512, iters=5,
                    opt='LBFGS', method='cholesky', rescale=True, var_mult=1.0,
                    prior_draws=100, generator=None):
    """
    Fit a kernel's hyperparameters by marginal likelihood along one axis.

    The GP/sample axis of ``data`` is ``dim`` (length ``len(x)``); every other axis is
    flattened into the pooled rows (so the full (Nbls, Ntimes, Nfreqs) cube can be passed
    directly -- ``dim=-1`` to fit the frequency kernel, ``dim=1`` the time kernel). Pools the
    rows (real/imag stacked for a real covariance, kept complex for a complex one), drawing
    the most-complete rows first so a few flagged pixels don't bias the fit; ``nsamp`` caps the
    fit batch. A thin wrapper over :func:`fit_kernel` for the per-axis inpaint workflow
    (compose with :func:`inpaint_1d` / :func:`inpaint_2d`).

    Parameters
    ----------
    data : tensor
        Data of shape (..., Nx, ...), with the sample axis at ``dim`` (real or complex).
    flags : tensor
        Boolean flags (True where flagged), broadcastable to ``data``; used only to rank rows
        by completeness for the fit, and scale kernel by empirical data variance.
    noise : tensor
        Noise variance, broadcastable to ``data``; flagged pixels assumed already down-weighted.
    x : tensor
        Axis grid of shape (Nx,).
    kernel : gpytorch.kernels.Kernel
        Kernel to fit; modified in place and returned.
    dim : int, optional
        The sample axis of ``data`` (length ``len(x)``). Default -1.
    nsamp : int, optional
        Maximum number of (most-complete) rows used in the fit.
    iters : int, optional
        Optimizer iterations.
    opt : str or torch.optim.Optimizer, optional
        Optimizer forwarded to :func:`fit_kernel` (e.g. 'LBFGS', 'Adam').
    method : str, optional
        Marginal-likelihood solver, 'cholesky' (stable) or 'woodbury'.
    rescale : bool, optional
        If True (default), rescale the kernel variance to match the data variance
        multiplied by var_mult. If kernel is not ScaleKernel, wraps it and returns.
    var_mult : float, optional
        Multiple of the empirical data variance to put into the kernel (default 1.0 =
        match ``data.var()`` exactly). Use ``< 1`` for a signal amplitude below the data variance
        (e.g. to leave out the noise contribution), ``> 1`` for above.        
    prior_draws : int, optional
        Number of 1D prior draws per factor used to estimate its variance for the unit/total
        rescaling (default 100).

    Returns
    -------
    gpytorch.kernels.Kernel
        The same ``kernel``, fit in place.
    """
    # move the sample axis to last and flatten the rest into pooled rows (flags/noise are
    # broadcast to data's shape first, then reshaped the same way -- like posterior_mean_1d)
    x = torch.as_tensor(x).reshape(-1)
    Nx = x.shape[-1]
    shape = data.shape
    data = data.movedim(dim, -1).reshape(-1, Nx)
    flags = torch.broadcast_to(flags, shape).movedim(dim, -1).reshape(-1, Nx)
    noise = torch.broadcast_to(noise, shape).movedim(dim, -1).reshape(-1, Nx)

    # rank rows by completeness, keep the most-complete `nsamp` so a few flagged pixels
    # don't bias the fit (noise already down-weights them; flags only rank here)
    good = 1.0 - flags.float().mean(1)
    order = torch.argsort(good, descending=True)[:min(nsamp, data.shape[0])]
    s, nz = data[order], noise[order]

    # a REAL covariance acts identically on real & imag -> stack them as extra rows and fit a
    # single real GP (cheaper); a COMPLEX covariance (e.g. a CarrierKernel axis) couples them,
    # so fit the complex data directly via the complex marginal likelihood.
    cov_complex = kernel(x[:2, None]).to_dense().is_complex()
    if s.is_complex() and not cov_complex:
        ys, nzs = stack_ri(s), torch.cat([nz, nz], 0)
    else:
        ys, nzs = s, nz

    fit_kernel(kernel, x[None, :, None], ys, nzs, Niter=iters, opt=opt, method=method)

    if rescale:
        from .kernels import ScaleKernel
        if not isinstance(kernel, ScaleKernel):
            p = next(kernel.parameters(), None)
            kernel = ScaleKernel(kernel)
            if p is not None:
                kernel = kernel.to(dtype=p.dtype, device=p.device)

        # draw prior from kernel to get total variance
        svar = prior_draws_1d(kernel, x, size=prior_draws, jitter=1e-10, generator=generator).var()

        # scale kernel1 to unit variance, and kernel2 to carry the whole composite variance
        dvar = data[~flags.expand(data.shape)].var() * var_mult
        kernel.outputscale = kernel.outputscale / svar * dvar

    return kernel


def fit_axis_kernel_2d(data, flags, noise, x1, x2, kernel1, kernel2, dims=(-2, -1),
                       rescale=True, var_mult=1.0, nsamp=512, iters=5, opt='LBFGS',
                       method='cholesky', prior_draws=100, generator=None):
    """
    Fit a separable 2D Kronecker kernel ``C1(x1) (x) C2(x2)`` by two independent 1D axis fits, then
    set the overall amplitude empirically, setting C1 to have unit diagonal and C2 to have
    the variance of the data.

    Parameters
    ----------
    data : tensor
        N-d data with the GP axes at ``dims``, assumed already de-trended (zero-mean); real or
        complex. Its variance sets the composite amplitude (see ``var_mult``).
    flags, noise : tensor
        Flag mask and noise (inverse weights), broadcastable to ``data``; forwarded to the two 1D
        fits (flags rank rows by completeness, noise down-weights flagged pixels).
    x1, x2 : tensor
        The two 1D grids (lengths ``data.shape[dims[0]]`` / ``data.shape[dims[1]]``).
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` / ``x2``. Fit in place, wrapped in a ScaleKernel if needed and
        amplitude-scaled; ``kernel1`` ends unit-variance, ``kernel2`` carries the total variance.
    dims : tuple of int, optional
        The two GP axes of ``data`` for (``kernel1``, ``kernel2``) respectively (default (-2, -1)).
    rescale : bool, optional
        If True (default), rescale the kernels so that the composite kernel variance
        match the data variance multiplied by var_mult.
    var_mult : float, optional
        Multiple of the empirical data variance to put into the composite kernel (default 1.0 =
        match ``data.var()`` exactly). Use ``< 1`` for a signal amplitude below the data variance
        (e.g. to leave out the noise contribution), ``> 1`` for above.
    nsamp, iters, opt, method
        Forwarded to the two per-axis :func:`fit_axis_kernel` calls (``iters=0`` sets only the
        amplitude, leaving the passed-in lengthscales untouched).
    prior_draws : int, optional
        Number of 1D prior draws per factor used to estimate its variance for the unit/total
        rescaling (default 100).
    generator : torch.Generator, optional
        RNG for the variance-estimation draws.

    Returns
    -------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        The fitted, ScaleKernel-wrapped factor kernels (``kernel1`` unit variance, ``kernel2``
        carrying the composite variance), ready for :func:`posterior_mean_2d` / :func:`inpaint_2d`.
    """
    from .kernels import ScaleKernel
    d1, d2 = dims

    # 1) two independent 1D axis fits (rescale=False: fit lengthscales only -- the 2D unit/total
    # amplitude split below does the rescaling, so inner rescaling would double-count / be redundant)
    fit_axis_kernel(data, flags, noise, x1, kernel1, dim=d1, nsamp=nsamp, iters=iters, opt=opt, method=method, rescale=False)
    fit_axis_kernel(data, flags, noise, x2, kernel2, dim=d2, nsamp=nsamp, iters=iters, opt=opt, method=method, rescale=False)

    # check if each kernel's outer layer is a ScaleKernel
    # if not, wrap in ScaleKernel
    if not isinstance(kernel1, ScaleKernel):
        p = next(kernel1.parameters(), None)
        kernel1 = ScaleKernel(kernel1)
        if p is not None:
            kernel1 = kernel1.to(dtype=p.dtype, device=p.device)
    if not isinstance(kernel2, ScaleKernel):
        p = next(kernel2.parameters(), None)
        kernel2 = ScaleKernel(kernel2)
        if p is not None:
            kernel2 = kernel2.to(dtype=p.dtype, device=p.device)

    if rescale:
        # draw prior from each kernel to get their respective total variance
        s1 = prior_draws_1d(kernel1, x1, size=prior_draws, jitter=1e-10, generator=generator).var()
        s2 = prior_draws_1d(kernel2, x2, size=prior_draws, jitter=1e-10, generator=generator).var()

        # scale kernel1 to unit variance, and kernel2 to carry the whole composite variance
        kernel1.outputscale = kernel1.outputscale / s1
        kernel2.outputscale = kernel2.outputscale / s2 * data[~flags.expand(data.shape)].var() * var_mult

    return kernel1, kernel2


def _sum_log_priors(module):
    """
    Sum of log-prior densities over every prior registered on ``module`` and its
    submodules (kernel, mean, likelihood), matching the prior term that
    ``ExactMarginalLogLikelihood`` adds to the marginal likelihood.
    """
    total = 0.0
    for _, mod, prior, closure, _ in module.named_priors():
        total = total + prior.log_prob(closure(mod)).sum()
    return total


def batched_log_prob(C, noise, y, method='woodbury', rcond=1e-15):
    """
    Per-row Gaussian log marginal likelihood ``log N(y_b; 0, C + diag(noise_b))``
    for a batch of rows sharing a single signal covariance ``C``, via the batched
    Woodbury identity (or a batched Cholesky).

    Building block for the batched hyperparameter fit (``fit_kernel(...,
    method=...)``). The quadratic data-fit term comes from the batched solve and
    the log-determinant from the matrix-determinant lemma on the same k x k
    capacitance, so no stochastic (SLQ) log-det is needed. ``y`` must already be
    mean-centered. The result is differentiable in ``C`` (hence in the kernel
    hyperparameters).

    Parameters
    ----------
    C : tensor
        Shared signal covariance (Nsamples, Nsamples), real or complex Hermitian PSD.
    noise : tensor
        Per-row noise variance (Nbatch, Nsamples), positive.
    y : tensor
        Mean-centered observations (Nbatch, Nsamples).
    method : str
        'woodbury' (fast when rank(C) < Nsamples) or 'cholesky'.
    rcond : float
        Relative eigenvalue cutoff for the 'woodbury' rank of C.

    Returns
    -------
    tensor
        Per-row log marginal likelihood of shape (Nbatch,).
    """
    C = C.squeeze()
    n = C.shape[-1]
    if method not in ('woodbury', 'cholesky'):
        raise ValueError(f"method must be 'woodbury' or 'cholesky', got {method!r}")

    alpha = None
    if method == 'woodbury':
        # share the signal eigendecomposition across the batch: C ~= U U^H. A
        # low-rank C (e.g. EigenKernel) has many repeated zero eigenvalues, on which
        # eigh can fail to converge -- fall back to the noise-regularized Cholesky
        # path (identical marginal likelihood) in that case.
        try:
            evals, evecs = torch.linalg.eigh(C)
        except torch.linalg.LinAlgError:
            method = 'cholesky'
        else:
            keep = evals > evals[-1] * rcond
            U = evecs[:, keep] * evals[keep].clamp_min(0).sqrt()        # (n, k)
            Ninv = 1.0 / noise                                         # (b, n)
            M = torch.einsum('nk,bn,nl->bkl', U.conj(), Ninv.to(U.dtype), U)  # (b, k, k)
            M.diagonal(dim1=-2, dim2=-1).add_(1.0)
            L = torch.linalg.cholesky(M)
            Dy = Ninv * y
            rhs = torch.einsum('nk,bn->bk', U.conj(), Dy).unsqueeze(-1)
            z = torch.cholesky_solve(rhs, L).squeeze(-1)
            alpha = Dy - Ninv * torch.einsum('nk,bk->bn', U, z)        # (C + diag(noise))^-1 y
            # determinant lemma: log|C + D| = log|D| + log|I + U^H D^-1 U|
            logdet = torch.log(noise).sum(-1) \
                + 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).real).sum(-1)
    if method == 'cholesky':
        A = C.unsqueeze(0) + torch.diag_embed(noise.to(C.dtype))       # (b, n, n)
        try:
            L = torch.linalg.cholesky(A)
        except torch.linalg.LinAlgError:
            # a sampled kernel matrix can be marginally non-PSD (e.g. a Sinc/composite axis whose
            # covariance dips slightly negative); lift the diagonal by a growing jitter tied to the
            # kernel's own scale until it factors (the fit is insensitive to a jitter this small)
            eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
            base = C.diagonal(dim1=-2, dim2=-1).real.mean().clamp_min(1e-30)
            L = None
            for p in range(-6, 1):
                try:
                    L = torch.linalg.cholesky(A + base * 10.0 ** p * eye)
                    break
                except torch.linalg.LinAlgError:
                    continue
            if L is None:
                raise
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).real).sum(-1)

    quad = (y.conj() * alpha).real.sum(-1)                             # (b,)
    return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


def fit_kernel(kernel, x, y, noise, mu=None, Niter=5, opt='LBFGS', method='cholesky',
               rcond=1e-15, thresh=None, loss=None, **kwargs):
    """
    Fit kernel hyperparameters IN PLACE by the batched marginal log likelihood.

    Optimizes ``kernel.parameters()`` (and ``mu.parameters()`` if a mean is given)
    against the per-row batched marginal likelihood (:func:`batched_log_prob`) of the rows
    of ``y`` sharing the covariance ``kernel(x)``. Matches gpytorch's
    ``ExactMarginalLogLikelihood`` for real, well-conditioned, small problems, but also
    handles the complex / large-N / low-rank cases gpytorch cannot -- which is why no
    ``ExactGP`` / dense-MLL path is provided (see the test suite for the gpytorch
    cross-check). Mean-centering, parameter priors, and the per-datapoint scaling are
    included, mirroring ``ExactMarginalLogLikelihood``.

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel, fit in place.
    x : tensor
        Training grid of shape (1, Nsamples, 1).
    y : tensor
        Training targets of shape (Nrows, Nsamples) (mean-subtracted if no ``mu``).
    noise : tensor
        Per-row noise variance of shape (Nrows, Nsamples).
    mu : gpytorch.means.Mean, optional
        Mean module, also fit; default zero mean.
    Niter : int
        Number of optimizer iterations.
    opt : str or torch.optim.Optimizer (class or instance)
        Optimizer; a string names a ``torch.optim`` class (e.g. 'LBFGS', 'Adam').
    method : str
        Batched marginal-likelihood solver, 'cholesky' (stable) or 'woodbury'.
    rcond : float
        Eigenvalue cutoff for the 'woodbury' batched path.
    thresh : float, optional
        Stop early once the loss change falls below this.
    loss : list, optional
        Append the per-iteration loss to this list if given.
    kwargs : dict
        Forwarded to the optimizer constructor.

    Returns
    -------
    loss : list
        Per-iteration loss values.
    opt : torch.optim.Optimizer
        The optimizer used.
    """
    # collect the learnable parameters (kernel, plus an optional mean module)
    params = list(kernel.parameters()) + (list(mu.parameters()) if mu is not None else [])
    if isinstance(opt, torch.optim.Optimizer):
        pass                                                  # already instantiated
    elif isinstance(opt, str):
        opt = getattr(torch.optim, opt)(params, **kwargs)     # 'LBFGS'/'Adam' -> instantiate
    else:
        opt = opt(params, **kwargs)                           # optimizer class -> instantiate

    # batched marginal likelihood: form the shared signal covariance once, then solve +
    # log-det per row, with mean-centering, priors, and per-datapoint scaling.
    def closure():
        opt.zero_grad()
        C = kernel(x).to_dense()
        resid = y if mu is None else y - mu(x)
        logprob = batched_log_prob(C, noise, resid, method=method, rcond=rcond)
        priors = _sum_log_priors(kernel)
        if mu is not None:
            priors = priors + _sum_log_priors(mu)
        l = -((logprob + priors) / y.shape[-1]).mean()
        l.backward()
        return l.detach()

    kernel.train()
    if mu is not None:
        mu.train()
    loss = [] if loss is None else loss
    for i in range(Niter):
        loss.append(opt.step(closure).cpu())
        if (thresh is not None) and (i > 0):
            if abs(loss[-1] - loss[-2]) < thresh:
                break

    return loss, opt
