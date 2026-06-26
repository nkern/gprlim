import math

import torch

from .solvers import (stack_ri, unstack_ri, promote_like, shrink, kron_cholesky,
                      gpr_invert, woodbury_batched, cholesky_batched,
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
                      rcond=1e-15, method='cholesky', chunk=None):
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
    mu : callable, optional
        Mean function ``mu(x[:, None]) -> (..., Nx)``; default zero mean.
    dim : int, optional
        The sample axis of ``y`` (must not be 0, the batch axis). Default -1.
    detrend : bool, optional
        If True, do another global mean-subtraction after subtracting mu from data.
    rcond, method, chunk
        Forwarded to the batched solver (see :func:`gprlim.solvers.gpr_invert`).

    Returns
    -------
    tensor
        Posterior mean: ``y`` with its ``dim`` axis replaced by length Npred.
    """
    if dim == 0:
        raise ValueError("dim must not be 0 (the batch axis); use a non-zero / negative dim.")
    if method == 'cg':
        raise NotImplementedError("'cg' is implemented only for the 2D path "
                                  "(posterior_mean_2d); the 1D dense covariance is small -- "
                                  "use 'cholesky' or 'woodbury'.")
    x = torch.as_tensor(x).reshape(-1)
    Nx = x.shape[-1]
    # move the sample axis to last and flatten the remaining axes into one batch
    yp = y.movedim(dim, -1)
    lead = yp.shape[:-1]
    yf = yp.reshape(-1, Nx)
    nf = noise.movedim(dim, -1).reshape(-1, Nx)

    with torch.no_grad():
        # get dense kernel
        gx = x if pred_x is None else torch.as_tensor(pred_x).reshape(-1)
        Cs = kernel(x[:, None]).to_dense()
        Cp = Cs if pred_x is None else kernel(gx[:, None], x[:, None]).to_dense()

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
            pred = unstack_ri(gpr_invert(Cs, nf, B=Cp, y=yc, rcond=rcond, method=method, chunk=chunk))
        else:
            Cs, Cp = promote_like(Cs, yc), promote_like(Cp, yc)
            pred = gpr_invert(Cs, nf, B=Cp, y=yc, rcond=rcond, method=method, chunk=chunk)
        pred = pred + mu_pred + trend

    return pred.reshape(*lead, pred.shape[-1]).movedim(-1, dim)


def posterior_mean_2d(kernel1, kernel2, x1, x2, y, noise, pred_x1=None, pred_x2=None,
                      eta=None, method='woodbury', mu=None, dims=(-2, -1), detrend=False,
                      rcond=1e-12, cg_tol=1e-8, cg_max_iter=1000, n_jobs=1):
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
        Prediction grids. NOT YET IMPLEMENTED -- pass None (predict on the (x1, x2) grid).
    eta : float, optional
        Time-decorrelation shrinkage of ``C1`` in [0, 1] (see :func:`gprlim.solvers.shrink`);
        default None (no shrink).
    method : str
        'woodbury' (default; low-rank, rank-truncated at ``rcond``), 'cg' (preconditioned CG,
        to ``cg_tol``, parallelized over the batch by ``n_jobs``), or 'cholesky' (densify
        ``C1 (x) C2`` -- small grids only).
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)``; default zero mean.
    dims : tuple, optional
        Dimensions to perform 2D posterior mean. Maps dims to (-2, -1) of working tensor.
    detrend : bool, optional
        If True, do another global mean-subtraction after subtracting mu from data.
    rcond, cg_tol, cg_max_iter, n_jobs
        Forwarded to the chosen structured solver.

    Returns
    -------
    tensor
        Posterior mean, shape (Nbatch, Nx1, Nx2).
    """
    if pred_x1 is not None or pred_x2 is not None:
        raise NotImplementedError("prediction at new 2D points is not yet implemented; pass "
                                  "pred_x1=pred_x2=None for the posterior mean on the grid.")
    x1 = torch.as_tensor(x1).reshape(-1)
    x2 = torch.as_tensor(x2).reshape(-1)
    Nx1, Nx2 = x1.shape[-1], x2.shape[-1]

    # move the two sample axes to the last two and flatten the rest into one batch
    yp = y.movedim(dims, (-2, -1))
    lead = yp.shape[:-2]
    yf = yp.reshape(-1, Nx1, Nx2)
    nf = noise.movedim(dims, (-2, -1)).reshape(-1, Nx1, Nx2)

    with torch.no_grad():
        # optional 2D mean function, then optional inverse-noise-weighted detrend; both are
        # subtracted before the (zero-mean) Wiener solve and added back to the prediction.
        # (pred grid == train grid until pred_x1/pred_x2 are implemented, so mu reused below.)
        mu_x = 0.0 if mu is None else mu(x1[:, None], x2[:, None])
        yc = yf - mu_x
        trend = 0.0
        if detrend:
            trend = mean_center(yc, nf, dim=(-2, -1))
            yc = yc - trend

        # dense per-axis factors (the (Nx1*Nx2)^2 covariance is never formed)
        C1 = kernel1(x1[:, None]).to_dense().detach()
        if eta:
            C1 = shrink(C1, eta)                          # eta: decorrelate the outer axis
        C2 = kernel2(x2[:, None]).to_dense().detach()

        # complex strategy: split real/imag for a real covariance, else solve directly
        # (promoting a real cov to complex for complex data).
        cov_dtype = torch.promote_types(C1.dtype, C2.dtype)
        split = yc.is_complex() and not cov_dtype.is_complex
        if yc.is_complex() and not split:
            cov_dtype = torch.promote_types(cov_dtype, yc.dtype)
        C1, C2 = C1.to(cov_dtype), C2.to(cov_dtype)
        if split:
            ys, nz = torch.cat([yc.real, yc.imag], 0), torch.cat([nf, nf], 0)
        else:
            ys, nz = yc, nf

        if method == 'woodbury':
            mm = kron_woodbury_predict(C1, C2, nz, ys, rcond=rcond)
        elif method == 'cg':
            mm, _ = kron_wiener_cg(C1, C2, nz, ys, tol=cg_tol, max_iter=cg_max_iter, n_jobs=n_jobs)
        elif method == 'cholesky':
            Cs = torch.kron(C1, C2)
            nb = ys.shape[0]
            mm = gpr_invert(Cs, nz.reshape(nb, -1), B=Cs, y=ys.reshape(nb, -1),
                            method='cholesky').reshape(nb, Nx1, Nx2)
        else:
            raise ValueError(f"method must be 'woodbury', 'cg' or 'cholesky', got {method!r}")

        # recombine real/imag, then restore the mean function + detrend offset
        pred = (unstack_ri(mm) if split else mm) + mu_x + trend

    return pred.reshape(*lead, *pred.shape[-2:]).movedim((-2, -1), dims)


def inpaint_1d(kernel, x, y, noise, flags, mu=None, dim=-1, detrend=False,
               rcond=1e-15, method='cholesky', chunk=None):
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
    rcond, method, chunk
        Forwarded to :func:`posterior_mean_1d` (the batched solver).

    Returns
    -------
    inp : tensor
        ``y`` with flagged pixels replaced by the posterior mean; good pixels untouched.
    mdl : tensor
        The full posterior-mean model over the grid (same shape/dtype as ``y``).
    """
    mdl = posterior_mean_1d(kernel, x, y, noise, mu=mu, dim=dim, detrend=detrend,
                            rcond=rcond, method=method, chunk=chunk)
    return torch.where(flags, mdl, y), mdl


def inpaint_2d(kernel1, kernel2, x1, x2, y, noise, flags, eta=None, mu=None, detrend=False,
               method='woodbury', rcond=1e-12, cg_tol=1e-8, cg_max_iter=1000, n_jobs=1):
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
        Data, shape (Nbatch, Nx1, Nx2) (real or complex).
    noise : tensor
        Per-pixel noise variance, must broadcast with ``y``; flagged pixels assumed already
        down-weighted (large variance).
    flags : tensor
        Boolean mask, same shape as ``y`` (True where flagged).
    eta : float, optional
        Time-decorrelation shrinkage of ``C1`` in [0, 1] (see :func:`gprlim.solvers.shrink`).
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)``.
    detrend : bool, optional
        Subtract an inverse-noise-weighted mean over the (x1, x2) axes before the solve and
        add it back (see :func:`posterior_mean_2d`).
    method, rcond, cg_tol, cg_max_iter, n_jobs
        Forwarded to :func:`posterior_mean_2d` (the structured solver).

    Returns
    -------
    inp : tensor
        ``y`` with flagged pixels replaced by the posterior mean; good pixels untouched.
    mdl : tensor
        The full posterior-mean model on the (x1, x2) grid (same shape/dtype as ``y``).
    """
    mdl = posterior_mean_2d(kernel1, kernel2, x1, x2, y, noise, eta=eta, mu=mu, detrend=detrend,
                            method=method, rcond=rcond, cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                            n_jobs=n_jobs)
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
    a = torch.randn(*batch, n, 1, dtype=torch.float64, generator=generator)
    if L.dtype.is_complex:
        b = torch.randn(*batch, n, 1, dtype=torch.float64, generator=generator)
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


def _conv_normal(shape, data_complex, cov_complex, generator=None):
    """Standard normal matching the Wiener-solve complex convention: real (real data);
    circular complex ``z = (a + i b)/sqrt(2)`` with ``E[|z|^2] = 1`` (complex data + complex
    covariance, real/imag coupled); or 'biparte' complex ``z = a + i b`` with real & imag each
    ``N(0,1)`` (complex data + real covariance, where the solve treats each part identically)."""
    a = torch.randn(*shape, dtype=torch.float64, generator=generator)
    if not data_complex:
        return a
    z = torch.complex(a, torch.randn(*shape, dtype=torch.float64, generator=generator))
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
        f_prior = _chol_matmul(L, _conv_normal((size, *lead, n), y.is_complex(), cov_c, generator))
        if mu is not None:
            f_prior = f_prior + mu(x[:, None])                   # prior carries the mean
        eps = np_.sqrt() * _conv_normal((size, *lead, n), y.is_complex(), cov_c, generator)
        resid = yp.unsqueeze(0) - f_prior - eps                  # (size, *lead, n)
        corr = posterior_mean_1d(kernel, x, resid, np_.unsqueeze(0).expand(size, *np_.shape),
                                 dim=-1, rcond=rcond, method=method, chunk=chunk)
        f_post = f_prior + corr
    return f_post.movedim(-1, dim + 1 if dim >= 0 else dim)      # restore y's axis order (+ size)


def posterior_draws_2d(kernel1, kernel2, x1, x2, y, noise, mu=None, size=1, eta=None,
                       jitter=1e-10, method='woodbury', rcond=1e-12, cg_tol=1e-8,
                       cg_max_iter=1000, n_jobs=1, generator=None):
    """
    Draw samples from the separable 2D GP posterior given data ``y`` via Matheron's rule.

    Pathwise (Matheron) conditioning, which is *exact*:

        f_post = f_prior + posterior_mean(y - f_prior - eps),   eps ~ N(0, noise)

    The prior draw uses the structured Kronecker Cholesky of the (``eta``-shrunk) ``C1 (x) C2``
    and the correction reuses :func:`posterior_mean_2d`, so the ``(Nx1*Nx2)^2`` covariance is
    never formed. Complex data is handled as in :func:`posterior_mean_2d`. A mean function
    ``mu`` is carried by the prior draw (and so appears in the posterior samples).

    Parameters
    ----------
    kernel1, kernel2 : gpytorch.kernels.Kernel
        Factor kernels over ``x1`` (outer/slow axis) and ``x2`` (inner/fast). ``kernel1`` may
        be complex.
    x1, x2 : tensor
        1D grids, shapes (Nx1,), (Nx2,).
    y : tensor
        Data, shape (Nbatch, Nx1, Nx2).
    noise : tensor
        Per-pixel noise variance, must broadcast with ``y``.
    mu : callable, optional
        2D mean function ``mu(x1[:, None], x2[:, None]) -> (..., Nx1, Nx2)`` carried by the
        prior draw; default zero mean.
    size : int, optional
        Number of posterior draws. Default 1.
    eta : float, optional
        Time-decorrelation shrinkage of ``C1`` in [0, 1] (see :func:`gprlim.solvers.shrink`);
        applied to both the prior draw and the correction.
    jitter : float, optional
        Relative diagonal jitter for the prior-draw Kronecker Cholesky.
    method, rcond, cg_tol, cg_max_iter, n_jobs
        Forwarded to :func:`posterior_mean_2d` for the Matheron correction.
    generator : torch.Generator, optional
        RNG for reproducible draws.

    Returns
    -------
    tensor
        Posterior draws of shape (size, Nbatch, Nx1, Nx2).
    """
    x1 = torch.as_tensor(x1).reshape(-1)
    x2 = torch.as_tensor(x2).reshape(-1)
    n1, n2, Nb = x1.shape[-1], x2.shape[-1], y.shape[0]
    with torch.no_grad():
        C1 = shrink(kernel1(x1[:, None]).to_dense(), eta)
        C2 = kernel2(x2[:, None]).to_dense()
        dt = torch.promote_types(C1.dtype, C2.dtype)             # uniform factor dtype
        cov_c = dt.is_complex
        L = kron_cholesky([C1.to(dt), C2.to(dt)], jitter=jitter)  # chol of shrink(C1) (x) C2
        z = _conv_normal((size, Nb, n1 * n2), y.is_complex(), cov_c, generator)
        f_prior = _chol_matmul(L, z).reshape(size, Nb, n1, n2)
        if mu is not None:
            f_prior = f_prior + mu(x1[:, None], x2[:, None])     # prior carries the mean
        eps = noise.sqrt() * _conv_normal((size, Nb, n1, n2), y.is_complex(), cov_c, generator)
        resid = (y.unsqueeze(0) - f_prior - eps).reshape(size * Nb, n1, n2)
        nr = noise.unsqueeze(0).expand(size, Nb, n1, n2).reshape(size * Nb, n1, n2)
        corr = posterior_mean_2d(kernel1, kernel2, x1, x2, resid, nr, eta=eta, method=method,
                                 rcond=rcond, cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                                 n_jobs=n_jobs).reshape(size, Nb, n1, n2)
    return f_prior + corr


def fit_axis_kernel(data, flags, noise, x, kernel, nsamp=512, iters=5,
                    opt='LBFGS', method='cholesky'):
    """
    Fit a kernel's hyperparameters by marginal likelihood along one axis (IN PLACE).

    Pools the rows (real/imag stacked for a real covariance, kept complex for a complex
    one), drawing the most-complete rows first so a few flagged pixels don't bias the fit;
    ``nsamp`` caps the fit batch. A thin wrapper over :func:`fit_kernel` for the per-axis
    inpaint workflow (compose with :func:`inpaint_1d` / :func:`inpaint_2d`).

    Parameters
    ----------
    data : tensor
        Data rows of shape (Nrows, Nx), real or complex.
    flags : tensor
        Boolean flags of shape (Nrows, Nx) (True where flagged); used only to rank rows by
        completeness for the fit.
    noise : tensor
        Noise variance of shape (Nrows, Nx); flagged pixels assumed already down-weighted.
    x : tensor
        Axis grid of shape (Nx,).
    kernel : gpytorch.kernels.Kernel
        Kernel to fit; modified in place and returned.
    nsamp : int, optional
        Maximum number of (most-complete) rows used in the fit.
    iters : int, optional
        Optimizer iterations.
    opt : str or torch.optim.Optimizer, optional
        Optimizer forwarded to :func:`fit_kernel` (e.g. 'LBFGS', 'Adam').
    method : str, optional
        Marginal-likelihood solver, 'cholesky' (stable) or 'woodbury'.

    Returns
    -------
    gpytorch.kernels.Kernel
        The same ``kernel``, fit in place.
    """
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
    return kernel


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
        L = torch.linalg.cholesky(A)
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
