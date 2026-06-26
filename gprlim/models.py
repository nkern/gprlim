import math

import torch

# batched dense (C + diag(N))^-1 solvers live in solvers.py; re-exported here (and used by
# posterior_mean) so the historical gprlim.models.* import path keeps working.
from .solvers import gpr_invert, gp_predict, woodbury_batched, cholesky_batched


def mean_center(y, noise):
    """
    Inverse-noise-weighted mean of ``y`` along the last axis (a per-row constant offset),
    for optional centering before a GP solve. For polynomial or other trends, use a real
    mean function instead.

    Parameters
    ----------
    y : tensor
        Data of shape (Nbatch, ..., Nx), real or complex.
    noise : tensor
        Noise variance, same shape as ``y`` (the inverse weights).

    Returns
    -------
    tensor
        The weighted mean of shape (Nbatch, ..., 1); subtract it from ``y`` to center.
    """
    w = noise.pow(-1)
    return (y * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True)


def posterior_mean(kernel, x, y, noise, input_x=None, mean=None,
                   rcond=1e-15, method='cholesky', chunk=None):
    """
    GP posterior mean (Wiener filter) ``Cp (Cs + diag(noise))^-1 (y - mu) + mu`` for a
    batch of rows sharing one covariance ``kernel`` over the grid ``x``.

    Evaluates the covariance from ``kernel`` directly (no GP model object). If the data
    are complex but the covariance is real, the covariance is promoted to complex
    (imag = 0) and the complex system solved in one go -- a real covariance acts
    identically on the real and imaginary parts, so this matches solving them separately.

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over ``x``.
    x : tensor
        Training grid of shape (1, Nsamples, 1).
    y : tensor
        Training targets of shape (Nrows, Nsamples), real or complex.
    noise : tensor
        Per-row noise variance of shape (Nrows, Nsamples).
    input_x : tensor, optional
        Prediction grid of shape (1, Npredict, 1); the cross-covariance K(input_x, x) is
        used. Default predicts at the training points.
    mean : callable, optional
        Mean function ``mean(x) -> (..., N)``; default zero mean.
    rcond : float
        Relative condition / eigenvalue cutoff.
    method : str
        Batched solver, 'cholesky' (default) or 'woodbury'.
    chunk : int, optional
        Process the batch in chunks of this size to bound memory.

    Returns
    -------
    tensor
        Posterior mean of shape (Nrows, Npredict) (Npredict = Nsamples by default).
    """
    with torch.no_grad():
        mu = 0.0 if mean is None else mean(x)
        yc = y - mu
        # promote a real covariance to complex for complex data (one complex solve)
        Cs = kernel(x).to_dense().squeeze()
        if yc.is_complex() and not Cs.is_complex():
            Cs = Cs.to(yc.dtype)
        if input_x is not None:
            # cross-covariance C* = K(x*, X) between prediction and training points
            Cp = kernel(input_x, x).to_dense().squeeze()
            if yc.is_complex() and not Cp.is_complex():
                Cp = Cp.to(yc.dtype)
            mu = 0.0 if mean is None else mean(input_x).to_dense().squeeze()
        else:
            Cp = Cs
        pred = gp_predict(Cs, noise, yc, Cp=Cp, rcond=rcond, method=method, chunk=chunk)
        pred = pred + mu
    return pred


def gp_inpaint(kernel, x, y, noise, flags, y_offset=None, y_scale=None, mean=None,
               unpack_complex=False, rcond=1e-15, method='cholesky', chunk=None):
    """
    Inpaint ``y`` at flagged pixels with the GP posterior mean.

    Computes the posterior mean over all points, copies it into the flagged pixels (good
    pixels untouched), then undoes any pre-centering / pre-scaling and recombines stacked
    real/imag parts. ``flags`` only marks where to insert; the down-weighting of flagged
    pixels must already be in ``noise`` (large variance).

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over ``x``.
    x : tensor
        Training grid of shape (1, Nsamples, 1).
    y : tensor
        Training targets of shape (Nrows, Nsamples) (already centered if ``y_offset`` given).
    noise : tensor
        Per-row noise variance of shape (Nrows, Nsamples); flagged pixels assumed already
        down-weighted (large variance).
    flags : tensor
        Boolean flags of shape (Nrows, Nsamples) marking where to insert the model.
    y_offset : tensor, optional
        Baseline subtracted before the solve, added back afterward (see :func:`mean_center`).
    y_scale : tensor, optional
        Scale undone (multiplied back) after inpainting and re-centering.
    mean : callable, optional
        Mean function passed to :func:`posterior_mean`; default zero mean.
    unpack_complex : bool
        If True, ``y`` is real with [real, imag] stacked along the row axis; recombine into
        a complex array after inpainting. Skipped automatically if ``y`` is already complex.
    rcond, method, chunk
        Forwarded to :func:`posterior_mean`.

    Returns
    -------
    inp : tensor
        Inpainted data (flagged pixels filled, good pixels kept).
    mdl : tensor
        The full posterior-mean model.
    """
    mdl = posterior_mean(kernel, x, y, noise, mean=mean, rcond=rcond, method=method, chunk=chunk)
    inp = y.clone()
    inp[flags] = mdl[flags]
    if y_offset is not None:
        inp = inp + y_offset
        mdl = mdl + y_offset
    if y_scale is not None:
        inp = inp * y_scale
        mdl = mdl * y_scale
    # recombine row-stacked [real, imag] -> complex (skipped if already complex)
    if unpack_complex and not inp.is_complex():
        inp = torch.complex(inp[:len(inp)//2], inp[len(inp)//2:])
        mdl = torch.complex(mdl[:len(mdl)//2], mdl[len(mdl)//2:])
    return inp, mdl


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


def fit_kernel(kernel, x, y, noise, mean=None, Niter=5, opt='LBFGS', method='cholesky',
               rcond=1e-15, thresh=None, loss=None, **kwargs):
    """
    Fit kernel hyperparameters IN PLACE by the batched marginal log likelihood.

    Optimizes ``kernel.parameters()`` (and ``mean.parameters()`` if a mean is given)
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
        Training targets of shape (Nrows, Nsamples) (mean-subtracted if no ``mean``).
    noise : tensor
        Per-row noise variance of shape (Nrows, Nsamples).
    mean : gpytorch.means.Mean, optional
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
    params = list(kernel.parameters()) + (list(mean.parameters()) if mean is not None else [])
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
        resid = y if mean is None else y - mean(x)
        logprob = batched_log_prob(C, noise, resid, method=method, rcond=rcond)
        priors = _sum_log_priors(kernel)
        if mean is not None:
            priors = priors + _sum_log_priors(mean)
        l = -((logprob + priors) / y.shape[-1]).mean()
        l.backward()
        return l.detach()

    kernel.train()
    if mean is not None:
        mean.train()
    loss = [] if loss is None else loss
    for i in range(Niter):
        loss.append(opt.step(closure).cpu())
        if (thresh is not None) and (i > 0):
            if abs(loss[-1] - loss[-2]) < thresh:
                break

    return loss, opt
