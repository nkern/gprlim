from copy import deepcopy
import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np

import torch
from linear_operator.operators import LinearOperator
from torch import Tensor

import gpytorch
from gpytorch.constraints import Interval, Positive, GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.module import Module
from gpytorch.priors import Prior, LogNormalPrior, NormalPrior
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, Kernel, ConstantKernel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean

from .kernels import SincKernel, NonStationaryScaleKernel


class GPModel(ExactGP):
    """
    Subclass of gpytorch.model.ExactGP
    """
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        """
        Parameters
        ----------
        train_x : Tensor
            Training samples of shape (Nbatch, Nsamples, 1)
        train_y : Tensor
            Training targets of shape (Nbatch, Nsamples)
        likelihood : Likelihood object
            The likelihood of the data
        mean : Kernel object
            The mean function
        covar : Kernel object
            The covariance function
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean = mean
        self.covar = covar

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(self, input_x=None, rcond=1e-15, method='cholesky', chunk=None):
        """
        Get MAP prediction
            Cp (Cs + Cn)^-1 y

        Parameters
        ----------
        input_x : tensor
            Prediction samples. Default is train_inputs.
        rcond : float
            Relative condition / eigenvalue cutoff.
        method : str
            Batched solver, 'cholesky' (default) or 'woodbury'.
        chunk : int, optional
            Process the batch in chunks of this size to bound memory.

        Returns
        -------
        tensor
        """
        with torch.no_grad():
            # get mean-subtracted y
            mu = self.mean(self.train_inputs[0])
            y = self.train_targets - mu

            # compute Cs and Cn. If the data are complex but the covariance is real,
            # promote it to complex (imag = 0) and solve the complex system directly --
            # a real covariance acts identically on the real and imaginary parts, so this
            # gives the same answer as solving them separately, just in one complex solve.
            Cs = self.covar(self.train_inputs[0]).to_dense().squeeze()
            if y.is_complex() and not Cs.is_complex():
                Cs = Cs.to(y.dtype)
            Cn = self.likelihood.noise.detach()
            if input_x is not None:
                # cross-covariance C* = K(x*, X) between prediction and training points
                Cp = self.covar(input_x, self.train_inputs[0]).to_dense().squeeze()
                if y.is_complex() and not Cp.is_complex():
                    Cp = Cp.to(y.dtype)
                mu = self.mean(input_x).to_dense().squeeze()
            else:
                Cp = Cs

            # compute MAP
            pred = gp_predict(Cs, Cn, y, Cp=Cp, rcond=rcond, method=method, chunk=chunk)

            # add back mean
            pred += mu

        return pred

    def inpaint(self, flags, y_offset=None, y_scale=None, unpack_complex=False, rcond=1e-15, method='cholesky', chunk=None):
        """
        Inpaint the training data at flagged pixels

        Parameters
        ----------
        flags : tensor
            Flagged pixels ([Nbatch], Nsamples)
        y_offset : tensor
            Pre-centering of training data to add after
            inpainting
        y_scale : tensor
            Pre-scaling of training data to multiply
            after inpainting and re-centering
        unpack_complex : bool
            If True, the training data are real with [real, imag] stacked along the
            batch axis; recombine them into a complex array after inpainting. Ignored
            when the data are already complex (a native complex covariance was used),
            in which case the complex solve runs directly and nothing is unpacked.
        rcond : float
            relative condition for matrix inverse
        method : str
            Batched solver, 'cholesky' (default) or 'woodbury'.
        chunk : int, optional
            Process the batch in chunks of this size to bound memory.

        Returns
        -------
        inp_y : tensor
            Inpainted data
        mdl : tensor
            Inpaint model
        """
        # get MAP prediction of training data
        mdl = self.predict(rcond=rcond, method=method, chunk=chunk)

        # clone training data
        inp_y = self.train_targets.clone()
        inp_y[flags] = mdl[flags]

        # add centering if needed
        if y_offset is not None:
            inp_y += y_offset
            mdl += y_offset

        # scale if needed
        if y_scale is not None:
            inp_y *= y_scale
            mdl *= y_scale

        # recombine batch-stacked [real, imag] -> complex (skipped automatically if
        # the data are already complex, i.e. a native complex covariance was used)
        if unpack_complex and not inp_y.is_complex():
            inp_y = torch.complex(inp_y[:len(inp_y)//2], inp_y[len(inp_y)//2:])
            mdl = torch.complex(mdl[:len(mdl)//2], mdl[len(mdl)//2:])

        return inp_y, mdl


def fixednoise_gp_1d(
    train_x,
    train_y,
    mean,
    covar,
    inv_wgts=None,
    center_y=False,
    Ndeg=1,
    ):
    """
    Parameters
    ----------
    train_x : Tensor
        Training samples of shape (Nbatch, Nsamples, 1)
    train_y : Tensor
        Training targets of shape (Nbatch, Nsamples)
    mean : Kernel object
        Kernel for mean function
    covar : Kernel object
        Kernel for covariance function
    inv_wgts : Tensor
        Inverse data weights (i.e. the noise variance) of shape (Nbatch, Nsamples)
    center_y : bool
        Center (not scale) train_y.
    Ndeg : int
        Polynomial degree to use in centering.

    Returns
    -------
    model : GPModel object
        The GP model
    y_offset : tensor
        The centering of the training data.
    """
    # setup likelihood
    likelihood = FixedNoiseGaussianLikelihood(inv_wgts)

    # center targets
    y_offset = None
    if center_y:
        A = torch.stack([train_x[0].squeeze()**i for i in range(Ndeg)]).T
        W = inv_wgts.pow(-1)
        AtA = torch.einsum("sc,bs,sd->bcd", A, W, A)
        AtAinv = torch.linalg.pinv(AtA)
        # cast the real polynomial basis to train_y's dtype so complex data centers too
        # (a real basis fit to complex y -> complex coefficients); no-op for real data
        Ad, AtAinvd, Wd = (t.to(train_y.dtype) for t in (A, AtAinv, W))
        y_offset = torch.einsum("oc,bcd,xd,bx,bx->bo", Ad, AtAinvd, Ad, Wd, train_y)
        train_y = train_y - y_offset

    # setup GP model
    model = GPModel(train_x, train_y, likelihood, mean, covar)

    return model, y_offset


def woodbury(A, U):
    """
    Perform woodbury inversion of (A + UV)^-1
    assuming A is diagonal and U = V.T

    Parameters
    ----------
    A : tensor
        Diagonal matrix of shape (Nsamples,)
    U : tensor
        SVD decomposition matrix of shape (Nsamples, Nmodes)
        where Nmodes < Nsamples
        
    Returns
    -------
    tensor
    """
    k = U.shape[1]
    A_inv_diag = 1. / A
    B_inv = torch.linalg.inv(torch.eye(k) + (U.T * A_inv_diag) @ U)
    return A_inv_diag.diag() - (A_inv_diag.reshape(-1, 1) * U @ B_inv @ U.T * A_inv_diag)


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


def gp_predict(Cs, Cn, train_y, Cp=None, rcond=1e-15, method='cholesky', chunk=None):
    """
    Compute MAP estimate of signal
        Cp (Cs + Cn)^-1 y

    Parameters
    ----------
    Cs : tensor
        Covariance matrix of signal (Nsamples, Nsamples)
    Cn : tensor
        Variance diagonal of noise ([Nbatch], Nsamples)
    train_y : tensor
        Training data ([Nbatch], Nsamples)
    Cp : tensor
        Covariance of training points and prediction points.
        Default is Cs, but can also predict at new points
        not in training vector (Npredict, Nsamples)
    rcond : float
        Relative condition / eigenvalue cutoff.
    method : str
        Batched solver, 'cholesky' (default) or 'woodbury'. See `gpr_invert`.
    chunk : int, optional
        Process the batch in chunks of this size to bound memory.

    Returns
    -------
    Tensor
    """
    if Cp is None:
        Cp = Cs

    # get prediction for all training samples
    return gpr_invert(Cs, Cn, B=Cp, y=train_y, rcond=rcond, method=method, chunk=chunk)


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

    Building block for the batched hyperparameter fit (``optimize_kernel(...,
    batched=...)``). The quadratic data-fit term comes from the batched solve and
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


def optimize_kernel(model, Niter=5, opt='LBFGS', thresh=None, loss=None,
                    batched=None, rcond=1e-15, **kwargs):
    """
    Optimize kernel hyperparameters with marginal log likelihood.

    Parameters
    ----------
    model : GPModel object
        The GP model to optimize
    Niter : int
        Number of total iterations
    opt : str or torch.optim.Optimizer subclass
        The optimzer to use. Must be attribute of torch.optim
    thresh : float
        If the loss residual falls below this, terminate optimization
    loss : list
        Append loss to this list if passed
    batched : str, optional
        If 'woodbury' or 'cholesky', evaluate the marginal likelihood with the
        batched solvers (see `batched_log_prob`) instead of gpytorch's dense
        ExactMarginalLogLikelihood -- much cheaper for a low-rank signal covariance
        ('woodbury' shares the eigendecomposition across the batch and works k x k
        per row). Mean-centering and parameter priors are included. Default None
        uses ExactMarginalLogLikelihood.
    rcond : float
        Eigenvalue cutoff for the 'woodbury' batched path.
    kwargs : dict
        Kwargs to pass Optimizer instantiation.
    """
    # setup optim
    if isinstance(opt, torch.optim.Optimizer):
        # already instantiated
        pass
    elif isinstance(opt, str):
        # a torch.optim name string (e.g. 'LBFGS', 'Adam') -> look up and instantiate
        opt = getattr(torch.optim, opt)(model.parameters(), **kwargs)
    else:
        # passed an uninstantiated optimizer
        # an optimizer class -> instantiate it
        opt = opt(model.parameters(), **kwargs)

    # set up the marginal-likelihood loss
    if batched is None:
        # gpytorch's exact (dense) marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        def closure():
            opt.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets).mean()
            loss.backward()
            return loss.detach()
    else:
        # batched marginal likelihood via the Woodbury / Cholesky solvers: form the
        # shared signal covariance once, then solve + log-det per row. Mirrors
        # ExactMarginalLogLikelihood (mean-centering, priors, per-datapoint scaling).
        def closure():
            opt.zero_grad()
            x = model.train_inputs[0]
            C = model.covar(x).to_dense()
            y = model.train_targets - model.mean(x)
            noise = model.likelihood.noise
            logprob = batched_log_prob(C, noise, y, method=batched, rcond=rcond)
            loss = -((logprob + _sum_log_priors(model)) / y.shape[-1]).mean()
            loss.backward()
            return loss.detach()

    model.train()
    model.likelihood.train()
    loss = [] if loss is None else loss
    for i in range(Niter):
        loss.append(opt.step(closure).cpu())
        if (thresh is not None) and (i > 0):
            if abs(loss[-1] - loss[-2]) < thresh:
                break

    return loss, opt


def multi_kernel_mixture(
    mean_fix_constant=False,
    mean_set_constant=None,
    mean_batch_constant=False,
    mean_prior_constant=None,
    mean_bound_constant=None,
    Nbatch=0,
    nonstn_scale=False,
    nonstn_fix=False,
    nonstn_set=None,
    nonstn_prior=None,
    nonstn_bound=None,
    train_x=None,
    kern0='sinc',
    **kwgs,
    ):
    """
    Setup the nested covariance kernel mixture:

    covar = ScaleKernel( SincKernel + ScaleKernel( SincKernel() + ... ) )

    where kern0 is the innermost SincKernel, and kernM is the outermost
    of M+1 sinc mixtures. Here we use SincKernel as a placeholder, but
    SincKernel or RBFKernel can be used. To add arbitrary number of covariance
    kernels, pass the notation kernM_xyz in the kwargs. Use mean_xyz as a template.
    Note that each sinc has a lengthscale and outputscale as parameters,
    each of which take fix, set, batch, and prior, options.

    Parameters
    ----------
    mean_fix_constant : bool
        Fix the mean function constant, making it not learnable
    mean_set_constant : float
        Set the initial value of the mean constant.
    mean_batch_constant : bool
        If True, create Nbatch mean constants for each batch element.
    mean_prior_constant : Prior object
        Set the prior for the mean constant hyperparameter.
    mean_bound_constant : Interval object
        Set constraints on the mean constant hyperparameter.
    nonstn_scale : bool
        If True, multiply a NonStationaryScale kernel at the end of
        the covariance mixture.
    nonstn_fix : bool
        If True, fix the non-stationary coefficient
    nonstn_set : float
        Set the value of the non-stationary coefficient
    nonstn_prior : Prior object
        Prior for nonstationary scale parameter.
    nonstn_bound : Internval object
        Bounds for nonstationary scale parameter.
    train_x : tensor
        Needed for NonStationaryScaleKernel instantiation
    kern0 : str
        If passed, create a kern0 object, with the following
        optional parameters. Options are ['sinc', 'rbf'].
        Defaults are the same as mean_constant defaults.
        To create more nested kernels, pass kern1='sinc', etc
    kern0_fix_lengthscale : bool
    kern0_fix_outputscale : bool
    kern0_set_lengthscale : float
    kern0_set_outputscale : float
    kern0_batch_lengthscale : bool
    kern0_batch_outputscale : bool
    kern0_prior_lengthscale : Prior
    kern0_prior_outputscale : Prior
    kern0_bound_lengthscale : Interval
    kern0_bound_outputscale : Interval

    Returns
    -------
    GPModel object
    """
    kernels = {'sinc': SincKernel, 'rbf': RBFKernel}
    # setup constant mean function
    mean = ConstantMean(
        batch_shape=torch.Size([Nbatch]) if mean_batch_constant else torch.Size([])
    )
    if mean_bound_constant is not None:
        mean.register_constraint('raw_constant', mean_bound_constant)
    if mean_fix_constant:
        mean.constant.requires_grad = False
    if mean_set_constant is not None:
        mean.constant = mean_set_constant
    if mean_prior_constant is not None:
        mean.register_prior('constant_prior', mean_prior_constant, 'constant')

    i = 0
    while True:
        if i > 0 and not kwgs.get(f'kern{i}', False):
            break

        lengthscale_batch = torch.Size([Nbatch]) if kwgs.get(f'kern{i}_batch_lengthscale') else None
        outputscale_batch = torch.Size([Nbatch]) if kwgs.get(f'kern{i}_batch_outputscale') else None

        if i == 0:
            kernel = kernels[kern0.lower()]
            covar = ScaleKernel(kernel(batch_shape=lengthscale_batch), batch_shape=outputscale_batch)
            kern = covar.base_kernel
        else:
            kernel = kernels[kwgs.get(f'kern{i}').lower()]
            covar = ScaleKernel(kernel(batch_shape=lengthscale_batch) + covar, batch_shape=outputscale_batch)
            kern = covar.base_kernel.kernels[0]
        if kwgs.get(f'kern{i}_bound_outputscale') is not None:
            covar.register_constraint('raw_outputscale', kwgs.get(f'kern{i}_bound_outputscale'))
        if kwgs.get(f'kern{i}_bound_lengthscale') is not None:
            kern.register_constraint('raw_lengthscale', kwgs.get(f'kern{i}_bound_lengthscale'))
        if kwgs.get(f'kern{i}_fix_lengthscale'):
            kern.raw_lengthscale.requires_grad = False
        if kwgs.get(f'kern{i}_fix_outputscale'):
            covar.raw_outputscale.requires_grad = False
        if kwgs.get(f'kern{i}_set_lengthscale') is not None:
            kern.lengthscale = kwgs.get(f'kern{i}_set_lengthscale')
        if kwgs.get(f'kern{i}_set_outputscale') is not None:
            covar.outputscale = kwgs.get(f'kern{i}_set_outputscale')
        if kwgs.get(f'kern{i}_prior_lengthscale') is not None:
            kern.register_prior(
                'lengthscale_prior',
                kwgs.get(f'kern{i}_prior_lengthscale'),
                'lengthscale'
                )
        if kwgs.get(f'kern{i}_prior_outputscale') is not None:
            covar.register_prior(
                'outputscale_prior',
                kwgs.get(f'kern{i}_prior_outputscale'),
                'outputscale'
                )

        i += 1

    if nonstn_scale:
        covar = NonStationaryScaleKernel(train_x) * covar
        if nonstn_bound is not None:
            covar.kernels[0].register_constraint(
                'lengthscale',
                nonstn_bound
            )
        if nonstn_fix:
            covar.kernels[0].raw_lengthscale.requires_grad = False
        if nonstn_set is not None:
            covar.kernels[0].lengthscale = nonstn_set
        if nonstn_prior is not None:
            covar.kernels[0].register_prior(
                'lengthscale_prior',
                nonstn_prior,
                'lengthscale'
                )

    return mean, covar

