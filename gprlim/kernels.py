from copy import deepcopy
import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import LinearOperator, KroneckerProductLinearOperator
from torch import Tensor

from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior, LogNormalPrior, NormalPrior
from gpytorch.kernels import RBFKernel, ScaleKernel, Kernel, AdditiveKernel
from gpytorch.means import ConstantMean

from . import utils
from . import solvers


class NonStationaryScaleKernel(Kernel):
    r"""
    Computes a non-stationary scale kernel, modeled as a linear relationship in
    log-amplitude.

    The lengthscale parameter is used as the linear coefficient (and is thus not a
    physical lengthscale).

    Parameters
    ----------
    train_x : torch.Tensor
        The input training samples, used to compute whitening x0 and dx0.
    ard_num_dims : int, optional
        Set this if you want a separate lengthscale for each input dimension. It should
        be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    batch_shape : torch.Size, optional
        Set this if you want a separate lengthscale for each batch of input data. It
        should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf{x_1}` is a
        :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    active_dims : tuple of int, optional
        Set this if you want to compute the covariance of only a few input dimensions.
        The ints corresponds to the indices of the dimensions. (Default: `None`.)
    lengthscale_prior : gpytorch.priors.Prior, optional
        Set this if you want to apply a prior to the lengthscale parameter.
        (Default: `NormalPrior(0, 0.1)`)
    lengthscale_constraint : gpytorch.constraints.Interval, optional
        Set this if you want to apply a constraint to the lengthscale parameter.
        (Default: `Interval`.)

    Attributes
    ----------
    lengthscale : torch.Tensor
        The lengthscale parameter. Size/shape of parameter depends on the ard_num_dims
        and batch_shape arguments.
    """

    has_lengthscale = True

    def __init__(
        self,
        train_x: torch.Tensor,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
        lengthscale_prior: Optional[Prior] = None,
        **kwargs,
    ):
        super(Kernel, self).__init__()
        self.x0 = train_x.mean()
        self.dx0 = (train_x.max() - train_x.min()) / 2

        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = 0.0

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is None:
                lengthscale_prior = NormalPrior(0.0, 0.1)
            if not isinstance(lengthscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
            self.register_prior(
                "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
            )
            # set interval of -1, 1
            self.register_constraint("raw_lengthscale", Interval(-1, 1, torch.tanh, torch.atanh))

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    def forward(self, x1, x2, diag=False, **params):
        # whiten x1 and x2
        x1 = (x1 - self.x0) / self.dx0
        x2 = (x2 - self.x0) / self.dx0

        # compute k1 and k2
        k1 = (x1 * self.lengthscale).exp().sum(-1)
        k2 = (x2 * self.lengthscale).exp().sum(-1)

        if diag:
            return k1 * k2

        else:
            return k1.unsqueeze(-1) * k2.unsqueeze(-2)


class FixedNonStationaryKernel(Kernel):
    r"""
    A fixed non-stationary kernel.

    Parameters
    ----------
    func : callable
        Function applied to each input to build the non-stationary scaling.
    """
    has_lengthscale = False

    def __init__(self, func):

        super(Kernel, self).__init__()

        self.func = func

    def forward(self, x1, x2, diag=False, **params):
        # compute k1 and k2
        k1 = self.func(x1).sum(-1)
        k2 = self.func(x2).sum(-1)

        if diag:
            return k1 * k2
        else:
            return k1.unsqueeze(-1) * k2.unsqueeze(-2)


class SincKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Sinc kernel.

    Parameters
    ----------
    ard_num_dims : int, optional
        Set this if you want a separate lengthscale for each input dimension. It should
        be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    batch_shape : torch.Size, optional
        Set this if you want a separate lengthscale for each batch of input data. It
        should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf{x_1}` is a
        :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    active_dims : tuple of int, optional
        Set this if you want to compute the covariance of only a few input dimensions.
        The ints corresponds to the indices of the dimensions. (Default: `None`.)
    lengthscale_prior : gpytorch.priors.Prior, optional
        Set this if you want to apply a prior to the lengthscale parameter.
        (Default: `None`)
    lengthscale_constraint : gpytorch.constraints.Positive, optional
        Set this if you want to apply a constraint to the lengthscale parameter.
        (Default: `Positive`.)
    eps : float, optional
        The minimum value that the lengthscale can take (prevents divide by zero
        errors). (Default: `1e-6`.)

    Attributes
    ----------
    lengthscale : torch.Tensor
        The lengthscale parameter. Size/shape of parameter depends on the ard_num_dims
        and batch_shape arguments.
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return torch.sinc(self.covar_dist(x1_, x2_, square_dist=False, diag=diag, **params))


class EmpiricalKernel(Kernel):
    r"""
    A non-stationary kernel defined by a fixed, empirically-measured covariance
    matrix sampled on a 1D grid.

    The covariance ``cov[i, j] = K(x[i], x[j])`` is stored on the grid ``x``. On a
    forward pass, if the requested inputs coincide with ``x`` the stored covariance
    is returned exactly; otherwise it is interpolated onto the requested points.
    The default ``'cubic'`` mode is a natural tensor-product (bi)cubic spline
    implemented natively in PyTorch (differentiable, GPU-capable): the three
    second-derivative tables (d2/dx1^2, d2/dx2^2 and the cross d4/dx1^2 dx2^2) are
    precomputed once at construction via a shared tridiagonal factorization, so
    evaluation is a closed-form combine with no per-call solve. ``'linear'`` and
    ``'nearest'`` are also supported. A non-uniform ``x`` is supported, and inputs
    outside the range of ``x`` are clamped to the grid boundary.

    There are no learnable hyperparameters: all tensors are registered as buffers,
    so they follow the module under ``.to()`` / ``.double()`` / ``.cuda()``. Wrap in
    a ``ScaleKernel`` for a learnable amplitude.

    Parameters
    ----------
    cov : tensor
        Empirical covariance of shape ``(N, N)``.
    x : tensor
        1D array of the ``N`` sample locations ``cov`` is defined on.
    interp : str, optional
        Interpolation mode: ``'cubic'`` (default), ``'linear'`` or ``'nearest'``.
    """

    has_lengthscale = False

    def __init__(self, cov, x, interp='cubic', **kwargs):
        super().__init__(**kwargs)

        cov = torch.as_tensor(cov)
        x = torch.as_tensor(x).reshape(-1)
        N = x.numel()

        if tuple(cov.shape[-2:]) != (N, N):
            raise ValueError(f"cov must be (N, N) = ({N}, {N}), got {tuple(cov.shape)}")
        if interp not in ('cubic', 'linear', 'nearest'):
            raise ValueError(f"interp must be 'cubic', 'linear' or 'nearest', got {interp!r}")
        if interp == 'cubic' and N < 4:
            raise ValueError(f"'cubic' interpolation needs >= 4 grid points, got {N}")

        # an ascending grid is required (searchsorted and the spline assume it);
        # flip the covariance to match if x is descending
        if N > 1 and x[0] > x[-1]:
            x = x.flip(0)
            cov = cov.flip(-1).flip(-2)

        self.register_buffer('x', x)
        self.register_buffer('cov', cov)
        self.interp = interp

        # precompute the cubic-spline second-derivative tables. The (N-2)
        # tridiagonal system depends only on x, so factor it once and reuse.
        if interp == 'cubic':
            h = x[1:] - x[:-1]
            n = N - 2

            # interior rows are the standard moment equations; the two corner
            # rows impose the not-a-knot condition (continuous third derivative
            # across the 2nd and 2nd-last knots), which is far more accurate at
            # the boundary than a natural (M''=0) spline
            A = torch.zeros(n, n, dtype=cov.dtype, device=cov.device)
            A[range(n), range(n)] = 2.0 * (h[:-1] + h[1:])
            idx = torch.arange(n - 1)
            A[idx, idx + 1] = h[1:n]
            A[idx + 1, idx] = h[1:n]
            h0, h1, hl, hr = h[0], h[1], h[N - 3], h[N - 2]
            A[0, 0] = 2.0 * (h0 + h1) + h0 + h0 * h0 / h1
            A[0, 1] = h1 - h0 * h0 / h1
            A[n - 1, n - 1] = 2.0 * (hl + hr) + hr + hr * hr / hl
            A[n - 1, n - 2] = hl - hr * hr / hl
            LU = torch.linalg.lu_factor(A)

            def deriv(Y):
                # cubic-spline 2nd derivatives of Y (M, N) along the last axis
                slope = (Y[:, 1:] - Y[:, :-1]) / h
                rhs = 6.0 * (slope[:, 1:] - slope[:, :-1])
                mi = torch.linalg.lu_solve(*LU, rhs.transpose(-1, -2)).transpose(-1, -2)
                M = torch.zeros_like(Y)
                M[:, 1:-1] = mi
                # not-a-knot endpoints: extrapolate from the first/last two moments
                M[:, 0] = mi[:, 0] * (1.0 + h0 / h1) - mi[:, 1] * (h0 / h1)
                M[:, -1] = mi[:, -1] * (1.0 + hr / hl) - mi[:, -2] * (hr / hl)
                return M

            m_row = deriv(cov.transpose(-1, -2)).transpose(-1, -2)  # d2/dx1^2
            m_col = deriv(cov)                                       # d2/dx2^2
            m_both = deriv(m_row)                                    # d4/dx1^2 dx2^2
            self.register_buffer('m_row', m_row)
            self.register_buffer('m_col', m_col)
            self.register_buffer('m_both', m_both)

    def _matches(self, c):
        # True if the last axis of c equals the stored grid x
        return c.shape[-1] == self.x.numel() and torch.allclose(c, self.x.to(c.dtype))

    def _weights(self, q):
        # interval index k and the cubic-spline basis weights at query q
        x = self.x
        N = x.numel()
        q = q.clamp(x[0], x[-1])
        k = (torch.searchsorted(x, q.contiguous()) - 1).clamp(0, N - 2)
        h = x[k + 1] - x[k]
        A = (x[k + 1] - q) / h
        B = (q - x[k]) / h
        C = (A ** 3 - A) * h ** 2 / 6.0
        D = (B ** 3 - B) * h ** 2 / 6.0
        return k, A, B, C, D

    def _bicubic(self, a, b, diag):
        ka, Aa, Ba, Ca, Da = self._weights(a)
        kb, Ab, Bb, Cb, Db = self._weights(b)
        cov, Mr, Mc, Mb = self.cov, self.m_row, self.m_col, self.m_both

        # pass 1: spline along rows (x1) onto a -> value and d2/dx2^2 at column grid
        G = (Aa[:, None] * cov[ka] + Ba[:, None] * cov[ka + 1]
             + Ca[:, None] * Mr[ka] + Da[:, None] * Mr[ka + 1])
        Gpp = (Aa[:, None] * Mc[ka] + Ba[:, None] * Mc[ka + 1]
               + Ca[:, None] * Mb[ka] + Da[:, None] * Mb[ka + 1])

        # pass 2: spline along columns (x2) onto b
        if diag:
            i = torch.arange(a.shape[0], device=a.device)
            return Ab * G[i, kb] + Bb * G[i, kb + 1] + Cb * Gpp[i, kb] + Db * Gpp[i, kb + 1]
        return (Ab[None, :] * G[:, kb] + Bb[None, :] * G[:, kb + 1]
                + Cb[None, :] * Gpp[:, kb] + Db[None, :] * Gpp[:, kb + 1])

    def _bilinear(self, a, b, diag):
        ka, Aa, Ba, _, _ = self._weights(a)
        kb, Ab, Bb, _, _ = self._weights(b)
        cov = self.cov
        G = Aa[:, None] * cov[ka] + Ba[:, None] * cov[ka + 1]
        if diag:
            i = torch.arange(a.shape[0], device=a.device)
            return Ab * G[i, kb] + Bb * G[i, kb + 1]
        return Ab[None, :] * G[:, kb] + Bb[None, :] * G[:, kb + 1]

    def _nearest(self, a, b, diag):
        ia, ib = self._near_idx(a), self._near_idx(b)
        return self.cov[ia, ib] if diag else self.cov[ia][:, ib]

    def _near_idx(self, q):
        x = self.x
        N = x.numel()
        q = q.clamp(x[0], x[-1])
        j = torch.searchsorted(x, q.contiguous()).clamp(1, N - 1)
        return torch.where(q - x[j - 1] <= x[j] - q, j - 1, j)

    def forward(self, x1, x2, diag=False, **params):
        # drop the trailing feature dim (1D inputs); match the covariance dtype
        c1 = x1[..., 0].to(self.cov.dtype)
        c2 = x2[..., 0].to(self.cov.dtype)
        lead = c1.shape[:-1]
        N = self.x.numel()

        # exact-grid short circuit -> return the stored covariance, exactly
        if self._matches(c1) and self._matches(c2):
            if diag:
                return self.cov.diagonal(dim1=-2, dim2=-1).expand(c1.shape).clone()
            return self.cov.expand(*lead, N, N).clone()

        # otherwise interpolate; loop over any leading batch of query rows
        interp = {'cubic': self._bicubic, 'linear': self._bilinear,
                  'nearest': self._nearest}[self.interp]
        c1f = c1.reshape(-1, c1.shape[-1])
        c2f = c2.reshape(-1, c2.shape[-1])
        out = torch.stack([interp(c1f[k], c2f[k], diag) for k in range(c1f.shape[0])])

        if diag:
            return out.reshape(*lead, c1.shape[-1])
        return out.reshape(*lead, c1.shape[-1], c2.shape[-1])


def _spline_second_derivs(x, Y):
    """
    Not-a-knot cubic-spline second derivatives of the columns of ``Y`` (N, ...) on an
    ascending grid, for off-grid interpolation.

    Requires N >= 4.

    Parameters
    ----------
    x : tensor
        Ascending grid of shape (N,).
    Y : tensor
        Values to differentiate, of shape (N, ...); the spline acts along the first axis.

    Returns
    -------
    tensor
        Second derivatives, same shape as ``Y``.
    """
    N = x.numel()
    h = x[1:] - x[:-1]
    n = N - 2
    A = torch.zeros(n, n, dtype=x.dtype, device=x.device)
    A[range(n), range(n)] = 2.0 * (h[:-1] + h[1:])
    idx = torch.arange(n - 1)
    A[idx, idx + 1] = h[1:n]
    A[idx + 1, idx] = h[1:n]
    h0, h1, hl, hr = h[0], h[1], h[N - 3], h[N - 2]
    A[0, 0] = 2.0 * (h0 + h1) + h0 + h0 * h0 / h1
    A[0, 1] = h1 - h0 * h0 / h1
    A[n - 1, n - 1] = 2.0 * (hl + hr) + hr + hr * hr / hl
    A[n - 1, n - 2] = hl - hr * hr / hl

    sh = (-1,) + (1,) * (Y.dim() - 1)
    slope = (Y[1:] - Y[:-1]) / h.reshape(sh)
    rhs = 6.0 * (slope[1:] - slope[:-1])
    mi = torch.linalg.solve(A.to(rhs.dtype), rhs)
    M2 = torch.zeros_like(Y)
    M2[1:-1] = mi
    M2[0] = mi[0] * (1.0 + h0 / h1) - mi[1] * (h0 / h1)
    M2[-1] = mi[-1] * (1.0 + hr / hl) - mi[-2] * (hr / hl)
    return M2


def _spline_eval(x, Y, M2, q):
    """
    Evaluate a cubic spline (values ``Y``, second derivatives ``M2`` on grid ``x``) at
    query points ``q``.

    Queries are clamped to the grid range.

    Parameters
    ----------
    x : tensor
        Grid of shape (N,) the spline is defined on.
    Y : tensor
        Spline values of shape (N, ...).
    M2 : tensor
        Spline second derivatives of shape (N, ...), as from :func:`_spline_second_derivs`.
    q : tensor
        Query points of shape (Q,).

    Returns
    -------
    tensor
        Spline values at ``q``, of shape (Q, ...).
    """
    N = x.numel()
    q = q.to(x.dtype).clamp(x[0], x[-1]).contiguous()
    k = (torch.searchsorted(x, q) - 1).clamp(0, N - 2)
    hk = x[k + 1] - x[k]
    A = (x[k + 1] - q) / hk
    B = (q - x[k]) / hk
    C = (A ** 3 - A) * hk ** 2 / 6.0
    D = (B ** 3 - B) * hk ** 2 / 6.0
    sh = (-1,) + (1,) * (Y.dim() - 1)
    return (A.reshape(sh) * Y[k] + B.reshape(sh) * Y[k + 1]
            + C.reshape(sh) * M2[k] + D.reshape(sh) * M2[k + 1])


class EigenKernel(Kernel):
    r"""
    A low-rank covariance built from a fixed set of basis modes with learnable
    per-mode weights:

    .. math::
        K(x_1, x_2) = V(x_1)\,\mathrm{diag}(w)\,V(x_2)^H,

    where the columns of ``V`` are fixed modes (e.g. eigen/SVD modes estimated from
    simulation) sampled on a 1D grid ``x``, and ``w >= 0`` are learnable per-mode
    powers -- the eigen/SVD coefficients. This is the kernel for "estimate the modes
    from simulation, fine-tune their amplitudes on data": the modes are buffers
    (fixed) and only ``weights`` is optimized by the marginal likelihood.

    PSD by construction for ``w >= 0`` (rank = number of modes). On a forward pass the
    requested inputs are matched to ``x`` (exact) or the modes are interpolated to
    them (cubic by default), so it serves as both ``K(X, X)`` and the cross-covariance
    ``K(x*, X)``. For the full unraveled covariance, prefer the low-rank factor
    ``root()`` fed straight into the Woodbury solver rather than densifying ``K``.

    Parameters
    ----------
    eigvecs : tensor
        (N, M) basis modes sampled on ``x`` (real or complex).
    x : tensor
        (N,) grid the modes are defined on.
    weights : tensor, optional
        (M,) initial per-mode weights / powers (default ones).
    interp : str, optional
        Mode interpolation, 'cubic' (default) or 'linear'.
    weight_constraint : gpytorch.constraints.Interval, optional
        Constraint on ``weights`` (default ``Positive``). Pass any ``gpytorch``
        constraint (e.g. ``Interval(lo, hi)`` with scalar or per-mode tensor bounds) for
        full control. Mutually exclusive with ``weight_bounds``.
    weight_bounds : tuple, optional
        Shorthand ``(lower, upper)`` to bound the weights with an ``Interval`` --
        scalars (same band for all modes) or length-M tensors (a per-mode band, e.g.
        ``(0.5*w_sim, 2*w_sim)`` to keep each mode within a factor of its simulation
        value). If ``weights`` is omitted it defaults to the in-bounds midpoint.
    weight_prior : gpytorch.priors.Prior, optional
        Optional prior on ``weights``.
    """

    has_lengthscale = False

    def __init__(self, eigvecs, x, weights=None, interp='cubic',
                 weight_constraint=None, weight_bounds=None, weight_prior=None, **kwargs):
        super().__init__(**kwargs)

        eigvecs = torch.as_tensor(eigvecs)
        x = torch.as_tensor(x).reshape(-1)
        N, M = eigvecs.shape
        if x.numel() != N:
            raise ValueError(f"x has length {x.numel()} but eigvecs has {N} rows")
        if interp not in ('cubic', 'linear'):
            raise ValueError(f"interp must be 'cubic' or 'linear', got {interp!r}")

        if N > 1 and x[0] > x[-1]:
            x = x.flip(0)
            eigvecs = eigvecs.flip(0)

        self.register_buffer('x', x)
        self.register_buffer('eigvecs', eigvecs)
        self.interp = interp

        # learnable per-mode weights -- the eigen/SVD coefficients
        self.register_parameter('raw_weights', torch.nn.Parameter(torch.zeros(M)))
        if weight_constraint is not None and weight_bounds is not None:
            raise ValueError("pass either weight_constraint or weight_bounds, not both")
        if weight_bounds is not None:
            lower = torch.as_tensor(weight_bounds[0]).to(self.raw_weights)
            upper = torch.as_tensor(weight_bounds[1]).to(self.raw_weights)
            weight_constraint = Interval(lower, upper)
            if weights is None:                          # default to the in-bounds midpoint
                weights = 0.5 * (lower + upper) * torch.ones(M)
        elif weight_constraint is None:
            weight_constraint = Positive()
        self.register_constraint('raw_weights', weight_constraint)
        self.weights = torch.ones(M) if weights is None else torch.as_tensor(weights)
        if weight_prior is not None:
            self.register_prior('weight_prior', weight_prior,
                                lambda m: m.weights, lambda m, v: m._set_weights(v))

        # precompute the fixed modes' spline second derivatives for interpolation
        M2 = torch.zeros_like(eigvecs)
        if interp == 'cubic' and N >= 4:
            M2 = _spline_second_derivs(x, eigvecs)
        self.register_buffer('eigvecs_M2', M2)

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(self.raw_weights)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

    def root(self):
        """Low-rank factor U = V diag(sqrt(w)) so that K = U U^H -- feed this directly
        to the Woodbury solver for the full (unraveled) covariance instead of K."""
        return self.eigvecs * self.weights.clamp_min(0).sqrt()

    def _modes_at(self, q):
        # exact-grid short circuit, else interpolate each mode to the query points
        q = q.to(self.x.dtype)
        if q.shape[-1] == self.x.numel() and torch.allclose(q, self.x):
            return self.eigvecs
        return _spline_eval(self.x, self.eigvecs, self.eigvecs_M2, q)

    def forward(self, x1, x2, diag=False, **params):
        c1, c2 = x1[..., 0], x2[..., 0]
        lead = c1.shape[:-1]
        c1f = c1.reshape(-1, c1.shape[-1])
        c2f = c2.reshape(-1, c2.shape[-1])
        w = self.weights

        out = []
        for a, b in zip(c1f, c2f):
            V1 = self._modes_at(a)                       # (n1, M)
            V2 = self._modes_at(b)                       # (n2, M)
            if diag:
                out.append(((V1 * V2.conj()) * w).sum(-1).real)
            else:
                out.append((V1 * w) @ V2.conj().transpose(-1, -2))
        out = torch.stack(out)
        if diag:
            return out.reshape(*lead, c1.shape[-1])
        return out.reshape(*lead, c1.shape[-1], c2.shape[-1])


class TwinRBFKernel(Kernel):
    r"""
    A "twin RBF" kernel: a Gaussian (RBF) envelope modulating a cosine, so that the
    Fourier transform of the covariance (its PSD / delay spectrum) shows two
    symmetric Gaussian peaks at :math:`\pm\tau`,

    .. math::
        k(\Delta) = \exp(-\Delta^2 / 2\ell^2)\,\cos(2\pi\,\tau\,\Delta),
        \qquad
        \mathrm{FT}[k](f) \propto e^{-2\pi^2\ell^2(f-\tau)^2} + e^{-2\pi^2\ell^2(f+\tau)^2}.

    With :math:`\Delta` a frequency separation, :math:`\tau` is the delay (the
    Fourier conjugate), so the peaks sit at delay :math:`\pm\tau`. The two peaks
    share amplitude and width by construction (the cosine makes them symmetric):
    location :math:`\pm\tau` and width :math:`1/(2\pi\ell)`. The PSD is non-negative,
    so the kernel is positive semi-definite. ``k(0) = 1``; wrap in a ``ScaleKernel``
    for a learnable amplitude. The peaks are only distinct when the envelope spans at
    least one carrier period :math:`1/\tau`, i.e. :math:`\ell\,\tau \gtrsim 1`.

    Parameters
    ----------
    tau : float
        The (positive) delay / cosine frequency setting the peak location. Learnable;
        constrained positive by default.
    lengthscale : float
        The Gaussian envelope lengthscale (the PSD peak width is :math:`1/(2\pi\ell)`).
    tau_prior : gpytorch.priors.Prior, optional
        Optional prior on ``tau``.
    tau_constraint : gpytorch.constraints.Positive, optional
        Optional constraint on ``tau`` (default ``Positive``).
    """

    has_lengthscale = True

    def __init__(self, tau_prior=None, tau_constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name="raw_tau",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        if tau_constraint is None:
            tau_constraint = Positive()
        self.register_constraint("raw_tau", tau_constraint)
        if tau_prior is not None:
            self.register_prior(
                "tau_prior", tau_prior, lambda m: m.tau, lambda m, v: m._set_tau(v)
            )

    @property
    def tau(self):
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value):
        self._set_tau(value)

    def _set_tau(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_tau)
        self.initialize(raw_tau=self.raw_tau_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Gaussian envelope: scale the inputs by the lengthscale, then take distance
        env = self.covar_dist(x1.div(self.lengthscale), x2.div(self.lengthscale),
                              diag=diag, **params)
        envelope = torch.exp(-0.5 * env.pow(2))

        # cosine carrier on the raw lag -> two PSD peaks at +/- tau
        dist = self.covar_dist(x1, x2, diag=diag, **params)
        tau = self.tau[..., 0] if diag else self.tau
        carrier = torch.cos(2 * np.pi * tau * dist)

        return envelope * carrier


class CarrierKernel(Kernel):
    r"""
    Modulate a real, even (stationary) base kernel by a complex carrier, producing a
    complex Hermitian kernel whose PSD is the base kernel's band shifted to a center
    frequency ``tau``:

    .. math::
        K(x_1, x_2) = K_\mathrm{base}(x_1, x_2)\, e^{+2\pi i\,\tau\,(x_1 - x_2)} .

    The ``+2 pi i`` sign follows the standard ``numpy``/``torch`` FFT convention: a draw
    from this covariance has its fringe-rate spectrum peak at ``+tau``.

    Equivalently ``K = D K_base D^H`` with ``D = diag(exp(+2j*pi*tau*x))`` (a unitary
    diagonal congruence), so ``K`` is Hermitian PSD whenever the base kernel is (e.g.
    ``RBFKernel`` / ``SincKernel``). This is the complex, one-sided counterpart of
    ``TwinRBFKernel`` (a real, two-sided ``+/- tau`` band): the carrier turns the base kernel's
    symmetric band into a one-sided band centered at ``tau``, as an asymmetric
    fringe-rate PSD requires. ``tau = 0`` recovers the real base kernel.

    The output is complex, so use it with the complex-capable solvers (``gpr_invert`` /
    ``batched_log_prob`` / ``fit_kernel(method=...)``), not gpytorch's real-only
    ``ExactMarginalLogLikelihood``. Fitting works through PyTorch's Wirtinger autograd:
    the real hyperparameters flow through the complex covariance to the real marginal
    likelihood with correct gradients.

    Parameters
    ----------
    base_kernel : gpytorch.kernels.Kernel
        A real, even base kernel, e.g. ``ScaleKernel(SincKernel())``.
    tau : float, optional
        Carrier frequency / band center, real (may be negative); units conjugate to the
        inputs. Default 0 (-> the real base kernel).
    tau_constraint : gpytorch.constraints.Interval, optional
        Optional constraint on ``tau`` (default: unconstrained, since the center may be
        either sign).
    tau_prior : gpytorch.priors.Prior, optional
        Optional prior on ``tau``.
    """

    is_stationary = True

    def __init__(self, base_kernel, tau=0.0, tau_constraint=None, tau_prior=None, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.register_parameter('raw_tau',
                                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if tau_constraint is not None:
            self.register_constraint('raw_tau', tau_constraint)
        self.tau = tau
        if tau_prior is not None:
            self.register_prior('tau_prior', tau_prior,
                                lambda m: m.tau, lambda m, v: m._set_tau(v))

    @property
    def tau(self):
        c = getattr(self, 'raw_tau_constraint', None)
        return c.transform(self.raw_tau) if c is not None else self.raw_tau

    @tau.setter
    def tau(self, value):
        self._set_tau(value)

    def _set_tau(self, value):
        value = torch.as_tensor(value).to(self.raw_tau).reshape(self.raw_tau.shape)
        c = getattr(self, 'raw_tau_constraint', None)
        self.initialize(raw_tau=c.inverse_transform(value) if c is not None else value)

    def forward(self, x1, x2, diag=False, **params):
        # real, even envelope from the base kernel
        Kr = self.base_kernel(x1, x2, diag=diag, **params)
        Kr = Kr if diag else Kr.to_dense()
        # complex carrier on the SIGNED lag -> Hermitian, one-sided PSD
        c1, c2 = x1[..., 0], x2[..., 0]
        if diag:
            lag, tau = c1 - c2, self.tau
        else:
            lag, tau = c1.unsqueeze(-1) - c2.unsqueeze(-2), self.tau.unsqueeze(-1)
        return Kr * torch.exp(2j * math.pi * tau * lag)


class GaussSincKernel(Kernel):
    r"""
    A Gaussian-convolved Sinc covariance (equivalently a top-hat-truncated-Gaussian
    PSD in Fourier space), wrapping :func:`gauss_sinc_cov`. See appendix A2 of
    arXiv:1608.05854.

    The covariance is stationary and normalized (``k(0) = 1``). ``gauss_ls`` and
    ``sinc_ls`` are stored as fixed buffers, **not** learnable hyperparameters: the
    underlying evaluation goes through mpmath/numpy and is not differentiable, so
    this kernel cannot be trained by marginal likelihood. Wrap it in a
    ``ScaleKernel`` for a learnable amplitude. ``forward`` evaluates the covariance
    for whatever inputs it is given, so it supplies both ``K(X, X)`` and the
    cross-covariance ``K(x*, X)`` exactly (no interpolation).

    Parameters
    ----------
    gauss_ls : float
        Gaussian lengthscale (the envelope width).
    sinc_ls : float
        Sinc lengthscale (sets the Fourier top-hat width).
    high_prec : bool, optional
        Use mpmath arbitrary precision (default) vs the faster numpy approximation; see
        :func:`gauss_sinc_cov`.

    Notes
    -----
    Each non-diagonal call recomputes ``gauss_sinc_cov``, which is expensive in
    ``high_prec`` mode. If you evaluate repeatedly on a fixed grid (e.g. inside a
    marginal-likelihood loop alongside learnable kernels), precompute the matrix once
    and wrap it in :class:`EmpiricalKernel`, or ask for a cached variant.
    """

    has_lengthscale = False

    def __init__(self, gauss_ls, sinc_ls, high_prec=True, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("gauss_ls", torch.as_tensor(gauss_ls))
        self.register_buffer("sinc_ls", torch.as_tensor(sinc_ls))
        self.high_prec = high_prec

    def forward(self, x1, x2, diag=False, **params):
        c1, c2 = x1[..., 0], x2[..., 0]
        lead = c1.shape[:-1]

        # stationary and normalized -> the diagonal (coincident points) is k(0) = 1
        if diag:
            return torch.ones(c1.shape, dtype=x1.dtype, device=x1.device)

        gls, sls = float(self.gauss_ls), float(self.sinc_ls)
        square = x1 is x2
        c1f = c1.reshape(-1, c1.shape[-1])
        c2f = c2.reshape(-1, c2.shape[-1])

        # gauss_sinc_cov(x=cols, x2=rows) returns K(rows, cols); here rows <- x1,
        # cols <- x2, so the result is K(x1, x2). Loop over any leading batch.
        out = torch.stack([
            gauss_sinc_cov(row2, gls, sls, x2=(None if square else row1),
                           high_prec=self.high_prec, dtype=x1.dtype, device=x1.device)
            for row1, row2 in zip(c1f, c2f)
        ])
        return out.reshape(*lead, c1.shape[-1], c2.shape[-1])


def gauss_sinc_cov(x, gauss_ls, sinc_ls, x2=None, dtype=None, device=None, high_prec=True):
    """
    A Gaussian-convolved Sinc covariance model,
    or a top-hat truncated Gaussian Fourier space kernel.

    Convolution of a Gaussian and Sinc covariance function
    See appendix A2 of arxiv:1608.05854

    Parameters
    ----------
    x : tensor
        Sampling points of covariance
    gauss_ls : float
        Gaussian length scale
    sinc_ls : float
        Sinc length scale
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.
    dtype : torch.dtype, optional
        Output dtype
    device : torch.device, optional
        Output device
    high_prec : bool, optional
        If True use mpmath arbitrary precision
        library, otherwise use numpy.

    Returns
    -------
    tensor
    """
    sinc_ls = sinc_ls / np.pi

    # get distances
    arg = gauss_ls / np.sqrt(2) / sinc_ls
    xc = x / gauss_ls / np.sqrt(2)
    x2c = xc if x2 is None else x2 / gauss_ls / np.sqrt(2)
    dists = (x2c[:, None] - xc[None, :])

    # get unique dists
    ud, ui = torch.unique(dists, return_inverse=True)

    if high_prec:
        import mpmath
        fn = lambda z: mpmath.exp(-z**2) * (mpmath.erf(arg + 1j*z) + mpmath.erf(arg - 1j*z)).real
        K = 0.5 * torch.as_tensor(np.asarray(np.frompyfunc(fn, 1, 1)(ud.numpy()), dtype=float))
        K /= math.erf(arg)

    else:
        K = (0.5 * torch.exp(-ud**2) / torch.special.erf(arg) \
            * (torch.special.erf(arg + 1j*ud) + torch.special.erf(arg - 1j*ud))).real
        # replace nans with zero: in this limit, you should use high_prec
        # but this is a faster approximation
        K[torch.isnan(K)] = 0.0

    # expand back to NxM
    cov = K[ui]

    # fill dx = 0 if needed
    cov[torch.isclose(dists, torch.tensor(0., dtype=dists.dtype), atol=1e-7)] = 1.0

    if dtype is not None:
        cov = cov.to(dtype)
    if device is not None:
        cov = cov.to(device)

    return cov


class LinearLengthscaleKernel(Kernel):
    r"""
    Base for non-stationary kernels with a lengthscale that evolves linearly with the
    input, ``ell(x) = offset + slope * x``.

    Non-stationarity is realized by *input warping*: the inputs are mapped to a
    coordinate ``u(x) = \int dx / ell(x)`` in which the process is stationary with unit
    lengthscale, then a stationary kernel is applied to the warped lag. This is PSD by
    construction for any positive ``ell(x)`` and works for any stationary base kernel
    (subclasses define ``_stationary(d)``). With ``slope = 0`` it reduces exactly to the
    stationary kernel with lengthscale ``offset``.

    Parameters
    ----------
    offset : float
        Lengthscale at ``x = 0`` (positive; ``Positive`` constraint).
    slope : float
        d(lengthscale)/dx (unconstrained by default).
    offset_prior, slope_prior : gpytorch.priors.Prior, optional
        Optional priors.
    offset_constraint, slope_constraint : gpytorch.constraints.Interval, optional
        Optional constraints.
    """

    has_lengthscale = False

    def __init__(self, offset_prior=None, offset_constraint=None,
                 slope_prior=None, slope_constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter('raw_offset',
                                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter('raw_slope',
                                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint('raw_offset', offset_constraint or Positive())
        if slope_constraint is not None:
            self.register_constraint('raw_slope', slope_constraint)
        self.offset = 1.0                               # default ell(0) = 1 (stationary)
        if offset_prior is not None:
            self.register_prior('offset_prior', offset_prior,
                                lambda m: m.offset, lambda m, v: m._set_offset(v))
        if slope_prior is not None:
            self.register_prior('slope_prior', slope_prior,
                                lambda m: m.slope, lambda m, v: m._set_slope(v))

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value):
        self._set_offset(value)

    def _set_offset(self, value):
        value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    @property
    def slope(self):
        c = getattr(self, 'raw_slope_constraint', None)
        return c.transform(self.raw_slope) if c is not None else self.raw_slope

    @slope.setter
    def slope(self, value):
        self._set_slope(value)

    def _set_slope(self, value):
        value = torch.as_tensor(value).to(self.raw_slope)
        c = getattr(self, 'raw_slope_constraint', None)
        self.initialize(raw_slope=c.inverse_transform(value) if c is not None else value)

    def _warp(self, x):
        # u(x) = log1p(slope*x/offset) / slope  (-> x/offset as slope -> 0); the clamp
        # keeps ell(x)/offset = 1 + slope*x/offset positive.
        xx = x[..., 0]
        offset, slope = self.offset, self.slope
        z = (slope * xx / offset).clamp_min(-1.0 + 1e-6)
        safe = torch.where(slope.abs() > 1e-12, slope, torch.ones_like(slope))
        return torch.where(slope.abs() > 1e-12, torch.log1p(z) / safe, xx / offset)

    def forward(self, x1, x2, diag=False, **params):
        u1 = self._warp(x1).unsqueeze(-1)
        u2 = self._warp(x2).unsqueeze(-1)
        d = self.covar_dist(u1, u2, square_dist=False, diag=diag, **params)
        return self._stationary(d)

    def _stationary(self, d):
        raise NotImplementedError


class NS_RBFKernel(LinearLengthscaleKernel):
    r"""
    Non-stationary RBF with a linear lengthscale ``ell(x) = offset + slope * x``
    (input-warped, PSD by construction). Reduces to the stationary RBF with lengthscale
    ``offset`` when ``slope = 0``.
    """

    def _stationary(self, d):
        return torch.exp(-0.5 * d ** 2)


class NS_SincKernel(LinearLengthscaleKernel):
    r"""
    Non-stationary Sinc with a linear lengthscale ``ell(x) = offset + slope * x``
    (input-warped, PSD by construction). Reduces to the stationary Sinc with lengthscale
    ``offset`` when ``slope = 0``.
    """

    def _stationary(self, d):
        return torch.sinc(d)


class KroneckerKernel(Kernel):
    r"""
    Separable 2D time-frequency covariance ``K = Ct (x) Cf``, kept as a
    ``KroneckerProductLinearOperator`` -- the full ``(Ntimes*Nfreqs)^2`` covariance is
    never formed. ``Ct`` and ``Cf`` are the per-axis factor *kernels*, evaluated lazily on
    their 1D grids; autodiff flows through their parameters, and the structured solvers
    (:func:`gprlim.solvers.kron_woodbury_predict` / :func:`gprlim.solvers.kron_wiener_cg`)
    consume the dense factors (:meth:`factor_mats`) directly. The unravel convention is
    row-major: time is the slow / outer axis, frequency the fast / inner axis.

    Parameters
    ----------
    Ct : gpytorch.kernels.Kernel
        Time-axis factor kernel (may be complex-valued, e.g. a :class:`CarrierKernel`).
    Cf : gpytorch.kernels.Kernel
        Frequency-axis factor kernel.
    times : tensor
        1D time grid, shape (Ntimes,).
    freqs : tensor
        1D frequency grid, shape (Nfreqs,).
    """

    is_stationary = False

    def __init__(self, Ct, Cf, times, freqs, **kwargs):
        super().__init__(**kwargs)
        self.Ct = Ct
        self.Cf = Cf
        self.register_buffer('times', torch.as_tensor(times).reshape(-1))
        self.register_buffer('freqs', torch.as_tensor(freqs).reshape(-1))

    def factor_mats(self):
        """Dense per-axis covariance factors ``(Ct(times), Cf(freqs))``, promoted to a
        common dtype so a complex factor (e.g. a CarrierKernel time axis) and a real factor
        (real freq axis) compose."""
        # Factor-kernel parameters must be REAL: RBF/Sinc form a (real) squared distance
        # internally (clamp_min_(0)), which errors on complex tensors. A complex-valued
        # covariance must come from a CarrierKernel/complex EigenKernel evaluated on real
        # inputs -- not from casting the kernel/grids to complex.
        for name, p in self.named_parameters():
            if p.is_complex():
                raise ValueError(
                    f"KroneckerKernel factor-kernel parameter '{name}' is complex. Grids and "
                    "factor-kernel parameters must be real: build the factor kernels with "
                    ".double() (not .to(torch.cdouble)). The complex covariance is produced "
                    "by the CarrierKernel carrier (or a complex EigenKernel) on real inputs."
                )
        Ct = self.Ct(self.times.unsqueeze(-1)).to_dense()
        Cf = self.Cf(self.freqs.unsqueeze(-1)).to_dense()
        dtype = torch.promote_types(Ct.dtype, Cf.dtype)
        return Ct.to(dtype), Cf.to(dtype)

    def covariance(self):
        """The structured ``KroneckerProductLinearOperator`` ``Ct (x) Cf`` (never densified)."""
        return KroneckerProductLinearOperator(*[to_linear_operator(m) for m in self.factor_mats()])

    def cholesky(self, jitter=1e-10):
        r"""
        Implicit lower-triangular Cholesky factor ``L`` with ``L L^H = K``, kept as a
        ``KroneckerProductTriangularLinearOperator`` via ``chol(A (x) B) = chol(A) (x)
        chol(B)`` -- the full ``(Ntimes*Nfreqs)`` factor is never densified.

        Each axis kernel is factorized with ``torch.linalg.cholesky`` (which handles a
        complex-Hermitian kernel, unlike ``linear_operator``'s ``psd_safe_cholesky`` whose
        jitter lookup rejects complex dtypes), after adding a small per-axis ``jitter``
        relative to that kernel's mean diagonal for positive-definiteness.

        This is the **noiseless** prior-draw / whitening factor (``x = L w`` with ``w``
        standard [complex] normal). ``K + noise`` is not Kronecker and cannot be factored
        this way -- use the eigendecomposition (white noise) or a Woodbury solve (flags).
        The per-axis factorization is :func:`gprlim.solvers.kron_cholesky`.

        Parameters
        ----------
        jitter : float, optional
            Per-axis diagonal jitter as a fraction of each kernel's mean diagonal (default
            1e-10). Set 0 to disable.

        Returns
        -------
        KroneckerProductTriangularLinearOperator
        """
        return solvers.kron_cholesky(self.factor_mats(), jitter=jitter)

    def sample(self, sample_shape=(), jitter=1e-6, generator=None):
        r"""Prior draw ``x = L w`` with ``w`` standard [complex] normal of ``L``'s
        dtype. Returns an unraveled vector of length ``prod(N_i)``.

        Parameters
        ----------
        sample_shape : tuple of int, optional
            Leading batch shape (default ``()`` -> single draw).
        jitter : float, optional
            Per-axis Cholesky jitter (default 1e-6).
        generator : torch.Generator, optional
            Optional RNG.
        """
        L = self.cholesky(jitter=jitter)
        w = torch.randn(*sample_shape, L.shape[-1], 1, dtype=L.dtype, device=L.device, generator=generator)
        return (L @ w).squeeze(-1)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        K = self.covariance()
        if diag:
            return K.diagonal()
        return K


def to_eigen(kernel, x, unravel=False, rcond=1e-12):
    r"""
    Compile a kernel into an :class:`EigenKernel` by eigendecomposing ``K = kernel(x)``:
    the modes become fixed buffers and only the per-mode weights remain free (to be
    fine-tuned by marginal likelihood).

    With ``unravel=True``, ``kernel`` is the pair ``[time_kernel, freq_kernel]`` and ``x``
    the pair ``[times, freqs]``; each axis is compiled to an ``EigenKernel`` and the two are
    assembled into a 2D :class:`KroneckerKernel` (``K = Ct (x) Cf``) that never forms the
    full covariance. (To keep an axis parametric instead -- e.g. ``RBFKernel`` -- build the
    ``KroneckerKernel`` directly with that base kernel.)

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel or [time_kernel, freq_kernel]
        A kernel instance, or the time/frequency factor kernels if ``unravel``.
    x : tensor or [times, freqs]
        A 1D grid, or the time/frequency grids if ``unravel``.
    unravel : bool, optional
        Assemble a 2D ``KroneckerKernel`` of per-axis ``EigenKernel`` factors.
    rcond : float, optional
        Relative eigenvalue cutoff for the low-rank truncation.

    Returns
    -------
    EigenKernel or KroneckerKernel
        An ``EigenKernel`` (or a 2D ``KroneckerKernel`` if ``unravel``).
    """
    if unravel:
        kt, kf = kernel
        gt, gf = (torch.as_tensor(xi).reshape(-1) for xi in x)
        return KroneckerKernel(to_eigen(kt, gt, rcond=rcond), to_eigen(kf, gf, rcond=rcond), gt, gf)

    x = torch.as_tensor(x).reshape(-1)
    with torch.no_grad():
        K = kernel(x.unsqueeze(-1)).to_dense()
        if K.dim() > 2:
            K = K.squeeze()
    w, V = torch.linalg.eigh(K)
    keep = w > w[-1].clamp_min(0) * rcond

    return EigenKernel(V[:, keep], x, weights=w[keep].clamp_min(1e-30))


def default_time_kernel(
    freqs, bl_vec, lat, buffer=None, hw_mult=1.0, min_hw=0.5,
    ml_scale=1e2, fz_scale=1e-1, fr_scale=3e-3,
    only_amp=True, only_global_amp=False, negate=True
    ):
    r"""
    Default complex time (fringe-rate) kernel for a baseline, a sum of three components:

    1. CarrierKernel(RBFKernel) -- the primary "main lobe" at the zenith fringe rate (a
       drift-scan beam's main lobe peaks there): a narrow RBF envelope modulated to that
       carrier. Unit-amplitude reference (no ScaleKernel of its own).
    2. ScaleKernel(RBFKernel) -- the fringe-rate = 0 (sky-stationary / DC) component: a
       real RBF centered at frate 0, amplitude relative to the main lobe.
    3. CarrierKernel(ScaleKernel(SincKernel)) -- the full visible-sky fringe-rate band: a
       Sinc brick-wall of half-width = the sky-frate half-width, modulated to the band
       center, amplitude relative to the main lobe.

    all scaled by a universal ScaleKernel. The output kernel mixture takes **time in
    seconds** and is complex Hermitian (the carriers make the fringe-rate PSD one-sided /
    asymmetric, which a net sky fringe rate requires).

    The conjugate variable to time [s] is fringe rate [Hz], so a CarrierKernel ``tau`` is
    a fringe rate in Hz. Fringe rates come from the baseline geometry: the main-lobe
    (zenith) rate from :func:`gprlim.utils.zenith_frate` and the full sky band from
    :func:`gprlim.utils.sky_frates` (both need ``freqs`` in Hz and scale with frequency;
    per-baseline values are averaged into one shared kernel).

    Parameters
    ----------
    freqs : array-like
        Observing frequency/frequencies in **Hz**, shape (Nfreqs,). Used only to compute
        the fringe rates (which scale with frequency); averaged over the band.
    bl_vec : array-like
        ENU baseline vector(s) in **meters**, shape (Nbls, 3) or (3,). Averaged over
        baselines to build one shared kernel.
    lat : float
        Array latitude in **degrees**.
    buffer : float, optional
        Extra fringe-rate half-width [mHz] added to the sky-frate band (component 3),
        forwarded to :func:`gprlim.utils.sky_frates`. Default None.
    hw_mult : float, optional
        Multiplier on the sky-frate half-width (component 3), forwarded to
        :func:`gprlim.utils.sky_frates`. Default 1.0.
    min_hw : float, optional
        Floor [mHz] on the sky-frate half-width (component 3). Default 0.5.
    ml_scale : float, optional
        Scaling of the main-lobe kernel (and the overall kernel itself)
    fz_scale : float, optional 
        Scaling of the FR=0 RBF kernel *relative* to the ml_scale
    fr_scale : float, optional
        Scaling of the full FR range Sinc kernel *relative* to the ml_scale
    only_amp : bool, optional
        If True (default) freeze every shape parameter (the RBF/Sinc lengthscales and the
        carrier taus), leaving only the amplitudes (outputscales) free to fit.
    only_global_amp : bool, optional
        If True, freeze all parameters expect for a single global ScaleKernel parameter.
        Supersedes only_amp.
    negate : bool, optional
        If True (default) negate ``bl_vec``, flipping the sign of the computed fringe
        rates so a covariance draw's fringe-rate spectrum lands at the physical (not
        mirror) rate under the numpy/torch FFT convention used by :class:`CarrierKernel`.

    Returns
    -------
    gpytorch.kernels.Kernel
        The complex Hermitian time-kernel mixture (takes time in seconds).
    """
    if negate:
        bl_vec = -bl_vec

    ## Main Lobe Kernel
    # get the main lobe fringe rate
    ml_frate = utils.zenith_frate(freqs, bl_vec, lat).mean() * 1e-3  # Hz

    # width of main lobe same for all bls, with tight prior
    ml = RBFKernel(
        lengthscale_constraint=Interval(0, 1e4),
        lengthscale_prior=LogNormalPrior(np.log(750.0), 0.3)
    )
    ml.lengthscale = 500.0  # sec, calibrated to sims

    # set frate of main lobe with tight prior
    ml = CarrierKernel(
        ml,
        tau=ml_frate,
        tau_constraint=Interval(-50e-3, 50e-3),
        tau_prior=NormalPrior(ml_frate, ml_frate * 0.01),
    )

    ## FR=0 Kernel: with amplitude relative to main lobe kernel
    fz = ScaleKernel(
        RBFKernel(lengthscale_constraint=Interval(1e1, 1e5),
                  lengthscale_prior=LogNormalPrior(np.log(1e3), 0.3)),
        outputscale_constraint=Interval(1e-5, 1e3),
        outputscale_prior=LogNormalPrior(np.log(fz_scale), 5.0),
    )
    fz.base_kernel.lengthscale = 3e3  # sec, calibrated to sims
    fz.outputscale = fz_scale

    ## Broad kernel: with amplitude relative to main lobe kernel
    # get frates of sky
    center, half_width = utils.sky_frates(freqs, bl_vec, lat, buffer=buffer,
                                          hw_mult=hw_mult, min_hw=min_hw)
    # reduce per-baseline (Nbls,) tensors to scalars: the carrier tau / sinc lengthscale
    # are scalar params, and scalar Interval bounds keep the kernel printable
    center = float(center.mean()) * 1e-3
    ls = 1 / (2 * float(half_width.mean()) * 1e-3)
    sinc = ScaleKernel(
        SincKernel(lengthscale_constraint=Interval(ls * .8, ls * 1.2),
                   lengthscale_prior=LogNormalPrior(np.log(ls), 0.3)),
        outputscale_constraint=Interval(1e-5, 1e4),
        outputscale_prior=LogNormalPrior(np.log(fr_scale), 5.0)
    )
    sinc.outputscale = fr_scale
    sinc.base_kernel.lengthscale = ls
    sinc = CarrierKernel(
        sinc,
        tau=center,
        tau_constraint=Interval(center - abs(center) * .2, center + abs(center) * .2),
        tau_prior=NormalPrior(center, center * 0.01),
    )

    # fix parameters if needed
    if only_amp or only_global_amp:
        ml.raw_tau.requires_grad_(False)
        ml.base_kernel.raw_lengthscale.requires_grad_(False)
        fz.base_kernel.raw_lengthscale.requires_grad_(False)
        sinc.raw_tau.requires_grad_(False)
        sinc.base_kernel.base_kernel.raw_lengthscale.requires_grad_(False)

    if only_global_amp:
        # also fix scale parameters
        sinc.base_kernel.raw_outputscale.requires_grad_(False)
        fz.raw_outputscale.requires_grad_(False)


    ## Add them all together with a universal scaling
    k = ScaleKernel(
        ml + fz + sinc,
        outputscale_constraint=Interval(1e-5, 1e10),
        outputscale_prior=LogNormalPrior(np.log(ml_scale), 5.0),
    )
    k.outputscale = ml_scale

    return k


def default_freq_kernel(
    bl_vec, ml_scale=1e2, pf_scale=1e-1, wd_scale=1e-3, lk_scale=1e-3, lk_kern='twinrbf',
    buffer=150.0, min_delay=50.0, only_amp=True, only_global_amp=False, real=True):
    r"""
    Default frequency kernel composed of

    1. RBFKernel for main lobe centered at delay = 0 ns
    2. pitchfork at +/- baseline horizon: a real TwinRBFKernel (``real=True``,
       default) or two CarrierKernel(RBFKernel) (``real=False``)
    3. SincKernel for -horizon < delays < horizon
    4. SincKernel for supra-horizon leakage or a wide
       TwinRBFKernel at pitchfork delays.

    all scaled by a universal ScaleKernel. The output kernel
    mixture takes frequency in MHz.

    The conjugate variable to frequency [MHz] is delay [us], so a carrier ``tau`` is
    a delay in microseconds and a SincKernel of lengthscale ``l`` has a brick-wall
    delay band of +/- 1 / (2 l). The horizon delay |b| / c comes from
    :func:`gprlim.utils.sky_delay` (returned in ns, converted to us here). The
    pitchfork covers the symmetric +/- horizon pair: with ``real=True`` (default) a
    single real ``TwinRBFKernel`` (Gaussian-windowed cosine), so the whole kernel is
    real-valued; with ``real=False`` two ``CarrierKernel``s (one per sign), giving a
    complex-dtype (Hermitian, zero-imaginary) kernel for the complex solver.

    Parameters
    ----------
    bl_vec : array-like
        ENU baseline vector(s) in meters, shape (3,) or (Nbls, 3). The horizon
        delay is averaged over baselines if more than one is given.
    ml_scale : float, optional
        Overall scaling of the main-lobe (and the entire kernel)
    pf_scale : float, optional
        Scaling of pitchfork *relative* to main-lobe scale (variance)
    wd_scale : float, optional
        Scaling of full horizon wedge *relative* to main-lobe scale
    lk_scale : float, optional
        Scaling of supra-horizon leakage *relative* to main-lobe scale
    lk_kern : str, optional
        Kernel for the leakage term: ['sinc', 'twinrbf'] (default: 'sinc').
        'sinc' : wide sinc spanning +/- horizon + buffer
        'twinrbf': double RBF at +/- pitchfork delays but with wider lobes
    buffer : float, optional
        Extra delay [ns] added beyond the horizon for the supra-horizon component
        (4). Default None (-> 0, i.e. component 4 shares component 3's band).
    min_delay : float, optional
        Floor [ns] on the horizon delay (short baselines), and the fiducial inner
        delay scale setting the delay=0 main-lobe and pitchfork widths. Default 50.
    only_amp : bool, optional
        If True (default) freeze every shape parameter (all lengthscales and
        carrier taus), leaving only the amplitudes (outputscales) free to fit.
    only_global_amp : bool, optional
        If True, freeze all parameters expect for a single global ScaleKernel parameter.
        Supersedes only_amp.
    real : bool, optional
        If True (default) model the pitchfork with one real ``TwinRBFKernel``, so the
        returned kernel is real-valued (use it with the real GP / real-imag-stacked
        path). If False, use two ``CarrierKernel``s and return a complex-dtype kernel
        (use the complex solver).

    Returns
    -------
    kernel mixture
    """
    buf = 0.0 if buffer is None else buffer

    # horizon / supra-horizon delays [us] (delay is conjugate to frequency in MHz);
    # `width` is the fixed inner delay scale for the delay=0 main lobe / pitchfork
    horizon = float(utils.sky_delay(bl_vec, buffer=0.0, min_delay=min_delay).mean()) * 1e-3
    supra = float(utils.sky_delay(bl_vec, buffer=buf, min_delay=min_delay).mean()) * 1e-3
    width = 40e-3

    # delay -> freq lengthscales: RBF via l = 1 / (2 pi sigma); Sinc via l = 1 / (2 delay)
    # (float() so every derived Interval bound / scalar param stays a 0-dim scalar)
    ls_main = float(1 / (2 * np.pi * width))     # delay=0 main lobe & pitchfork width
    ls_wedge = float(1 / (2 * horizon))          # brick-wall band over +/- horizon
    ls_supra = float(1 / (2 * supra))            # brick-wall band over +/- (horizon + buffer)

    ## 1. Main lobe: real RBF centered at delay = 0 (unit-amplitude reference)
    ml = RBFKernel(
        lengthscale_constraint=Interval(0, 1e3),
        lengthscale_prior=LogNormalPrior(np.log(ls_main), 0.3),
    )
    ml.lengthscale = ls_main

    ## 2. Pitchfork: symmetric +/- horizon band, amplitude relative to the main lobe
    if real:
        # single real TwinRBFKernel (Gaussian-windowed cosine) -> kernel stays real
        pf = ScaleKernel(
            TwinRBFKernel(lengthscale_constraint=Interval(0, 1e3),
                          lengthscale_prior=LogNormalPrior(np.log(ls_main * 3), 0.3),
                          tau_constraint=Interval(horizon * 0.8, horizon * 1.2),
                          tau_prior=NormalPrior(horizon, horizon * 0.01)),
            outputscale_constraint=Interval(1e-6, 1e3),
            outputscale_prior=LogNormalPrior(np.log(pf_scale), 5.0),
        )
        pf.base_kernel.lengthscale = ls_main * 3
        pf.base_kernel.tau = horizon
        pf.outputscale = pf_scale

    else:
        # two CarrierKernels (one per sign), summed -> complex-dtype kernel
        pf = None
        for tau in (horizon, -horizon):
            horn = ScaleKernel(
                RBFKernel(lengthscale_constraint=Interval(0, 1e3),
                          lengthscale_prior=LogNormalPrior(np.log(ls_main * 3), 0.3)),
                outputscale_constraint=Interval(1e-6, 1e3),
                outputscale_prior=LogNormalPrior(np.log(pf_scale), 5.0),
            )
            horn.base_kernel.lengthscale = ls_main * 3
            horn.outputscale = pf_scale
            horn = CarrierKernel(
                horn, tau=tau,
                tau_constraint=Interval(min(tau * 0.8, tau * 1.2), max(tau * 0.8, tau * 1.2)),
                tau_prior=NormalPrior(tau, horizon * 0.01),
            )
            pf = horn if pf is None else pf + horn

    ## 3. Wedge interior: real Sinc brick-wall over -horizon < delay < horizon
    wedge = ScaleKernel(
        SincKernel(lengthscale_constraint=Interval(ls_wedge * 0.8, ls_wedge * 1.2),
                   lengthscale_prior=LogNormalPrior(np.log(ls_wedge), 0.3)),
        outputscale_constraint=Interval(1e-6, 1e4),
        outputscale_prior=LogNormalPrior(np.log(wd_scale), 5.0),
    )
    wedge.base_kernel.lengthscale = ls_wedge
    wedge.outputscale = wd_scale

    ## 4. Supra-horizon leakage
    if lk_kern == 'sinc':
        # wider real Sinc over +/- (horizon + buffer)
        supra_k = ScaleKernel(
            SincKernel(lengthscale_constraint=Interval(ls_supra * 0.8, ls_supra * 1.2),
                       lengthscale_prior=LogNormalPrior(np.log(ls_supra), 0.3)),
            outputscale_constraint=Interval(1e-6, 1e4),
            outputscale_prior=LogNormalPrior(np.log(lk_scale), 5.0),
        )
        supra_k.base_kernel.lengthscale = ls_supra
        supra_k.outputscale = lk_scale

    elif lk_kern == 'twinrbf':
        # real wider RBF at pitchfork delays: ls is the width, tau is the delay
        _ls = 1 / (2 * max([buf, 1.0]) * 1e-3)  # ls in MHz
        supra_k = ScaleKernel(
            TwinRBFKernel(lengthscale_constraint=Interval(_ls * 0.8, _ls * 1.2),
                          lengthscale_prior=LogNormalPrior(np.log(_ls), 0.3),
                          tau_constraint=Interval(horizon * 0.8, horizon * 1.2),
                          tau_prior=NormalPrior(horizon, horizon * 0.01)),
            outputscale_constraint=Interval(1e-6, 1e4),
            outputscale_prior=LogNormalPrior(np.log(lk_scale), 5.0),
        )
        supra_k.base_kernel.lengthscale = _ls
        supra_k.base_kernel.tau = horizon
        supra_k.outputscale = lk_scale

    # deactivate parameters if desired
    if only_amp or only_global_amp:
        ml.raw_lengthscale.requires_grad_(False)
        if real:
            pf.base_kernel.raw_lengthscale.requires_grad_(False)
            pf.base_kernel.raw_tau.requires_grad_(False)
        else:
            pf.kernels[0].base_kernel.base_kernel.raw_lengthscale.requires_grad_(False)
            pf.kernels[0].raw_tau.requires_grad_(False)
            pf.kernels[1].base_kernel.base_kernel.raw_lengthscale.requires_grad_(False)
            pf.kernels[1].raw_tau.requires_grad_(False)
        wedge.base_kernel.raw_lengthscale.requires_grad_(False)
        supra_k.base_kernel.raw_lengthscale.requires_grad_(False)

    if only_global_amp:
        if real:
            pf.raw_outputscale.requires_grad_(False)
        else:
            pf.kernels[0].raw_outputscale.requires_grad_(False)
            pf.kernels[1].raw_outputscale.requires_grad_(False)
        wedge.raw_outputscale.requires_grad_(False)
        supra_k.raw_outputscale.requires_grad_(False)

    ## Add them all together with a universal scaling
    k = ScaleKernel(
        ml + pf + wedge + supra_k,
        outputscale_constraint=Interval(1e-10, 1e10),
        outputscale_prior=LogNormalPrior(np.log(ml_scale), 5.0),
    )
    k.outputscale = ml_scale

    return k


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
    mean : gpytorch.means.Mean
        The constant mean function.
    covar : gpytorch.kernels.Kernel
        The nested covariance kernel mixture.
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



