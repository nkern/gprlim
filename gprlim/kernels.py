from copy import deepcopy
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


class NonStationaryScaleKernel(Kernel):
    r"""
    Computes a non-stationary scale kernel, modeled as
    a linear relationship in log-amplitude. The lengthscale
    parameter is used as the linear coefficient (and is
    thus not a physical lengthscale).

    :param train_x: The input training samples, used to compute whitening x0 and dx0.
    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf{x_1}` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `NormalPrior(0, 0.1)`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Interval`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
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
    A fixed non-stationary kernel
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

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf{x_1}` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return torch.sinc(self.covar_dist(x1_, x2_, square_dist=False, diag=diag, **params))



class GaussSincKernel(Kernel):
    """
    A Gaussian-convolved Sinc covariance model,
    or a top-hat truncated Gaussian Fourier space kernel.

    Convolution of a Gaussian and Sinc covariance function
    See appendix A2 of arxiv:1608.05854
    """

    def __init__(self, x, gauss_ls, sinc_ls, x2=None, dtype=None, device=None, high_prec=True):
        """
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
        pass


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
        K /= special.erf(arg)

    else:
        K = (0.5 * torch.exp(-ud**2) / torch.special.erf(arg) \
            * (torch.special.erf(arg + 1j*ud) + torch.special.erf(arg - 1j*ud))).real
        # replace nans with zero: in this limit, you should use high_prec
        # but this is a faster approximation
        K[torch.isnan(K)] = 0.0

    # expand back to NxM
    cov = K[ui]

    # fill dx = 0 if needed
    cov[torch.isclose(dists, torch.tensor(0.), atol=1e-7)] = 1.0

    if dtype is not None:
        cov = cov.to(dtype)
    if device is not None:
        cov = cov.to(device)

    return cov


