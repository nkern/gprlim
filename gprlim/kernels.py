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
			# not enforced...
			self.register_constraint("raw_lengthscale", Interval(-1,1,None,None))

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

