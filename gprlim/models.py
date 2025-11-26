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

	def predict(self, input_x=None, rcond=1e-15):
		"""
		Get MAP prediction
			Cp (Cs + Cn)^-1 y

		Parameters
		----------
		input_x : tensor
			Prediction samples. Default is train_inputs.

		Returns
		-------
		tensor
		"""
		with torch.no_grad():
			# get mean-subtracted y
			mu = self.mean(self.train_inputs[0])
			y = self.train_targets - mu

			# compute Cs and Cn
			Cs = self.covar(self.train_inputs[0]).to_dense().squeeze()
			Cn = self.likelihood.noise.detach()
			if input_x is not None:
				Cp = self.covar(input_x).to_dense().squeeze()
				mu = self.mean(input_x).to_dense().squeeze()
			else:
				Cp = Cs

			# compute MAP
			pred = gp_predict(Cs, Cn, y, Cp=Cp, rcond=rcond)

			# add back mean
			pred += mu

		return pred

	def inpaint(self, flags, y_offset=None, to_complex=False, rcond=1e-15):
		"""
		Inpaint the training data at flagged pixels

		Parameters
		----------
		flags : tensor
			Flagged pixels ([Nbatch], Nsamples)
		y_offset : tensor
			Pre-centering of training data to add after
			inpainting
		to_complex : bool
			If True, assume training data [real, imag] are stacked,
			convert back to complex
		rcond : float
			relative condition for matrix inverse

		Returns
		-------
		tensor
		"""
		# get MAP prediction of training data
		pred = self.predict(rcond=rcond)

		# clone training data
		inp_y = model.train.targets.clone()
		inp_y[flags] = pred[flags]

		# add centering if needed
		if y_offset is not None:
			inp_y += y_offset

		# turn to complex if needed
		if to_complex:
			inp_y = torch.complex(inp_y[:len(inp_y)//2], inp_y[len(inp_y)//2:])

		return inp_y


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
		AtA = torch.einsum("sc,bs,sd->bcd", A, inv_wgts.pow(-1), A)
		AtAinv = torch.linalg.pinv(AtA)
		y_offset = torch.einsum("oc,bcd,xd,bx,bx->bo", A, AtAinv, A, inv_wgts.pow(-1), train_y)
		train_y = train_y - y_offset

	# setup GP model
	model = GPModel(train_x, train_y, likelihood, mean, covar, y_offset=y_offset)

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


def gpr_invert(C, N, B=None, y=None, rcond=1e-15):
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
	    
	Returns
	-------
	tensor
	"""
	if N.ndim == 1:
		# just use pinv
		out = torch.linalg.pinv(C + N.diag(), hermitian=True, rcond=rcond)
		if y is not None:
			out = out @ y
		if B is not None:
			out = B @ out

	else:
		assert y.ndim > 1, "If N has batch dim then y must too"
		if C.ndim > 2:
			assert B.ndim > 2, "If C has batch dim then B must too"
			# C also has batch dimension (not standard)
			# revert to slow pinv per batch
			out = []
			for i in range(len(N)):
				out.append(gpr_invert(C[i], C[i], B=B[i], y=y[i], rcond=rcond))


		else:
			# C has no batch dimension, use woodbury
			evals, evecs = torch.linalg.eigh(C)
			keep = evals > evals[-1] * rcond
			U = evals[keep].sqrt() * evecs[:, keep]

			out = []
			for i in range(len(N)):
				_out = woodbury(N[i], U)
				if y is not None:
					_out = _out @ y[i]
				if B is not None:
					_out = B @ _out
				out.append(_out)
			out = torch.stack(out)
		    
	return out


def gp_predict(Cs, Cn, train_y, Cp=None, rcond=1e-15):
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

	Returns
	-------
	Tensor
	"""
	if Cp is None:
		Cp = Cs

	# get prediction for all training samples
	return gpr_invert(Cs, Cn, B=Cp, y=train_y, rcond=rcond)


def optimize_kernel(model, Niter=5, opt='LBFGS', thresh=None, loss=None, **kwargs):
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
	kwargs : dict
		Kwargs to pass Optimizer instantiation.
	"""
	# setup optim
	if isinstance(opt, torch.optim.Optimizer):
		# already instantiated
		pass
	elif issubclass(opt, torch.optim.Optimizer):
		# need to instantiate it
		opt = opt(model.parameters(), **kwargs)
	else:
		# need to instantiate it
		opt = getattr(torch.optim, opt)(model.parameters(), **kwargs)

	# marginal log likelihood is the loss function
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

	def closure():
		# Zero gradients from previous iteration
		opt.zero_grad()
		# Output from model
		output = model(model.train_inputs[0])
		# Calc loss and backprop gradients
		loss = -mll(output, model.train_targets).mean()
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
	train_x=None,
	nonstn_prior=None,
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
	train_x : tensor
		Needed for NonStationaryScaleKernel instantiation
	nonstn_prior : Prior object
		Prior for nonstationary scale parameter.
	nonstn_bound : Internval object
		Bounds for nonstationary scale parameter.
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
		if nonstn_prior is not None:
			covar.kernels[0].register_prior(
				'lengthscale_prior',
				nonstn_prior,
				'lengthscale'
				)

	return mean, covar

