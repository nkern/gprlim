import os
import random
import warnings

import numpy as np
import pytest
from pyuvdata import UVData

from gprlim import kernels
import torch

random.seed(0)
import copy


def test_sinc_kernel():
	# setup sinc kernel with scale
	train_x = torch.linspace(100, 120, 32)[:, None]
	kern = kernels.ScaleKernel(kernels.SincKernel())
	kern.outputscale = 1.0
	kern.base_kernel.lengthscale = 2.0

	# draw from posterior
	cov = kern(train_x).to_dense().detach()

	assert torch.isclose(cov.diag(), torch.tensor(1.0)).all()


def test_twin_rbf():
	# Gaussian-windowed cosine cos(2 pi tau Delta): the FT of the covariance has two
	# peaks at +/- tau (tau is the delay, i.e. ordinary frequency)
	tau, ell = 1.0, 2.0          # ell * tau >= 1 so the two peaks are distinct
	kern = kernels.TwinRBFKernel().double()
	kern.tau = tau
	kern.lengthscale = ell

	with torch.no_grad():
		# normalized to k(0) = 1
		k0 = kern(torch.zeros(1, 1, dtype=torch.float64)).to_dense().squeeze()
		assert torch.isclose(k0, torch.tensor(1.0, dtype=torch.float64))

		# FT of the covariance peaks at +/- tau
		M, L = 2048, 40.0
		lags = torch.linspace(-L, L, M, dtype=torch.float64)
		klag = kern(lags[:, None], torch.zeros(1, 1, dtype=torch.float64)).to_dense().squeeze()
		freq = np.fft.fftshift(np.fft.fftfreq(M, d=float(lags[1] - lags[0])))
		psd = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(klag.numpy()))))
		peaks = np.sort(freq[np.argsort(psd)[-2:]])
		assert np.allclose(peaks, [-tau, tau], atol=0.05)

		# positive semi-definite
		x = torch.linspace(0, 20, 100, dtype=torch.float64)
		assert torch.linalg.eigvalsh(kern(x[:, None]).to_dense()).min() > -1e-8


def test_gauss_sinc_kernel():
	# wraps gauss_sinc_cov: stationary, normalized, PSD, exact cross-covariance
	kern = kernels.GaussSincKernel(1.0, 2.0)
	x = torch.linspace(0, 8, 12, dtype=torch.float64)

	with torch.no_grad():
		Kxx = kern(x[:, None]).to_dense()
		assert torch.allclose(Kxx, Kxx.T)
		assert torch.allclose(Kxx.diag(), torch.ones(12, dtype=torch.float64))
		assert torch.linalg.eigvalsh(Kxx).min() > -1e-8

		# matches the underlying function, with the right cross-cov orientation
		xs = torch.linspace(0.4, 7.6, 7, dtype=torch.float64)
		Kc = kern(xs[:, None], x[:, None]).to_dense()
		assert Kc.shape == (7, 12)
		assert torch.allclose(Kc, kernels.gauss_sinc_cov(x, 1.0, 2.0, x2=xs))


def test_carrier_kernel():
	# CarrierKernel turns a real base kernel into a complex Hermitian PSD kernel whose
	# band is shifted to the carrier `tau`; tau=0 recovers the real base kernel.
	base = kernels.ScaleKernel(kernels.SincKernel()).double()
	base.base_kernel.lengthscale = 1.0
	x = torch.linspace(0, 20, 64, dtype=torch.float64)[:, None]

	with torch.no_grad():
		# tau = 0 -> identical to the real base kernel
		k0 = kernels.CarrierKernel(copy.deepcopy(base), tau=0.0).double()
		assert torch.allclose(k0(x).to_dense().real, base(x).to_dense())
		assert k0(x).to_dense().imag.abs().max() < 1e-12

		# tau != 0 -> complex Hermitian, PSD, real diagonal, equal to base * carrier
		tau = 0.3
		k = kernels.CarrierKernel(copy.deepcopy(base), tau=tau).double()
		K = k(x).to_dense()
		assert K.is_complex() and K.imag.abs().max() > 0
		assert torch.allclose(K, K.conj().T)                       # Hermitian
		assert torch.linalg.eigvalsh(K).min() > -1e-8              # PSD
		assert K.diagonal().imag.abs().max() < 1e-12              # real diagonal

		lag = x[:, 0:1] - x[:, 0:1].T
		expected = base(x).to_dense() * torch.exp(2j * np.pi * tau * lag)
		assert torch.allclose(K, expected)


def test_carrier_kernel_wirtinger_autograd():
	# real hyperparameters -> complex covariance -> real loss: PyTorch's Wirtinger
	# autograd yields correct real gradients (match to finite difference).
	torch.manual_seed(0)
	base = kernels.ScaleKernel(kernels.SincKernel()).double()
	base.base_kernel.lengthscale = 0.5
	k = kernels.CarrierKernel(base, tau=-1.0).double()
	x = torch.linspace(0, 10, 50, dtype=torch.float64)[:, None]
	y = torch.randn(50, dtype=torch.cdouble)

	def loss_fn():
		A = k(x).to_dense() + 0.1 * torch.eye(50, dtype=torch.cdouble)
		L = torch.linalg.cholesky(A)
		alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
		return (y.conj() * alpha).real.sum() + 2 * torch.log(torch.diagonal(L).real).sum()

	k.zero_grad()
	loss = loss_fn()
	assert not loss.is_complex()
	loss.backward()
	g = k.raw_tau.grad
	assert g is not None and not g.is_complex() and torch.isfinite(g).all()

	# finite-difference on raw_tau, perturbing .data in float64 (the `tau` setter would
	# round-trip a python float through the default dtype and spoil a 1e-6 step)
	eps = 1e-6
	with torch.no_grad():
		k.raw_tau.data += eps; lp = float(loss_fn())
		k.raw_tau.data -= 2 * eps; lm = float(loss_fn())
		k.raw_tau.data += eps
	fd = (lp - lm) / (2 * eps)
	assert abs(float(g) - fd) < 1e-4 * abs(fd)