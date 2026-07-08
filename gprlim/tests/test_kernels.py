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


def test_default_freq_kernel():
    # composite frequency (delay) kernel. pf_real=True -> a real-valued kernel (a real TwinRBF
    # pitchfork); pf_real=False -> a complex-dtype kernel (the +/- horizon CarrierKernels sum to a
    # real cosine, so it is Hermitian PSD with ~zero imaginary part). Both are valid covariances.
    bl_vec = torch.tensor([[14.6, 0.0, 0.0]], dtype=torch.float64)   # ENU baseline [m]
    nu = torch.linspace(120, 180, 48, dtype=torch.float64)          # frequency [MHz]

    for pf_real in (True, False):
        k = kernels.default_freq_kernel(bl_vec, pf_real=pf_real).double()
        K = k(nu[:, None]).to_dense().detach()
        assert K.shape == (48, 48)
        assert K.is_complex() == (not pf_real)                      # pf_real -> real dtype, else complex
        assert torch.allclose(K, K.conj().transpose(-1, -2))        # Hermitian / symmetric
        assert torch.linalg.eigvalsh(K).min() > -1e-6 * K.diagonal().real.abs().max()   # PSD
        if not pf_real:
            # +/- horizon carriers cancel -> real-valued despite the complex dtype
            assert K.imag.abs().max() < 1e-8 * K.real.abs().max()


def test_reflection_kernel():
    # gain congruence K = g(x1) K_base(x1,x2) conj(g(x2)); +/- tau symmetric (real) by default,
    # one-sided (complex) optional; amp=0 recovers the base; rank + PSD preserved; phase = arg(A).
    nu = torch.linspace(120, 180, 40, dtype=torch.float64)
    base = kernels.ScaleKernel(kernels.SincKernel()).double()
    base.base_kernel.lengthscale = 5.0; base.outputscale = 1.0
    Kb = base(nu[:, None]).to_dense().detach()
    amp, tau, phi = 0.1, 0.2, 0.7

    # amp = 0 -> the base kernel, unchanged and real
    K0 = kernels.ReflectionKernel(base, amp=0.0, tau=tau).double()(nu[:, None]).to_dense().detach()
    assert not K0.is_complex() and torch.allclose(K0, Kb, atol=1e-12)

    # symmetric=True -> real; symmetric=False -> complex; both Hermitian PSD; rank preserved
    Ks = kernels.ReflectionKernel(base, amp=amp, tau=tau, symmetric=True ).double()(nu[:, None]).to_dense().detach()
    Ko = kernels.ReflectionKernel(base, amp=amp, tau=tau, symmetric=False).double()(nu[:, None]).to_dense().detach()
    assert not Ks.is_complex() and Ko.is_complex()
    for K in (Ks, Ko):
        Kc = K.to(torch.cdouble)
        assert torch.allclose(Kc, Kc.conj().T, atol=1e-10)                    # Hermitian
        assert torch.linalg.eigvalsh(Kc).min() > -1e-8                        # PSD
    assert torch.linalg.matrix_rank(Ko) == torch.linalg.matrix_rank(Kb)       # G invertible -> rank kept

    # one-sided == the exact congruence  G K_base G^H  with  g = 1 + amp e^{2 pi i tau nu}
    g = 1 + amp * torch.exp(2j * np.pi * tau * nu)
    assert torch.allclose(Ko.to(torch.cdouble), g[:, None] * Kb.to(torch.cdouble) * g.conj()[None, :], atol=1e-10)

    # phase folds in as A = amp e^{i phi}; phase=0 (default) reproduces the plain kernel
    Kp = kernels.ReflectionKernel(base, amp=amp, tau=tau, phase=phi, symmetric=False).double()(nu[:, None]).to_dense().detach()
    gp = 1 + (amp * np.exp(1j * phi)) * torch.exp(2j * np.pi * tau * nu)
    assert torch.allclose(Kp.to(torch.cdouble), gp[:, None] * Kb.to(torch.cdouble) * gp.conj()[None, :], atol=1e-10)
    Kp0 = kernels.ReflectionKernel(base, amp=amp, tau=tau, phase=0.0, symmetric=False).double()(nu[:, None]).to_dense().detach()
    assert torch.allclose(Kp0, Ko, atol=1e-12)                                # phase=0 == no phase

    # symmetric + phase stays real, = 1 + 2 amp cos(2 pi tau nu + phi)
    Ksp = kernels.ReflectionKernel(base, amp=amp, tau=tau, phase=phi, symmetric=True).double()(nu[:, None]).to_dense().detach()
    gs = 1 + 2 * amp * torch.cos(2 * np.pi * tau * nu + phi)
    assert not Ksp.is_complex() and torch.allclose(Ksp, gs[:, None] * Kb * gs[None, :], atol=1e-10)


def test_delta_kernel():
    # boost a base kernel's delay support by modulating with a lag carrier: symmetric (real,
    # 1 + 2 amp cos, +/- tau) or one-sided (complex, 1 + amp e^{i.}, +tau). amp=0 recovers the base;
    # Hermitian PSD; the delay-space power gains a copy at the boosted delay(s). Base evaluated once.
    Nf = 128
    nu = torch.linspace(120, 180, Nf, dtype=torch.float64)
    dnu = float(nu[1] - nu[0])
    tau_bins = torch.fft.fftshift(torch.fft.fftfreq(Nf, d=dnu))
    base = kernels.ScaleKernel(kernels.SincKernel()).double()
    base.base_kernel.lengthscale = 5.0; base.outputscale = 1.0           # delay band ~ +/-0.1 us
    Kb = base(nu[:, None]).to_dense().detach()
    amp, tau = 0.3, 0.25                                                 # boost at 0.25 us (outside the base band)
    lag = nu[:, None] - nu[None, :]

    # amp = 0 -> base unchanged (both modes)
    for sym in (True, False):
        K0 = kernels.DeltaKernel(base, amp=0.0, tau=tau, symmetric=sym).double()(nu[:, None]).to_dense().detach()
        assert torch.allclose(K0.to(torch.cdouble), Kb.to(torch.cdouble), atol=1e-12)

    # symmetric -> real, = Kb (1 + 2 amp cos); one-sided -> complex, = Kb (1 + amp e^{i.})
    Ks = kernels.DeltaKernel(base, amp=amp, tau=tau, symmetric=True ).double()(nu[:, None]).to_dense().detach()
    Ko = kernels.DeltaKernel(base, amp=amp, tau=tau, symmetric=False).double()(nu[:, None]).to_dense().detach()
    assert not Ks.is_complex() and Ko.is_complex()
    assert torch.allclose(Ks, Kb * (1 + 2 * amp * torch.cos(2 * np.pi * tau * lag)), atol=1e-12)
    assert torch.allclose(Ko, Kb.to(torch.cdouble) * (1 + amp * torch.exp(2j * np.pi * tau * lag)), atol=1e-12)

    # Hermitian PSD (amp >= 0)
    for K in (Ks, Ko):
        Kc = K.to(torch.cdouble)
        assert torch.allclose(Kc, Kc.conj().T, atol=1e-9)
        assert torch.linalg.eigvalsh(Kc).min() > -1e-8

    # delay power: symmetric boosts +tau and -tau equally; one-sided boosts only +tau
    F = torch.fft.fft(torch.eye(Nf, dtype=torch.cdouble), dim=0)
    dpow = lambda K: torch.fft.fftshift((F @ K.to(torch.cdouble) @ F.conj().T).diagonal().real)
    Pb, Ps, Po = dpow(Kb), dpow(Ks), dpow(Ko)
    kp = int((tau_bins - tau).abs().argmin()); km = int((tau_bins + tau).abs().argmin())
    assert Ps[kp] > 10 * Pb[kp] and Ps[km] > 10 * Pb[km]                 # both sidebands boosted
    assert torch.isclose(Ps[kp], Ps[km], rtol=0.1)                       # ... and symmetric
    assert Po[kp] > 10 * Pb[kp] and Po[km] < 2 * Pb[km]                  # one-sided: only +tau

    # wraps (evaluates the base once), not an additive composite
    calls = {'n': 0}; orig = base.base_kernel.forward
    def counted(*a, **k):
        calls['n'] += 1; return orig(*a, **k)
    base.base_kernel.forward = counted
    try:
        kernels.DeltaKernel(base, amp=amp, tau=tau).double()(nu[:, None]).to_dense()
    finally:
        base.base_kernel.forward = orig
    assert calls['n'] == 1


def test_band_limit_kernel():
    # project a base covariance onto |tau| < tau_c (DPSS): Hermitian PSD, low rank ~ the Shannon
    # number, out-of-band power crushed; reject is the complement; num_modes bounds the rank.
    Nf = 64
    nu = torch.linspace(120, 180, Nf, dtype=torch.float64)
    dnu = float(nu[1] - nu[0])
    tau = torch.fft.fftshift(torch.fft.fftfreq(Nf, d=dnu))                    # delay bins [us]
    tau_c = 0.1

    # base with power both in-band (RBF main lobe) and out-of-band (a carrier at 0.3 us)
    main = kernels.ScaleKernel(kernels.RBFKernel()).double(); main.base_kernel.lengthscale = 3.0; main.outputscale = 1.0
    horn = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()).double(), tau=0.30).double()
    horn.base_kernel.base_kernel.lengthscale = 3.0; horn.base_kernel.outputscale = 0.5
    base = main + horn
    Kb = base(nu[:, None]).to_dense().detach()
    Kbl = kernels.BandLimitKernel(base, tau_c=tau_c).double()(nu[:, None]).to_dense().detach()

    # delay power = diag(F K F^H); out-of-band fraction = power at |tau| > tau_c
    F = torch.fft.fft(torch.eye(Nf, dtype=torch.cdouble), dim=0)
    def oob(K):
        P = torch.fft.fftshift((F @ K.to(torch.cdouble) @ F.conj().T).diagonal().real)
        return float(P[tau.abs() > tau_c].sum() / P.sum())

    # Hermitian, PSD, low rank ~ the Shannon number 2 tau_c BW
    Kc = Kbl.to(torch.cdouble)
    assert torch.allclose(Kc, Kc.conj().T, atol=1e-9)
    assert torch.linalg.eigvalsh(Kc).min() > -1e-8
    shannon = round(2 * tau_c * (Nf * dnu))
    assert 1 <= torch.linalg.matrix_rank(Kbl).item() <= shannon + 3

    # out-of-band power is crushed relative to the base
    assert oob(Kb) > 0.15 and oob(Kbl) < 0.05

    # reject (I - B) keeps the out-of-band, so its in-band fraction is tiny
    rej = kernels.BandLimitKernel(base, tau_c=tau_c, reject=True).double()(nu[:, None]).to_dense().detach()
    assert (1 - oob(rej)) < 0.05

    # num_modes bounds the projector (hence the covariance) rank
    kn = kernels.BandLimitKernel(base, tau_c=tau_c, num_modes=6).double()(nu[:, None]).to_dense().detach()
    assert torch.linalg.matrix_rank(kn).item() <= 6