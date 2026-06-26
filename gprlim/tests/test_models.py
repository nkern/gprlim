import os

import numpy as np
import pytest
import torch
import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import NormalPrior, GammaPrior
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from gprlim import kernels
from gprlim.models import (
    gpr_invert,
    cholesky_batched,
    batched_log_prob,
    _sum_log_priors,
    mean_center,
    posterior_mean_1d,
    posterior_mean_2d,
    inpaint_1d,
    inpaint_2d,
    prior_draws_1d,
    prior_draws_2d,
    posterior_draws_1d,
    posterior_draws_2d,
    fit_axis_kernel,
    fit_kernel,
)
from gprlim.solvers import shrink


# The library is functional (no GPModel); this minimal ExactGP exists ONLY here, as the
# gpytorch reference for cross-checking batched_log_prob / fit_kernel in the real, small-N
# regime where ExactMarginalLogLikelihood is exact (it is real-only and, for large N,
# stochastic -- which is why the library uses the batched solvers instead).
class _RefGP(ExactGP):
    def __init__(self, x, y, noise, mean, covar):
        super().__init__(x, y, FixedNoiseGaussianLikelihood(noise))
        self.mean, self.covar = mean, covar

    def forward(self, x):
        return MultivariateNormal(self.mean(x), self.covar(x))


def _ref_mll(x, y, noise, mean, covar):
    """A trained gpytorch ExactGP and its ExactMarginalLogLikelihood (the reference loss);
    shares the given ``mean``/``covar`` modules so grads/fits compare directly."""
    model = _RefGP(x, y, noise, mean, covar)
    model.train()
    model.likelihood.train()
    return model, gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)


def _legacy_pinv(C, N, B=None, y=None, rcond=1e-12):
    """Reference: per-batch pinv solve of B @ (C + diag(N_b))^-1 @ y_b."""
    out = []
    for i in range(len(N)):
        o = torch.linalg.pinv(C + N[i].diag(), hermitian=True, rcond=rcond)
        if y is not None:
            o = o @ y[i]
        if B is not None:
            o = B @ o
        out.append(o)
    return torch.stack(out)


def test_gpr_invert_methods_match_pinv():
    """Both batched solvers reproduce a per-batch pinv reference across all four
    (B, y) shape combinations, and chunking is exact."""
    torch.manual_seed(0)
    n, b, m = 48, 16, 30
    F = torch.randn(n, n, dtype=torch.float64)
    C = F @ F.T / n + 0.1 * torch.eye(n, dtype=torch.float64)   # strictly PD
    N = 0.1 + torch.rand(b, n, dtype=torch.float64)             # heteroscedastic, > 0
    y = torch.randn(b, n, dtype=torch.float64)
    B = torch.randn(m, n, dtype=torch.float64)

    cases = {
        "both": dict(B=B, y=y),       # -> (b, m)
        "y_only": dict(y=y),          # -> (b, n)
        "B_only": dict(B=B),          # -> (b, m, n)
        "neither": dict(),            # -> (b, n, n)
    }
    for name, kw in cases.items():
        ref = _legacy_pinv(C, N, **kw)
        for method in ("woodbury", "cholesky"):
            out = gpr_invert(C, N, method=method, **kw)
            chunked = gpr_invert(C, N, method=method, chunk=5, **kw)
            assert out.shape == ref.shape, (name, method)
            assert (out - ref).abs().max() < 1e-9, (name, method)
            assert (out - chunked).abs().max() < 1e-12, (name, method, "chunk")


def test_cholesky_batched_nonpd_fallback():
    """cholesky_batched falls back to a truncated-eigh (pinv-equivalent) solve when
    (C + diag(N_b)) is singular, matching the pinv reference."""
    torch.manual_seed(0)
    n, b = 30, 10

    # PD covariance, then zero out two rows/cols so those pixels carry no signal
    core = torch.randn(n, n, dtype=torch.float64)
    C = core @ core.T / n + 0.5 * torch.eye(n, dtype=torch.float64)
    for j in (5, 17):
        C[j, :] = 0.0
        C[:, j] = 0.0

    N = 0.1 + torch.rand(b, n, dtype=torch.float64)
    # zero the noise on those pixels for half the batch -> singular C + diag(N)
    N[: b // 2, 5] = 0.0
    N[: b // 2, 17] = 0.0
    y = torch.randn(b, n, dtype=torch.float64)

    # the construction must genuinely produce non-PD elements, or the fallback
    # path would never be exercised
    A = C.unsqueeze(0) + torch.diag_embed(N)
    _, info = torch.linalg.cholesky_ex(A)
    assert (info > 0).any(), "test did not construct a non-PD element"

    out = cholesky_batched(C, N, y=y, rcond=1e-12)
    ref = _legacy_pinv(C, N, y=y, rcond=1e-12)
    assert torch.isfinite(out).all()
    assert (out - ref).abs().max() < 1e-8


def _mean_covar_with_priors():
    """A constant mean + scaled-RBF covariance with priors on the constant, lengthscale,
    and outputscale (priors registered, then cast to double like the old model.double())."""
    mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
    covar.base_kernel.register_prior('ls', GammaPrior(2.0, 1.0), 'lengthscale')
    covar.register_prior('os', GammaPrior(2.0, 1.0), 'outputscale')
    mean.register_prior('m', NormalPrior(0.0, 1.0), 'constant')
    mean.double()
    covar.double()
    return mean, covar


def test_batched_log_prob_matches_gpytorch_mll():
    """batched_log_prob (+ priors, mean-centering) reproduces gpytorch's
    ExactMarginalLogLikelihood loss and gradients (real, small-N reference)."""
    torch.manual_seed(0)
    n, b = 25, 5
    x = torch.linspace(0, 4, n, dtype=torch.float64)[None, :, None]
    y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    mean, covar = _mean_covar_with_priors()
    params = dict(list(mean.named_parameters()) + list(covar.named_parameters()))

    # gpytorch reference (shares the same mean/covar objects, so grads compare directly)
    model, mll = _ref_mll(x, y, noise, mean, covar)
    model.zero_grad()
    ref = -mll(model(model.train_inputs[0]), model.train_targets).mean()
    ref.backward()
    gref = {k: p.grad.clone() for k, p in params.items()}

    for method in ('woodbury', 'cholesky'):
        for p in params.values():
            p.grad = None
        lp = batched_log_prob(covar(x).to_dense(), noise, y - mean(x), method=method)
        loss = -((lp + _sum_log_priors(covar) + _sum_log_priors(mean)) / n).mean()
        loss.backward()
        assert abs(loss.item() - ref.item()) < 1e-9, method
        for k, p in params.items():
            assert (p.grad - gref[k]).abs().max() < 1e-8, (method, k)


def test_fit_kernel_matches_gpytorch():
    """fit_kernel (batched MLL) tracks a gpytorch ExactMarginalLogLikelihood fit to the
    same optimum (the dense gpytorch path, kept here only as a reference)."""
    torch.manual_seed(0)
    n, b = 30, 6
    x = torch.linspace(0, 5, n, dtype=torch.float64)[None, :, None]
    y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    def fit_ref():
        torch.manual_seed(1)
        mean, covar = _mean_covar_with_priors()
        model, mll = _ref_mll(x, y, noise, mean, covar)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(20):
            opt.zero_grad()
            loss = -mll(model(model.train_inputs[0]), model.train_targets).mean()
            loss.backward()
            opt.step()
        return mean, covar

    def fit_batched(method):
        torch.manual_seed(1)
        mean, covar = _mean_covar_with_priors()
        fit_kernel(covar, x, y, noise, mu=mean, Niter=20, opt='Adam', method=method, lr=0.1)
        return mean, covar

    def hyper(mc):
        mean, covar = mc
        return torch.tensor([float(covar.base_kernel.lengthscale.detach()),
                             float(covar.outputscale.detach()), float(mean.constant.detach())])

    base = hyper(fit_ref())
    for method in ('woodbury', 'cholesky'):
        assert (hyper(fit_batched(method)) - base).abs().max() < 1e-6, method


def test_priors_active_and_recovered_in_batched():
    """A strong lengthscale prior visibly shifts the fit, and the batched paths
    recover the same prior-influenced optimum as gpytorch's ExactMLL."""
    torch.manual_seed(0)
    n, b = 40, 8
    xv = torch.linspace(0, 10, n, dtype=torch.float64)

    # draw data from a GP with lengthscale 2 -> the data alone wants ls ~ 2
    Kt = ScaleKernel(RBFKernel())
    Kt.outputscale = 1.0
    Kt.base_kernel.lengthscale = 2.0
    Kmat = Kt(xv[:, None]).to_dense().detach() + 1e-6 * torch.eye(n, dtype=torch.float64)
    y = torch.randn(b, n, dtype=torch.float64) @ torch.linalg.cholesky(Kmat).T
    noise = 0.01 * torch.ones(b, n, dtype=torch.float64)
    x = xv[None, :, None]

    def fit(prior=False, method=None):
        torch.manual_seed(1)
        mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
        covar.base_kernel.lengthscale = 1.0
        if prior:                                     # tight prior pulling ls -> 0.5
            covar.base_kernel.register_prior('ls', NormalPrior(0.5, 0.1), 'lengthscale')
        mean.double()
        covar.double()
        if method is None:                            # gpytorch reference fit
            model, mll = _ref_mll(x, y, noise, mean, covar)
            opt = torch.optim.Adam(model.parameters(), lr=0.05)
            for _ in range(150):
                opt.zero_grad()
                loss = -mll(model(model.train_inputs[0]), model.train_targets).mean()
                loss.backward()
                opt.step()
        else:                                         # batched marginal-likelihood fit
            fit_kernel(covar, x, y, noise, mu=mean, Niter=150, opt='Adam', method=method, lr=0.05)
        return float(covar.base_kernel.lengthscale.detach())

    ls_noprior = fit(prior=False, method=None)
    ls_gpt = fit(prior=True, method=None)
    ls_wood = fit(prior=True, method='woodbury')
    ls_chol = fit(prior=True, method='cholesky')

    # the prior is genuinely active: it pulls ls well away from the data-only fit,
    # toward the prior mean (0.5)
    assert ls_noprior > 1.5                                       # data alone -> ls ~ 2
    assert ls_gpt < 1.0                                           # prior pulls it down
    assert abs(ls_gpt - 0.5) < abs(ls_noprior - 0.5)             # moved toward prior mean
    # the batched paths recover gpytorch's prior-influenced optimum
    assert abs(ls_wood - ls_gpt) < 1e-5
    assert abs(ls_chol - ls_gpt) < 1e-5


def test_mean_center():
    """mean_center: inverse-noise-weighted mean over a single axis or a tuple of axes
    (keepdim), for complex data."""
    torch.manual_seed(0)
    z = torch.randn(3, 4, 5, dtype=torch.cdouble)
    w = 0.1 + torch.rand(3, 4, 5, dtype=torch.float64)            # noise variance (inverse weights)
    for dim in (-1, (-2, -1), (0, 2)):
        mc = mean_center(z, w, dim=dim)
        ref = (z / w).sum(dim, keepdim=True) / (1.0 / w).sum(dim, keepdim=True)
        assert mc.shape == ref.shape
        assert torch.allclose(mc, ref, atol=1e-12)


def test_posterior_mean_1d():
    """posterior_mean_1d matches a dense per-row Wiener solve across: complex data + real
    kernel (auto real/imag split), complex data + complex kernel (direct), real data, a
    non-trailing `dim`, a prediction grid (pred_x), a mean function (mu), and detrend
    (inverse-noise-weighted mean subtracted before the solve, added back)."""
    torch.manual_seed(0)
    Nx, B = 9, 4
    x = torch.linspace(0, 10, Nx, dtype=torch.float64)
    const = 0.5
    mean = lambda xx: torch.full(xx.shape[:-1], const, dtype=torch.float64)

    def dense_ref(Cs, Cp, yc, noise, mu_pred=0.0):
        rows = [Cp @ torch.linalg.solve(Cs + torch.diag(noise[i].to(Cs.dtype)), yc[i].to(Cs.dtype))
                for i in range(yc.shape[0])]
        return torch.stack(rows) + mu_pred

    kr = kernels.ScaleKernel(kernels.RBFKernel()).double(); kr.base_kernel.lengthscale = 2.0
    kc = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.1).double()
    kc.base_kernel.base_kernel.lengthscale = 2.0
    K = kr(x[:, None]).to_dense().detach()
    Kc = kc(x[:, None]).to_dense().detach()

    noise = 0.1 + torch.rand(B, Nx, dtype=torch.float64)
    y = torch.randn(B, Nx, dtype=torch.cdouble)
    yr = torch.randn(B, Nx, dtype=torch.float64)

    # complex data + real kernel -> auto-split, matches the promoted-complex dense solve
    out = posterior_mean_1d(kr, x, y, noise)
    assert out.dtype == torch.cdouble
    assert torch.allclose(out, dense_ref(K.to(torch.cdouble), K.to(torch.cdouble), y, noise), atol=1e-9)

    # complex data + complex kernel -> solved directly
    out_c = posterior_mean_1d(kc, x, y, noise)
    assert torch.allclose(out_c, dense_ref(Kc, Kc, y, noise), atol=1e-9)

    # real data + real kernel
    out_r = posterior_mean_1d(kr, x, yr, noise)
    assert torch.allclose(out_r, dense_ref(K, K, yr, noise), atol=1e-9)

    # dim != -1: sample axis in the middle of a 3-D y; matches looping the 2-D solve
    y3 = torch.randn(B, Nx, 5, dtype=torch.cdouble)
    n3 = 0.1 + torch.rand(B, Nx, 5, dtype=torch.float64)
    out3 = posterior_mean_1d(kr, x, y3, n3, dim=1)
    assert out3.shape == y3.shape
    for k in range(5):
        assert torch.allclose(out3[:, :, k], posterior_mean_1d(kr, x, y3[:, :, k], n3[:, :, k]), atol=1e-12)

    # pred_x at new points + a mean function (mu)
    pred_x = torch.linspace(0, 10, 6, dtype=torch.float64)
    outp = posterior_mean_1d(kr, x, yr, noise, pred_x=pred_x, mu=mean)
    Cp = kr(pred_x[:, None], x[:, None]).to_dense().detach()
    assert outp.shape == (B, 6)
    assert torch.allclose(outp, dense_ref(K, Cp, yr - const, noise, const), atol=1e-9)

    # detrend: subtract the per-row inverse-noise-weighted mean before the solve, add it back
    out_d = posterior_mean_1d(kr, x, y, noise, detrend=True)
    Kc2 = K.to(torch.cdouble)
    ref_d = torch.empty_like(y)
    for i in range(B):
        t = (y[i] / noise[i]).sum() / (1.0 / noise[i]).sum()
        ref_d[i] = Kc2 @ torch.linalg.solve(Kc2 + torch.diag(noise[i].to(torch.cdouble)), y[i] - t) + t
    assert torch.allclose(out_d, ref_d, atol=1e-9)

    # 'cg' is 2D-only; the 1D dense covariance is small -> NotImplementedError
    import pytest
    with pytest.raises(NotImplementedError):
        posterior_mean_1d(kr, x, y, noise, method='cg')


def test_posterior_mean_2d():
    """posterior_mean_2d (full 2D posterior mean on the grid, no flags) matches a dense
    per-baseline Wiener solve for woodbury/cg/cholesky, complex & real covariances, the
    C1_rcond time-shrink, and a 2D mean function (mu) + detrend; pred_x raises NotImplementedError."""
    torch.manual_seed(0)
    Nb, N1, N2 = 3, 6, 7
    x1 = torch.linspace(0, 50, N1, dtype=torch.float64)
    x2 = torch.linspace(120, 180, N2, dtype=torch.float64)
    k1c = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    k1c.base_kernel.base_kernel.lengthscale = 8.0
    k1r = kernels.ScaleKernel(kernels.RBFKernel()).double(); k1r.base_kernel.lengthscale = 15.0
    k2 = kernels.ScaleKernel(kernels.SincKernel()).double(); k2.base_kernel.lengthscale = 2.0

    y = torch.randn(Nb, N1, N2, dtype=torch.cdouble)
    noise = 0.1 + torch.rand(Nb, N1, N2, dtype=torch.float64)
    C1c = k1c(x1[:, None]).to_dense().detach()
    C1r = k1r(x1[:, None]).to_dense().detach()
    C2 = k2(x2[:, None]).to_dense().detach()

    def dense_ref(C1):                                   # full mean, promoted to complex
        Ks = torch.kron(C1, C2).to(torch.cdouble)
        out = [(Ks @ torch.linalg.solve(Ks + torch.diag(noise[b].reshape(-1).to(torch.cdouble)),
                                        y[b].reshape(-1))).reshape(N1, N2) for b in range(Nb)]
        return torch.stack(out)

    for method in ('woodbury', 'cg', 'cholesky'):
        # complex covariance -> solved directly
        outc = posterior_mean_2d(k1c, k2, x1, x2, y, noise, method=method,
                                 rcond=1e-12, cg_tol=1e-11, cg_max_iter=3000)
        assert outc.dtype == torch.cdouble and outc.shape == y.shape
        assert torch.allclose(outc, dense_ref(C1c), atol=1e-6), method
        # real covariance + complex data -> real/imag split
        outr = posterior_mean_2d(k1r, k2, x1, x2, y, noise, method=method,
                                 rcond=1e-12, cg_tol=1e-11, cg_max_iter=3000)
        assert torch.allclose(outr, dense_ref(C1r), atol=1e-6), ('split', method)

    # C1_rcond shrink: matches a dense solve with the shrunk outer factor
    out_e = posterior_mean_2d(k1c, k2, x1, x2, y, noise, C1_rcond=1e-2, method='cholesky')
    assert torch.allclose(out_e, dense_ref(shrink(C1c, 1e-2)), atol=1e-6)

    # mu (2D mean fn) + detrend: subtract both before the solve, add both back to the prediction
    muf = lambda a, b: torch.full((a.shape[0], b.shape[0]), 0.3 + 0.2j, dtype=torch.cdouble)
    out_m = posterior_mean_2d(k1c, k2, x1, x2, y, noise, mu=muf, detrend=True, method='cholesky')
    mu_x = muf(x1[:, None], x2[:, None])
    Ks = torch.kron(C1c.to(torch.cdouble), C2.to(torch.cdouble))
    ref_m = torch.empty_like(y)
    for b in range(Nb):
        yc = y[b] - mu_x
        t = (yc / noise[b]).sum() / (1.0 / noise[b]).sum()
        m = (Ks @ torch.linalg.solve(Ks + torch.diag(noise[b].reshape(-1).to(torch.cdouble)),
                                     (yc - t).reshape(-1))).reshape(N1, N2)
        ref_m[b] = m + mu_x + t
    assert torch.allclose(out_m, ref_m, atol=1e-6)

    # dims=: the 2D axes need not be the last two -- remap via dims and round-trip
    yp, np_ = y.permute(1, 2, 0).contiguous(), noise.permute(1, 2, 0).contiguous()   # (N1,N2,Nb)
    out_p = posterior_mean_2d(k1c, k2, x1, x2, yp, np_, dims=(0, 1), method='cholesky')
    assert out_p.shape == (N1, N2, Nb)
    assert torch.allclose(out_p, dense_ref(C1c).permute(1, 2, 0), atol=1e-6)

    # prediction at new points not yet supported
    import pytest
    with pytest.raises(NotImplementedError):
        posterior_mean_2d(k1c, k2, x1, x2, y, noise, pred_x1=x1)


def test_posterior_mean_broadcasts_noise():
    """Shared / lower-rank noise broadcasts against y (the documented contract): 1d & 2d with
    noise (1, ...) or (Nx,) match the fully-expanded noise -- regression for the Nbls>1
    batch-mismatch bug (nf/yf had different leading sizes)."""
    torch.manual_seed(0)
    Nbls, Nt, Nf = 6, 5, 9
    nu = torch.linspace(120, 180, Nf, dtype=torch.float64)
    t = torch.linspace(0, 200, Nt, dtype=torch.float64)
    kf = kernels.ScaleKernel(kernels.SincKernel()).double(); kf.base_kernel.lengthscale = 3.0
    kt = kernels.ScaleKernel(kernels.RBFKernel()).double(); kt.base_kernel.lengthscale = 40.0
    y = torch.randn(Nbls, Nt, Nf, dtype=torch.cdouble)

    # 1D (freq): shared (1, Nt, Nf) and lower-rank (Nf,) both broadcast over (Nbls, Nt)
    nz1 = 0.05 + torch.rand(1, Nt, Nf, dtype=torch.float64)
    for method in ('cholesky', 'woodbury'):
        out = posterior_mean_1d(kf, nu, y, nz1, method=method)
        ref = posterior_mean_1d(kf, nu, y, nz1.expand(Nbls, -1, -1).contiguous(), method=method)
        assert torch.allclose(out, ref, atol=1e-9), method
    nzrow = 0.05 + torch.rand(Nf, dtype=torch.float64)
    assert torch.allclose(posterior_mean_1d(kf, nu, y, nzrow),
                          posterior_mean_1d(kf, nu, y, nzrow.expand(Nbls, Nt, Nf).contiguous()),
                          atol=1e-9)

    # 2D (joint): shared (1, Nt, Nf) broadcasts over the baseline axis, all methods
    nz2 = 0.05 + torch.rand(1, Nt, Nf, dtype=torch.float64)
    for method in ('cholesky', 'woodbury', 'cg'):
        out = posterior_mean_2d(kt, kf, t, nu, y, nz2, method=method, cg_tol=1e-11, cg_max_iter=3000)
        ref = posterior_mean_2d(kt, kf, t, nu, y, nz2.expand(Nbls, -1, -1).contiguous(),
                                method=method, cg_tol=1e-11, cg_max_iter=3000)
        assert torch.allclose(out, ref, atol=1e-7), method


def test_cg_tol_flag_invariant():
    """cg_tol is data-independent: with the inverse-variance-weighted CG stop test, a fixed
    cg_tol reaches the dense Wiener mean to the same accuracy whether flagged pixels carry
    flag_var=1e6 or 1e12 -- regression for the flag-var/noise-sensitive convergence."""
    torch.manual_seed(0)
    Nt, Nf = 10, 16
    t = torch.linspace(0, 120, Nt, dtype=torch.float64)
    nu = torch.linspace(120, 180, Nf, dtype=torch.float64)
    kt = kernels.ScaleKernel(kernels.RBFKernel()).double(); kt.base_kernel.lengthscale = 40.0
    kf = kernels.ScaleKernel(kernels.SincKernel()).double(); kf.base_kernel.lengthscale = 3.0
    Ks = torch.kron(kt(t[:, None]).to_dense().detach(),
                    kf(nu[:, None]).to_dense().detach()).to(torch.cdouble)

    y = torch.randn(1, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(1, Nt, Nf, dtype=bool); flags[0, :, 7] = True; flags[0, 4, :] = True
    y[flags] = 50.0                                          # RFI-like garbage at flagged pixels
    base = 0.01 * torch.ones(1, Nt, Nf, dtype=torch.float64)
    for flag_var in (1e6, 1e12):
        nz = base.clone(); nz[flags] = flag_var
        out = posterior_mean_2d(kt, kf, t, nu, y, nz, method='cg', cg_tol=1e-6, cg_max_iter=4000)
        A = Ks + torch.diag(nz.reshape(-1).to(torch.cdouble))
        ref = (Ks @ torch.linalg.solve(A, y.reshape(-1))).reshape(Nt, Nf)
        assert (out[0] - ref).abs().max() < 1e-6, flag_var


def test_inpaint_1d_2d():
    """inpaint_1d/2d return (inp, mdl): inp fills flagged pixels with the posterior mean and
    leaves good pixels untouched; mdl is the full posterior mean (thin wrappers over
    posterior_mean_1d/2d)."""
    torch.manual_seed(0)
    # --- 1D ---
    Nx, B = 9, 4
    x = torch.linspace(0, 10, Nx, dtype=torch.float64)
    kr = kernels.ScaleKernel(kernels.RBFKernel()).double(); kr.base_kernel.lengthscale = 2.0
    y = torch.randn(B, Nx, dtype=torch.cdouble)
    flags = torch.zeros(B, Nx, dtype=bool); flags[:, 4] = True; flags[0, 1] = True
    noise = 0.01 * torch.ones(B, Nx, dtype=torch.float64); noise[flags] = 1e12
    inp, mdl = inpaint_1d(kr, x, y, noise, flags)
    ref = posterior_mean_1d(kr, x, y, noise)
    assert inp.dtype == torch.cdouble
    assert torch.allclose(mdl, ref)                         # mdl is the full posterior mean
    assert torch.equal(inp[~flags], y[~flags])              # good untouched
    assert torch.allclose(inp[flags], mdl[flags])           # flagged = posterior mean

    # --- 2D ---
    Nb, N1, N2 = 2, 6, 8
    x1 = torch.linspace(0, 50, N1, dtype=torch.float64)
    x2 = torch.linspace(120, 180, N2, dtype=torch.float64)
    k1 = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    k1.base_kernel.base_kernel.lengthscale = 8.0
    k2 = kernels.ScaleKernel(kernels.SincKernel()).double(); k2.base_kernel.lengthscale = 2.0
    y2 = torch.randn(Nb, N1, N2, dtype=torch.cdouble)
    fl2 = torch.zeros(Nb, N1, N2, dtype=bool); fl2[:, :, 3] = True; fl2[0, 2, 1] = True
    nz2 = 0.01 * torch.ones(Nb, N1, N2, dtype=torch.float64); nz2[fl2] = 1e12
    kw = dict(method='cg', cg_tol=1e-11, cg_max_iter=3000)
    inp2, mdl2 = inpaint_2d(k1, k2, x1, x2, y2, nz2, fl2, **kw)
    ref2 = posterior_mean_2d(k1, k2, x1, x2, y2, nz2, **kw)
    assert torch.allclose(mdl2, ref2)
    assert torch.equal(inp2[~fl2], y2[~fl2])
    assert torch.allclose(inp2[fl2], mdl2[fl2])


def test_prior_draws_1d_2d():
    """prior_draws_1d/2d sample the GP prior: the empirical covariance E[f f^H] matches the
    kernel covariance, for real and complex (circular) kernels, in 1D and 2D."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    S = 200_000

    def emp_cov(f):                                  # f: (S, n) -> (n, n) ~ E[f f^H]
        return torch.einsum('si,sj->ij', f, f.conj()) / f.shape[0]

    def rel_err(emp, K):
        return float((emp - K).abs().max() / K.abs().max())

    x = torch.linspace(0, 10, 6, dtype=torch.float64)
    kr = kernels.ScaleKernel(kernels.RBFKernel()).double(); kr.base_kernel.lengthscale = 2.5
    kc = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.1).double()
    kc.base_kernel.base_kernel.lengthscale = 2.5

    # 1D real -> real draws, E[f f^T] == K
    fr = prior_draws_1d(kr, x, size=S, generator=g)
    assert fr.shape == (S, 6) and not fr.is_complex()
    assert rel_err(emp_cov(fr), kr(x[:, None]).to_dense().detach()) < 0.05

    # 1D complex (Carrier) -> circular complex draws, E[f f^H] == K
    fc = prior_draws_1d(kc, x, size=S, generator=g)
    assert fc.is_complex()
    assert rel_err(emp_cov(fc), kc(x[:, None]).to_dense().detach()) < 0.05

    # 2D: E[f f^H] == C1 (x) C2 (complex C1, real C2)
    x1 = torch.linspace(0, 10, 4, dtype=torch.float64)
    x2 = torch.linspace(0, 5, 5, dtype=torch.float64)
    k2 = kernels.ScaleKernel(kernels.SincKernel()).double(); k2.base_kernel.lengthscale = 1.5
    f2 = prior_draws_2d(kc, k2, x1, x2, size=S, generator=g)
    assert f2.shape == (S, 4, 5)
    C1 = kc(x1[:, None]).to_dense().detach(); C2 = k2(x2[:, None]).to_dense().detach()
    Kron = torch.kron(C1, C2.to(C1.dtype))
    assert rel_err(emp_cov(f2.reshape(S, -1)), Kron) < 0.06

    # mu (2D mean fn) shifts the mean, not the covariance
    muf = lambda a, b: torch.full((a.shape[0], b.shape[0]), 1.0 - 0.5j, dtype=torch.cdouble)
    fmu = prior_draws_2d(kc, k2, x1, x2, mu=muf, size=S, generator=g)
    assert torch.allclose(fmu.mean(0), muf(x1[:, None], x2[:, None]), atol=0.05)
    assert rel_err(emp_cov((fmu - fmu.mean(0)).reshape(S, -1)), Kron) < 0.06


def test_posterior_draws_1d_moments():
    """Matheron posterior draws reproduce the analytic posterior moments. Mean -> posterior
    mean; covariance -> K - K(K+Sigma)^-1 K. Validated for a complex (circular) kernel and a
    real kernel on complex data (real/imag drawn with equal posterior variance)."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    S, Nx = 300_000, 5
    x = torch.linspace(0, 10, Nx, dtype=torch.float64)
    kc = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.1).double()
    kc.base_kernel.base_kernel.lengthscale = 3.0
    kr = kernels.ScaleKernel(kernels.RBFKernel()).double()
    kr.base_kernel.lengthscale = 3.0

    y = torch.randn(1, Nx, dtype=torch.cdouble)
    noise = 0.2 + 0.3 * torch.rand(1, Nx, dtype=torch.float64)

    def analytic(K):                                     # complex y, Sigma = diag(noise)
        Kc = K.to(torch.cdouble)
        A = Kc + torch.diag(noise[0].to(torch.cdouble))
        m = Kc @ torch.linalg.solve(A, y[0])
        Cpost = Kc - Kc @ torch.linalg.solve(A, Kc)      # K - K(K+Sigma)^-1 K
        return m, Cpost

    def cov(a, b):
        return torch.einsum('si,sj->ij', a, b.conj()) / a.shape[0]

    # --- complex (circular) covariance: E[(f-m)(f-m)^H] == Cpost ---
    Kc = kc(x[:, None]).to_dense().detach()
    f = posterior_draws_1d(kc, x, y, noise, size=S, generator=g)[:, 0, :]
    m, Cpost = analytic(Kc)
    assert f.shape == (S, Nx) and f.is_complex()
    emp_m = f.mean(0)
    assert (emp_m - m).abs().max() < 0.02
    assert (emp_m - posterior_mean_1d(kc, x, y, noise)[0]).abs().max() < 0.02
    fc = f - emp_m
    assert (cov(fc, fc) - Cpost).abs().max() / Cpost.abs().max() < 0.05

    # --- real covariance, complex data: each part ~ N(m_part, Cpost_real) ---
    Kr = kr(x[:, None]).to_dense().detach()
    fr = posterior_draws_1d(kr, x, y, noise, size=S, generator=g)[:, 0, :]
    mr, Cpost_r = analytic(Kr)                            # Cpost_r real-valued
    emp_mr = fr.mean(0)
    assert (emp_mr - mr).abs().max() < 0.02
    Cr = torch.einsum('si,sj->ij', fr.real - emp_mr.real, fr.real - emp_mr.real) / S
    Ci = torch.einsum('si,sj->ij', fr.imag - emp_mr.imag, fr.imag - emp_mr.imag) / S
    assert (Cr - Cpost_r.real).abs().max() / Cpost_r.abs().max() < 0.06
    assert (Ci - Cpost_r.real).abs().max() / Cpost_r.abs().max() < 0.06


def test_posterior_draws_2d_mean():
    """2D Matheron draws: empirical mean reproduces posterior_mean_2d (complex C1, real C2)."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(2)
    S, N1, N2 = 120_000, 4, 5
    x1 = torch.linspace(0, 50, N1, dtype=torch.float64)
    x2 = torch.linspace(120, 180, N2, dtype=torch.float64)
    k1 = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    k1.base_kernel.base_kernel.lengthscale = 8.0
    k2 = kernels.ScaleKernel(kernels.SincKernel()).double()
    k2.base_kernel.lengthscale = 2.0

    y = torch.randn(1, N1, N2, dtype=torch.cdouble)
    noise = 0.2 + 0.3 * torch.rand(1, N1, N2, dtype=torch.float64)

    f = posterior_draws_2d(k1, k2, x1, x2, y, noise, size=S, method='cholesky', generator=g)
    assert f.shape == (S, 1, N1, N2) and f.is_complex()
    m = posterior_mean_2d(k1, k2, x1, x2, y, noise, method='cholesky')[0]
    assert (f.mean(0)[0] - m).abs().max() < 0.02

    # with a 2D mean function: empirical mean reproduces posterior_mean_2d(mu=...)
    muf = lambda a, b: torch.full((a.shape[0], b.shape[0]), 0.5 + 0.3j, dtype=torch.cdouble)
    fmu = posterior_draws_2d(k1, k2, x1, x2, y, noise, mu=muf, size=S, method='cholesky', generator=g)
    mmu = posterior_mean_2d(k1, k2, x1, x2, y, noise, mu=muf, method='cholesky')[0]
    assert (fmu.mean(0)[0] - mmu).abs().max() < 0.02


# representative HERA geometry for the default kernels in the real-data compose test
BL_VECS = torch.tensor([[14.6, 0.0, 0.0]], dtype=torch.float64)    # meters (ENU)
LAT = -30.72                                                       # degrees
DATA = os.path.join(os.path.dirname(__file__), "zen.h6c_idr2_validation.sum.uvh5")


def test_fit_axis_kernel():
    """fit_axis_kernel does its job -- rank rows by completeness (dropping flagged/garbage
    rows beyond nsamp), stack real/imag for a real covariance, pool and fit in place. It must
    therefore equal fit_kernel applied directly to the pooled, real/imag-stacked clean rows
    (convergence quality itself is fit_kernel's test)."""
    torch.manual_seed(0)
    Nx, Nclean, Ngarb = 16, 40, 16
    x = torch.linspace(0, 20, Nx, dtype=torch.float64)
    ktrue = kernels.ScaleKernel(kernels.RBFKernel()).double()
    ktrue.base_kernel.lengthscale = 3.0
    g = torch.Generator().manual_seed(0)
    clean = torch.complex(prior_draws_1d(ktrue, x, size=Nclean, generator=g),
                          prior_draws_1d(ktrue, x, size=Nclean, generator=g))
    garbage = 1e6 * torch.ones(Ngarb, Nx, dtype=torch.cdouble)      # fully-flagged junk rows
    data = torch.cat([clean, garbage], 0)
    flags = torch.cat([torch.zeros(Nclean, Nx, dtype=bool), torch.ones(Ngarb, Nx, dtype=bool)])
    noise = 0.01 * torch.ones(Nclean + Ngarb, Nx, dtype=torch.float64)

    def fresh():
        k = kernels.ScaleKernel(kernels.RBFKernel()).double()
        k.base_kernel.lengthscale = 1.0
        return k

    kfit = fresh()
    out = fit_axis_kernel(data, flags, noise, x, kfit, nsamp=Nclean, iters=15, opt='Adam')
    assert out is kfit                                              # fit in place, returns kernel

    # equal to fitting only the pooled clean rows (garbage dropped by the completeness ranking;
    # the per-row MLL is order-invariant, so any tie order among clean rows is fine)
    kref = fresh()
    ys = torch.cat([clean.real, clean.imag], 0)
    nzs = torch.cat([noise[:Nclean], noise[:Nclean]], 0)
    fit_kernel(kref, x[None, :, None], ys, nzs, Niter=15, opt='Adam', method='cholesky')
    assert torch.allclose(kfit.base_kernel.lengthscale, kref.base_kernel.lengthscale, atol=1e-6)
    assert torch.allclose(kfit.outputscale, kref.outputscale, atol=1e-6)


def _load(nbls=3, ntimes=48, fslice=slice(None)):
    """Load the least-flagged baselines from the test uvh5 with h5py (no pyuvdata), zero the
    flagged pixels (files store garbage there) and normalize to ~unit scale."""
    pytest.importorskip("h5py")
    import h5py
    if not os.path.exists(DATA):
        pytest.skip("test uvh5 file not present")
    with h5py.File(DATA, "r") as f:
        H = f["Header"]
        a1, a2 = H["ant_1_array"][:], H["ant_2_array"][:]
        freqs = np.asarray(H["freq_array"][:])
        flags_all = f["Data/flags"][:, :, 0]
        bl_ids = a1.astype(np.int64) * 100000 + a2
        cross = np.unique(bl_ids[a1 != a2])
        chosen = [b for _, b in sorted((flags_all[bl_ids == b].mean(), int(b)) for b in cross)[:nbls]]
        d, fl = [], []
        for b in chosen:
            rows = np.sort(np.where(bl_ids == b)[0])[:ntimes]
            d.append(f["Data/visdata"][rows, :, 0][:, fslice])
            fl.append(flags_all[rows][:, fslice])
    data = torch.as_tensor(np.stack(d))
    flags = torch.as_tensor(np.stack(fl)).bool()
    nu = torch.as_tensor(freqs[fslice] / 1e6)                      # MHz
    t = torch.arange(data.shape[1], dtype=torch.float64) * 10.7    # seconds
    data = torch.where(flags, torch.zeros_like(data), data)
    data = data / data[~flags].abs().std()
    return data, flags, t, nu


def test_inpaint_real_data_compose():
    """End-to-end on real HERA data with the composed API (no monolithic entry):
    fit_axis_kernel + inpaint_1d (freq) and + inpaint_2d (joint) fill flagged pixels
    (incl. a fully-flagged channel), leave good pixels untouched, stay finite and bounded."""
    data, flags, t, nu = _load(nbls=3, ntimes=40, fslice=slice(60, 140))
    assert (flags.all(1)).any(), "expected at least one fully-flagged channel"
    Nbls, Nt, Nf = data.shape
    noise = 0.05 ** 2 * torch.ones_like(data.real)
    noise = noise.clone(); noise[flags] = 1e12                     # caller down-weights flags

    # --- freq mode: fit the shared freq kernel on pooled spectra, then per-row inpaint ---
    fk = kernels.default_freq_kernel(BL_VECS).double()
    rows, rfl, rnz = data.reshape(-1, Nf), flags.reshape(-1, Nf), noise.reshape(-1, Nf)
    fit_axis_kernel(rows, rfl, rnz, nu, fk, nsamp=128, iters=20)
    out, _ = inpaint_1d(fk, nu, rows, rnz, rfl)
    out = out.reshape(data.shape)

    assert out.shape == data.shape and out.is_complex() and torch.isfinite(out).all()
    assert torch.equal(out[~flags], data[~flags])                          # good untouched
    assert (out[flags] != data[flags]).float().mean() > 0.5                # flagged filled
    assert out[flags].abs().median() < 10 * data[~flags].abs().median()    # no blow-up

    # --- joint mode: fit both axes, then the separable 2D inpaint (no densify) ---
    tk = kernels.default_time_kernel(nu * 1e6, BL_VECS, LAT).double()
    fit_axis_kernel(data.permute(0, 2, 1).reshape(-1, Nt),
                    flags.permute(0, 2, 1).reshape(-1, Nt),
                    noise.permute(0, 2, 1).reshape(-1, Nt), t, tk, nsamp=128, iters=20)
    outj, _ = inpaint_2d(tk, fk, t, nu, data, noise, flags, method='woodbury', rcond=1e-12)

    assert outj.shape == data.shape and outj.is_complex() and torch.isfinite(outj).all()
    assert torch.equal(outj[~flags], data[~flags])                         # good untouched
