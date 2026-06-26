import torch
import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import NormalPrior, GammaPrior
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from gprlim.models import (
    gpr_invert,
    cholesky_batched,
    batched_log_prob,
    _sum_log_priors,
    posterior_mean,
    fit_kernel,
)


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


def test_predict_at_new_points():
    """posterior_mean(input_x) matches an explicit K(x*,X)(K+D)^-1(y-mu)+mu with a
    true cross-covariance (m != n)."""
    torch.manual_seed(0)
    n, b, m = 40, 8, 25
    train_x = torch.linspace(0, 1, n, dtype=torch.float64)[None, :, None]
    train_y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
    mean.double()
    covar.double()

    input_x = torch.linspace(0, 1, m, dtype=torch.float64)[None, :, None]
    pred = posterior_mean(covar, train_x, train_y, noise, input_x=input_x, mean=mean)

    with torch.no_grad():
        Cs = covar(train_x).to_dense().squeeze()             # (n, n)
        Cp = covar(input_x, train_x).to_dense().squeeze()    # (m, n) cross-cov
        yc = train_y - mean(train_x)
        mu = mean(input_x).squeeze()
        ref = torch.stack([
            Cp @ torch.linalg.solve(Cs + noise[i].diag(), yc[i]) for i in range(b)
        ]) + mu

    assert pred.shape == (b, m)
    assert Cp.shape == (m, n)
    assert (pred - ref).abs().max() < 1e-8


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
        fit_kernel(covar, x, y, noise, mean=mean, Niter=20, opt='Adam', method=method, lr=0.1)
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
            fit_kernel(covar, x, y, noise, mean=mean, Niter=150, opt='Adam', method=method, lr=0.05)
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
