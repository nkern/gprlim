import torch
import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import NormalPrior, GammaPrior

from gprlim.models import (
    gpr_invert,
    cholesky_batched,
    fixednoise_gp_1d,
    batched_log_prob,
    _sum_log_priors,
    optimize_kernel,
)


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
    """GPModel.predict(input_x) matches an explicit K(x*,X)(K+D)^-1(y-mu)+mu with a
    true cross-covariance (m != n)."""
    torch.manual_seed(0)
    n, b, m = 40, 8, 25
    train_x = torch.linspace(0, 1, n, dtype=torch.float64)[None, :, None]
    train_y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
    model, _ = fixednoise_gp_1d(train_x, train_y, mean, covar, inv_wgts=noise)
    model.double()

    input_x = torch.linspace(0, 1, m, dtype=torch.float64)[None, :, None]
    pred = model.predict(input_x=input_x)

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


def _model_with_priors(x, y, noise):
    mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
    covar.base_kernel.register_prior('ls', GammaPrior(2.0, 1.0), 'lengthscale')
    covar.register_prior('os', GammaPrior(2.0, 1.0), 'outputscale')
    mean.register_prior('m', NormalPrior(0.0, 1.0), 'constant')
    model, _ = fixednoise_gp_1d(x, y, mean, covar, inv_wgts=noise)
    return model.double()


def test_batched_log_prob_matches_gpytorch_mll():
    """batched_log_prob (+ priors, mean-centering) reproduces gpytorch's
    ExactMarginalLogLikelihood loss and gradients."""
    torch.manual_seed(0)
    n, b = 25, 5
    x = torch.linspace(0, 4, n, dtype=torch.float64)[None, :, None]
    y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    model = _model_with_priors(x, y, noise)
    model.train(); model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model.zero_grad()
    ref = -mll(model(model.train_inputs[0]), model.train_targets).mean()
    ref.backward()
    gref = {k: p.grad.clone() for k, p in model.named_parameters()}

    for method in ('woodbury', 'cholesky'):
        model.zero_grad()
        xx = model.train_inputs[0]
        lp = batched_log_prob(model.covar(xx).to_dense(), model.likelihood.noise,
                              model.train_targets - model.mean(xx), method=method)
        loss = -((lp + _sum_log_priors(model)) / n).mean()
        loss.backward()
        assert abs(loss.item() - ref.item()) < 1e-9, method
        for k, p in model.named_parameters():
            if p.grad is not None:
                assert (p.grad - gref[k]).abs().max() < 1e-8, (method, k)


def test_optimize_kernel_batched_matches_default():
    """optimize_kernel(batched=...) tracks the dense gpytorch path to the same fit."""
    torch.manual_seed(0)
    n, b = 30, 6
    x = torch.linspace(0, 5, n, dtype=torch.float64)[None, :, None]
    y = torch.randn(b, n, dtype=torch.float64)
    noise = 0.1 + 0.05 * torch.rand(b, n, dtype=torch.float64)

    def fit(**kw):
        torch.manual_seed(1)
        model = _model_with_priors(x, y, noise)
        optimize_kernel(model, Niter=20, opt='Adam', lr=0.1, **kw)
        return dict(model.named_parameters())

    base = fit()
    for method in ('woodbury', 'cholesky'):
        got = fit(batched=method)
        assert max(float((got[k] - base[k]).abs().max()) for k in base) < 1e-7, method


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

    def fit(prior=False, batched=None):
        torch.manual_seed(1)
        mean, covar = ConstantMean(), ScaleKernel(RBFKernel())
        covar.base_kernel.lengthscale = 1.0
        if prior:                                     # tight prior pulling ls -> 0.5
            covar.base_kernel.register_prior('ls', NormalPrior(0.5, 0.1), 'lengthscale')
        model, _ = fixednoise_gp_1d(x, y, mean, covar, inv_wgts=noise)
        model.double()
        optimize_kernel(model, Niter=150, opt='Adam', lr=0.05, batched=batched)
        return float(model.covar.base_kernel.lengthscale.detach())

    ls_noprior = fit(prior=False)
    ls_gpt = fit(prior=True, batched=None)
    ls_wood = fit(prior=True, batched='woodbury')
    ls_chol = fit(prior=True, batched='cholesky')

    # the prior is genuinely active: it pulls ls well away from the data-only fit,
    # toward the prior mean (0.5)
    assert ls_noprior > 1.5                                       # data alone -> ls ~ 2
    assert ls_gpt < 1.0                                           # prior pulls it down
    assert abs(ls_gpt - 0.5) < abs(ls_noprior - 0.5)             # moved toward prior mean
    # the batched paths recover gpytorch's prior-influenced optimum
    assert abs(ls_wood - ls_gpt) < 1e-5
    assert abs(ls_chol - ls_gpt) < 1e-5
