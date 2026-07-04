import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # macOS libomp (see project runs)

import pytest
import torch

from gprlim import solvers, kernels

DOUBLE, CDOUBLE = torch.float64, torch.cdouble


# --------------------------------------------------------------------------------------
# helpers: random vectors / Hermitian-PD matrices / Kronecker references
# --------------------------------------------------------------------------------------
def _vec(shape, cplx, seed):
    """Random tensor of the given shape; complex (re + i*im) if ``cplx``."""
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(*shape, generator=g, dtype=DOUBLE)
    if not cplx:
        return re
    im = torch.randn(*shape, generator=g, dtype=DOUBLE)
    return torch.complex(re, im)


def _hpd(n, cplx, seed, jitter=None):
    """Random Hermitian (cplx) / symmetric (real) positive-definite (n, n) matrix."""
    A = _vec((n, n), cplx, seed)
    return A @ A.conj().transpose(-1, -2) + (jitter or n) * torch.eye(n, dtype=A.dtype)


def _kron_cores(Nt, Nf, cplx, seed=0):
    """Time core P (complex if ``cplx``) and a real frequency core F -- the
    complex-time / real-freq pairing of the actual GP model."""
    F = _hpd(Nf, False, seed)         # frequency core: always real
    P = _hpd(Nt, cplx, seed + 1)      # time core: real or complex Hermitian
    return P, F


def _kron_dense(P, F):
    """Dense P (x) F, with the real factor promoted so a complex/real pair composes."""
    dt = torch.promote_types(P.dtype, F.dtype)
    return torch.kron(P.to(dt), F.to(dt))


def _dense_wiener(P, F, noise, y):
    """Reference Wiener mean  (P (x) F) (P (x) F + diag(noise))^-1 y  formed densely."""
    Ks = _kron_dense(P, F)
    A = Ks + torch.diag(noise.reshape(-1).to(Ks.dtype))
    return (Ks @ torch.linalg.solve(A, y.reshape(-1).to(Ks.dtype))).reshape(y.shape)


# --------------------------------------------------------------------------------------
# core pcg on a DENSE operator -- real and complex
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_pcg_dense(cplx):
    n = 50
    A, b = _hpd(n, cplx, 1), _vec((n,), cplx, 2)
    x, info = solvers.pcg(A, b, tol=1e-10, max_iter=300)
    assert info["resid"] <= 1e-10
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-7)


@pytest.mark.parametrize("cplx", [False, True])
def test_pcg_batched(cplx):
    n, B = 40, 4
    A, b = _hpd(n, cplx, 5), _vec((B, n), cplx, 6)
    x, _ = solvers.pcg(A, b, tol=1e-10, max_iter=300)
    ref = torch.linalg.solve(A, b.transpose(-1, -2)).transpose(-1, -2)   # solve each row
    assert torch.allclose(x, ref, atol=1e-7)


def test_pcg_warmstart_already_solved():
    n = 30
    A, b = _hpd(n, True, 7), _vec((n,), True, 8)
    x0 = torch.linalg.solve(A, b)
    x, info = solvers.pcg(A, b, x0=x0, tol=1e-8)
    assert info["iters"] == 0 and torch.allclose(x, x0)


# --------------------------------------------------------------------------------------
# preconditioners: dense (exact -> 1 step), operator (exact -> 1 step), diagonal
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_dense_preconditioner_exact(cplx):
    n = 40
    A, b = _hpd(n, cplx, 8), _vec((n,), cplx, 9)
    x, info = solvers.pcg(A, b, M=solvers.dense_preconditioner(A), tol=1e-10, max_iter=20)
    assert info["iters"] <= 1                          # exact M^-1 -> converge in one step
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-7)


def test_operator_preconditioner_exact():
    n = 30
    A, b = _hpd(n, True, 12), _vec((n,), True, 13)
    L = torch.linalg.cholesky(A)

    class Chol:                                        # any object exposing .solve
        def solve(self, rhs):
            return torch.cholesky_solve(rhs, L)

    x, info = solvers.pcg(A, b, M=solvers.operator_preconditioner(Chol()), tol=1e-10, max_iter=20)
    assert info["iters"] <= 1
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-7)


@pytest.mark.parametrize("cplx", [False, True])
def test_diag_preconditioner(cplx):
    n = 40
    A, b = _hpd(n, cplx, 10), _vec((n,), cplx, 11)
    M = solvers.diag_preconditioner(A.diagonal().real.to(A.dtype))    # Hermitian -> real diagonal
    x, _ = solvers.pcg(A, b, M=M, tol=1e-10, max_iter=300)
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-7)


# --------------------------------------------------------------------------------------
# structured (kron-sparse) matvec and preconditioner -- real and complex cores
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_kron_matvec(cplx):
    Nt, Nf = 6, 5
    P, F = _kron_cores(Nt, Nf, cplx)
    v = _vec((Nt * Nf,), cplx, 7)
    assert torch.allclose(solvers.kron_matvec(P, F)(v), _kron_dense(P, F) @ v, atol=1e-9)
    # with an added diagonal
    d = torch.rand(Nt * Nf, dtype=DOUBLE) + 0.1
    ref = _kron_dense(P, F) @ v + d * v
    assert torch.allclose(solvers.kron_matvec(P, F, diag=d)(v), ref, atol=1e-9)


@pytest.mark.parametrize("cplx", [False, True])
def test_kron_eigen_preconditioner(cplx):
    Nt, Nf = 6, 5
    P, F = _kron_cores(Nt, Nf, cplx)
    shift = 0.7
    v = _vec((Nt * Nf,), cplx, 3)
    Ks = _kron_dense(P, F)
    ref = torch.linalg.solve(Ks + shift * torch.eye(Nt * Nf, dtype=Ks.dtype), v)
    assert torch.allclose(solvers.kron_eigen_preconditioner(P, F, shift=shift)(v), ref, atol=1e-8)


@pytest.mark.parametrize("cplx", [False, True])
def test_pcg_kron_unpreconditioned(cplx):
    # pcg driving a kron-sparse matvec (never densified) matches the dense solve
    Nt, Nf = 6, 5
    P, F = _kron_cores(Nt, Nf, cplx)
    noise = 0.1 * torch.ones(Nt * Nf, dtype=DOUBLE)
    b = _vec((Nt * Nf,), cplx, 16)
    x, _ = solvers.pcg(solvers.kron_matvec(P, F, diag=noise), b, tol=1e-10, max_iter=1000)
    Ks = _kron_dense(P, F)
    ref = torch.linalg.solve(Ks + torch.diag(noise.to(Ks.dtype)), b)
    assert torch.allclose(x, ref, atol=1e-6)


def test_kron_eigen_preconditioner_accelerates():
    # the structured preconditioner should cut CG iterations vs none
    Nt, Nf = 10, 9
    P, F = _kron_cores(Nt, Nf, True, seed=20)
    noise = 0.01 * torch.ones(Nt * Nf, dtype=DOUBLE)
    A, b = solvers.kron_matvec(P, F, diag=noise), _vec((Nt * Nf,), True, 21)
    _, none = solvers.pcg(A, b, tol=1e-8, max_iter=3000)
    _, pc = solvers.pcg(A, b, M=solvers.kron_eigen_preconditioner(P, F, shift=0.01), tol=1e-8, max_iter=3000)
    assert pc["iters"] < none["iters"]


@pytest.mark.parametrize("cplx", [False, True])
def test_kron_blockdiag_preconditioner(cplx):
    # M^-1 for M = P (x) F + I (x) diag(Df): inverts M exactly (dense reference), real & complex P
    Nt, Nf = 6, 5
    P, F = _kron_cores(Nt, Nf, cplx)
    Df = 0.1 + torch.rand(Nf, dtype=DOUBLE)
    v = _vec((3, Nt * Nf), cplx, 3)
    Ks = _kron_dense(P, F)
    M = Ks + torch.kron(torch.eye(Nt, dtype=Ks.dtype), torch.diag(Df).to(Ks.dtype))
    ref = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(solvers.kron_blockdiag_preconditioner(P, F, Df)(v), ref, atol=1e-8)


def test_kron_blockdiag_preconditioner_accelerates():
    # for time-separable noise (I (x) diag(Df)) the block-diag preconditioner is the EXACT inverse,
    # so preconditioned CG converges in ~1 iteration -- far fewer than the scalar-shift eigen one
    Nt, Nf = 12, 10
    P, F = _kron_cores(Nt, Nf, True, seed=7)
    Df = torch.full((Nf,), 1e-3, dtype=DOUBLE); Df[3:6] = 1e8       # a fully-flagged channel "gap"
    noise = Df.expand(Nt, Nf).reshape(-1)                          # separable: same Df at every time
    A, b = solvers.kron_matvec(P, F, diag=noise), _vec((Nt * Nf,), True, 8)
    _, eig = solvers.pcg(A, b, M=solvers.kron_eigen_preconditioner(P, F, shift=float(Df.min())),
                         tol=1e-8, max_iter=3000, weight=noise.reciprocal())
    _, bd = solvers.pcg(A, b, M=solvers.kron_blockdiag_preconditioner(P, F, Df),
                        tol=1e-8, max_iter=3000, weight=noise.reciprocal())
    assert bd["iters"] <= 2 and bd["iters"] < eig["iters"]


# --------------------------------------------------------------------------------------
# high-level kron_wiener_cg: kron-sparse solve == dense model, real and complex
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_kron_wiener_cg_vs_dense(cplx):
    Nt, Nf = 8, 7
    P, F = _kron_cores(Nt, Nf, cplx)
    y = _vec((Nt, Nf), cplx, 14)
    flags = torch.zeros(Nt, Nf, dtype=bool)
    flags[3, :] = True            # a fully-flagged time integration
    flags[:, 2] = True            # a fully-flagged channel
    noise = 0.1 * torch.ones(Nt, Nf, dtype=DOUBLE)
    noise[flags] = 1e6            # flagged -> large noise variance
    m, info = solvers.kron_wiener_cg(P, F, noise, y, tol=1e-9, max_iter=2000)
    assert m.dtype == (CDOUBLE if cplx else DOUBLE)
    assert torch.allclose(m, _dense_wiener(P, F, noise, y), atol=1e-5)


@pytest.mark.parametrize("cplx", [False, True])
def test_kron_wiener_cg_batched(cplx):
    Nt, Nf, B = 8, 7, 3
    P, F = _kron_cores(Nt, Nf, cplx)
    y = _vec((B, Nt, Nf), cplx, 15)
    flags = torch.zeros(B, Nt, Nf, dtype=bool)
    for b in range(B):
        flags[b, 2 + b, :] = True          # a different flagged integration per baseline
    noise = 0.1 * torch.ones(B, Nt, Nf, dtype=DOUBLE)
    noise[flags] = 1e6
    m, _ = solvers.kron_wiener_cg(P, F, noise, y, tol=1e-9, max_iter=2000)
    assert m.shape == (B, Nt, Nf)
    for b in range(B):
        assert torch.allclose(m[b], _dense_wiener(P, F, noise[b], y[b]), atol=1e-5)


@pytest.mark.parametrize("cplx", [False, True])
def test_kron_wiener_cg_n_threads_matches_serial(cplx):
    """Parallelizing the CG over the baseline axis (n_threads>1) is bit-for-bit the serial
    result -- with per-baseline noise and with a shared (1, Nt, Nf) noise diagonal (the
    operator is then identical across baselines, only the RHS differs)."""
    Nt, Nf, B = 8, 7, 6
    P, F = _kron_cores(Nt, Nf, cplx)
    y = _vec((B, Nt, Nf), cplx, 21)
    flags = torch.zeros(B, Nt, Nf, dtype=bool)
    flags[:, 3, :] = True

    # per-baseline noise
    noise = 0.1 * torch.ones(B, Nt, Nf, dtype=DOUBLE)
    noise[flags] = 1e6
    serial, _ = solvers.kron_wiener_cg(P, F, noise, y, tol=1e-9, max_iter=2000, n_threads=1)
    for nj in (3, -1):
        par, _ = solvers.kron_wiener_cg(P, F, noise, y, tol=1e-9, max_iter=2000, n_threads=nj)
        assert torch.allclose(par, serial, atol=1e-10), ("per-bl", nj)

    # shared (1, Nt, Nf) noise broadcasts -> one operator for all baselines
    shared = 0.1 * torch.ones(1, Nt, Nf, dtype=DOUBLE)
    shared[flags[:1]] = 1e6
    s1, _ = solvers.kron_wiener_cg(P, F, shared, y, tol=1e-9, max_iter=2000, n_threads=1)
    s4, _ = solvers.kron_wiener_cg(P, F, shared, y, tol=1e-9, max_iter=2000, n_threads=4)
    assert torch.allclose(s4, s1, atol=1e-10)


# --------------------------------------------------------------------------------------
# KroneckerKernel.cholesky (delegates to solvers.kron_cholesky): implicit L with L L^H = K
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_kronecker_kernel_cholesky(cplx):
    torch.manual_seed(0)
    Nt, Nf = 6, 5
    times = torch.linspace(0., 10., Nt)
    freqs = torch.linspace(0., 5., Nf)
    kf = kernels.ScaleKernel(kernels.SincKernel()).double()
    kf.base_kernel.lengthscale = 1.5
    kt = kernels.ScaleKernel(kernels.RBFKernel()).double()
    kt.base_kernel.lengthscale = 3.0
    if cplx:                                          # CarrierKernel -> complex Hermitian core
        kt = kernels.CarrierKernel(kt, tau=0.3).double()

    kron = kernels.KroneckerKernel(kt, kf, times, freqs)
    L = kron.cholesky()                               # KroneckerProductTriangularLinearOperator
    Ld = L.to_dense()
    K = kron.covariance().to_dense()

    assert K.dtype == (CDOUBLE if cplx else DOUBLE)
    assert torch.allclose(Ld, torch.tril(Ld))                       # lower triangular
    assert torch.allclose(Ld @ Ld.conj().transpose(-1, -2), K, atol=1e-6)   # L L^H = K
    assert torch.linalg.eigvalsh(K).min() > -1e-8                   # PSD


# --------------------------------------------------------------------------------------
# truncate: low-rank truncated-eigendecomposition reconstruction of a covariance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cplx", [False, True])
def test_truncate(cplx):
    """truncate(C, rcond): keep eigenmodes with lambda >= rcond*lambda_max, drop the rest to
    zero (a pinv-style cutoff -- NOT lifting toward a floor). Hermitian PSD preserved; rank =
    #kept; lazy == dense; rcond=1 -> rank 1."""
    torch.manual_seed(0)
    n = 8
    t = torch.linspace(0, 20, n, dtype=DOUBLE)
    if cplx:
        k = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.1).double()
        k.base_kernel.base_kernel.lengthscale = 5.0
    else:
        k = kernels.ScaleKernel(kernels.RBFKernel()).double(); k.base_kernel.lengthscale = 5.0
    C = k(t[:, None]).to_dense().detach()                       # smooth, rank-deficient

    # rcond=0 / None are no-ops (same object)
    assert solvers.truncate(C, 0.0) is C
    assert solvers.truncate(C, None) is C

    w, Q = torch.linalg.eigh(C)
    for rcond in (1e-3, 1e-1, 1.0):
        S = solvers.truncate(C, rcond)
        keep = w >= rcond * w[-1]                                           # modes kept
        expected = (Q * torch.where(keep, w, torch.zeros_like(w))) @ Q.conj().T
        assert torch.allclose(S, expected, atol=1e-10)                      # the truncated recon
        assert torch.allclose(S, S.conj().transpose(-1, -2))               # Hermitian preserved
        evS = torch.linalg.eigvalsh(S)
        assert int((evS > 1e-9 * evS.amax()).sum()) == int(keep.sum())     # rank = #kept (dropped->0)
        assert evS.amin() > -1e-9                                           # PSD, no lifted floor
    assert int((torch.linalg.eigvalsh(solvers.truncate(C, 1.0)) > 1e-9).sum()) == 1   # rcond=1 -> rank 1

    # lazy operator in -> dense out, same values as the dense path
    op = solvers.to_linear_operator(C)
    assert torch.allclose(solvers.truncate(op, 1e-2), solvers.truncate(C, 1e-2))

    # out-of-range guard
    with pytest.raises(ValueError):
        solvers.truncate(C, 1.5)


# --------------------------------------------------------------------------------------
# kron_woodbury_predict / kron_woodbury_inpaint (moved from the deleted test_workflows)
# --------------------------------------------------------------------------------------
def test_kron_woodbury_selftest():
    """kron_woodbury_predict matches the dense Kronecker Wiener mean for full-rank and
    low-rank cores; kron_woodbury_inpaint fills flagged pixels (incl. a fully-flagged
    channel) on complex data and leaves good pixels untouched."""
    Nt, Nf, b = 6, 5, 4
    P, F = _hpd(Nt, False, 1), _hpd(Nf, False, 2)
    noise = 0.1 + torch.rand(b, Nt, Nf, dtype=DOUBLE)
    y = _vec((b, Nt, Nf), False, 3)

    # full-rank cores -> exact match to the dense Kronecker solve
    m = solvers.kron_woodbury_predict(P, F, noise, y, rcond=1e-15)
    ref = torch.stack([_dense_wiener(P, F, noise[i], y[i]) for i in range(b)])
    assert torch.isfinite(m).all()
    assert (m - ref).abs().max() < 1e-9

    # low-rank cores (rank 2 (x) rank 3) -> still exact for the low-rank Ks
    Plr = _vec((Nt, 2), False, 4); Plr = Plr @ Plr.T
    Flr = _vec((Nf, 3), False, 5); Flr = Flr @ Flr.T
    m2 = solvers.kron_woodbury_predict(Plr, Flr, noise, y, rcond=1e-12)
    ref2 = torch.stack([_dense_wiener(Plr, Flr, noise[i], y[i]) for i in range(b)])
    assert (m2 - ref2).abs().max() < 1e-7

    # complex inpaint wrapper: fills flags (real cov split == complex solve), good untouched
    data = _vec((b, Nt, Nf), True, 6)
    flags = torch.rand(b, Nt, Nf) < 0.25
    flags[:, :, 2] = True                                    # all-times-flagged channel
    var = 0.05 ** 2 * torch.ones(b, Nt, Nf, dtype=DOUBLE)
    out = solvers.kron_woodbury_inpaint(data, flags, P, F, var, rcond=1e-15)

    nz = var.clone(); nz[flags] = 1e12
    Ksc = _kron_dense(P, F).to(CDOUBLE)                       # promote real cov -> complex solve
    ref_c = torch.stack([
        (Ksc @ torch.linalg.solve(Ksc + torch.diag(nz[i].reshape(-1).to(CDOUBLE)),
                                  data[i].reshape(-1))).reshape(Nt, Nf) for i in range(b)])
    assert torch.allclose(out, torch.where(flags, ref_c, data), atol=1e-7)
    assert torch.allclose(out[~flags], data[~flags])         # good pixels untouched
    assert (out[:, :, 2].abs() > 1e-9).all()                 # all-flagged channel filled
