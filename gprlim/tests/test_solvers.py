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

    kron = kernels.KroneckerKernel([kt, kf], [times, freqs])
    L = kron.cholesky()                               # KroneckerProductTriangularLinearOperator
    Ld = L.to_dense()
    K = kron.covariance().to_dense()

    assert K.dtype == (CDOUBLE if cplx else DOUBLE)
    assert torch.allclose(Ld, torch.tril(Ld))                       # lower triangular
    assert torch.allclose(Ld @ Ld.conj().transpose(-1, -2), K, atol=1e-6)   # L L^H = K
    assert torch.linalg.eigvalsh(K).min() > -1e-8                   # PSD
