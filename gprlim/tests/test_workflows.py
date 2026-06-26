import os

import numpy as np
import pytest
import torch
import gpytorch

h5py = pytest.importorskip("h5py")

from gprlim.workflows import inpaint
from gprlim.solvers import kron_woodbury_inpaint, kron_woodbury_predict, kron_wiener_cg
from gprlim import kernels

DATA = os.path.join(os.path.dirname(__file__), "zen.h6c_idr2_validation.sum.uvh5")
# Tests that read the uvh5 file all go through _load(), which skips if it's absent; the
# synthetic kron-woodbury self-test below needs no file and runs unconditionally.

# representative HERA geometry for building the default kernels (these tests check
# mechanics, not physical fidelity, so one representative baseline + latitude suffice)
BL_VECS = torch.tensor([[14.6, 0.0, 0.0]], dtype=torch.float64)    # meters (ENU)
LAT = -30.72                                                       # degrees


def _flagged_noise(noise, flags, big=1e12):
    """New inpaint contract: the caller down-weights flagged pixels (large variance);
    inpaint no longer injects this. Returns a copy of ``noise`` with flags -> ``big``."""
    noise = noise.clone()
    noise[flags] = big
    return noise


def _real_kernels():
    """Explicit real (Sinc/RBF) freq + time kernels for the real-core tests (the default
    time kernel is complex, so the real-core paths must supply their own)."""
    fk = kernels.ScaleKernel(kernels.SincKernel()).double()
    fk.base_kernel.lengthscale = 5.0
    tk = kernels.ScaleKernel(kernels.RBFKernel()).double()
    tk.base_kernel.lengthscale = 50.0
    return fk, tk


def _load(nbls=3, ntimes=48, fslice=slice(None)):
    """
    Load a few baselines straight from the uvh5 file with h5py (no pyuvdata dep).

    Picks the least-flagged cross-correlation baselines, returns complex data and
    flags of shape (nbls, ntimes, Nfreqs), the time/frequency axes, and -- since real
    files store garbage at flagged pixels -- zeros the flagged pixels and normalizes
    to ~unit scale (visibilities are Jy-scale, which otherwise makes the kernel fit
    numerically touchy).
    """
    if not os.path.exists(DATA):
        pytest.skip("test uvh5 file not present")
    with h5py.File(DATA, "r") as f:
        H = f["Header"]
        a1, a2 = H["ant_1_array"][:], H["ant_2_array"][:]
        freqs = np.asarray(H["freq_array"][:])
        flags_all = f["Data/flags"][:, :, 0]                       # (Nblts, Nfreqs)

        bl_ids = a1.astype(np.int64) * 100000 + a2
        cross = np.unique(bl_ids[a1 != a2])
        fracs = sorted((flags_all[bl_ids == b].mean(), int(b)) for b in cross)
        chosen = [b for _, b in fracs[:nbls]]

        d, fl = [], []
        for b in chosen:
            rows = np.sort(np.where(bl_ids == b)[0])[:ntimes]
            d.append(f["Data/visdata"][rows, :, 0][:, fslice])
            fl.append(flags_all[rows][:, fslice])

    data = torch.as_tensor(np.stack(d))
    flags = torch.as_tensor(np.stack(fl)).bool()
    nu = torch.as_tensor(freqs[fslice] / 1e6)                      # MHz
    t = torch.arange(data.shape[1], dtype=torch.float64) * 10.7    # seconds (HERA ~10.7 s)

    data = torch.where(flags, torch.zeros_like(data), data)        # flagged -> 0
    data = data / data[~flags].abs().std()                         # -> ~unit scale
    return data, flags, t, nu


def test_inpaint_freq_real_data():
    """Frequency inpaint on real HERA data, per-baseline noise, with a fully-flagged
    channel present: output finite, good pixels untouched, flagged pixels filled with
    a sensible (bounded) model."""
    data, flags, t, nu = _load(nbls=3, ntimes=48)
    assert (flags.all(1)).any(), "expected at least one fully-flagged channel"

    torch.manual_seed(0)
    noise = 0.05 ** 2 * torch.ones_like(data.real)                 # (Nbls, Nt, Nf) per-baseline
    noise = _flagged_noise(noise, flags)                           # caller down-weights flags
    out = inpaint(data, flags, t, nu, noise=noise, bl_vecs=BL_VECS, fit=True, fit_iter=20, fit_nsamp=128)

    assert out.shape == data.shape and out.is_complex()
    assert torch.isfinite(out).all()
    assert torch.allclose(out[~flags], data[~flags], atol=1e-4)            # good untouched
    assert (out[flags] != data[flags]).float().mean() > 0.5               # flagged filled
    assert out[flags].abs().median() < 10 * data[~flags].abs().median()   # no blow-up


def test_inpaint_handles_2d_and_shared_noise():
    """A single baseline (2-D input) with shared (1, Nt, Nf) noise round-trips shape
    and leaves good pixels untouched."""
    data, flags, t, nu = _load(nbls=1, ntimes=48)
    data2d, flags2d = data[0], flags[0]

    torch.manual_seed(0)
    noise = 0.05 ** 2 * torch.ones(1, *data2d.shape)              # (1, Nt, Nf) shared
    noise = _flagged_noise(noise, flags2d[None])                  # caller down-weights flags
    out = inpaint(data2d, flags2d, t, nu, noise=noise, bl_vecs=BL_VECS, fit=True, fit_iter=20, fit_nsamp=128)

    assert out.shape == data2d.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[~flags2d], data2d[~flags2d], atol=1e-4)


def test_inpaint_joint_real_data():
    """Joint (P (x) F) inpaint on a frequency sub-band of real data: finite, good
    pixels untouched."""
    data, flags, t, nu = _load(nbls=2, ntimes=40, fslice=slice(60, 140))

    torch.manual_seed(0)
    noise = 0.05 ** 2 * torch.ones_like(data.real)               # (Nbls, Nt, Nf) per-baseline
    noise = _flagged_noise(noise, flags)                         # caller down-weights flags
    out = inpaint(data, flags, t, nu, noise=noise, mode="joint", bl_vecs=BL_VECS, latitude=LAT,
                  fit=True, fit_iter=20, fit_nsamp=128)

    assert out.shape == data.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[~flags], data[~flags], atol=1e-4)


def _fit_real_cores(nbls=2, ntimes=24, fslice=slice(70, 130)):
    """Build real time/frequency cores on a slab of real data from explicit (real) Sinc/RBF
    kernels -- a helper for the real-core solver tests (the default time kernel is complex)."""
    data, flags, t, nu = _load(nbls=nbls, ntimes=ntimes, fslice=fslice)
    noise = 0.05 ** 2 * torch.ones(1, *data.shape[1:], dtype=data.real.dtype)
    fk, tk = _real_kernels()
    P = tk(t[:, None]).to_dense().detach()
    F = fk(nu[:, None]).to_dense().detach()
    return data, flags, t, nu, noise, P, F


def test_kron_woodbury_matches_dense_real_cores():
    """The batched Kronecker-Woodbury solver reproduces an exact dense per-baseline solve
    on real fitted cores (moderate, well-conditioned noise -- the flag-conditioning is
    tested separately by the inpaint wrapper)."""
    _, _, t, nu, _, P, F = _fit_real_cores()
    Nt, Nf = len(t), len(nu)
    assert torch.linalg.matrix_rank(F) < Nf      # real cores really are low rank

    torch.manual_seed(1)
    b = 4
    noise = 0.1 + torch.rand(b, Nt, Nf, dtype=torch.float64)
    y = torch.randn(b, Nt, Nf, dtype=torch.float64)
    m = kron_woodbury_predict(P, F, noise, y, rcond=1e-12)

    Ks = torch.kron(P, F)
    ref = torch.stack([
        (Ks @ torch.linalg.solve(Ks + torch.diag(noise[i].reshape(-1)),
                                 y[i].reshape(-1))).reshape(Nt, Nf)
        for i in range(b)
    ])
    assert torch.isfinite(m).all()
    assert (m - ref).abs().max() < 1e-7


def test_kron_woodbury_inpaint_real_data():
    """The Kronecker-Woodbury inpaint wrapper runs on real data: finite, good pixels
    untouched, flagged pixels filled with a bounded model."""
    data, flags, t, nu, noise, P, F = _fit_real_cores(nbls=2, ntimes=40,
                                                       fslice=slice(60, 140))
    out = kron_woodbury_inpaint(data, flags, P, F, noise, rcond=1e-10)

    assert out.shape == data.shape and torch.isfinite(out).all()
    assert torch.allclose(out[~flags], data[~flags], atol=1e-5)            # good untouched
    assert out[flags].abs().median() < 10 * data[~flags].abs().median()   # filled, no blow-up


def test_joint_real_shared_equals_per_baseline_noise():
    """Joint mode, real covariance (unpack_complex=True): passing flags+noise shared as
    (1, Nt, Nf) vs the same broadcast to per-baseline (Nbls, ...) gives the identical
    result -- both go through one batched kron solve over all (real/imag x baseline) RHS,
    never a per-baseline loop."""
    data, flags, t, nu = _load(nbls=3, ntimes=16, fslice=slice(60, 90))    # N = 16*30
    Nbls = data.shape[0]
    shared_flags = flags[:1]                                               # (1, Nt, Nf) -> shared
    shared_noise = 0.05 ** 2 * torch.ones(1, *data.shape[1:], dtype=data.real.dtype)
    shared_noise = _flagged_noise(shared_noise, shared_flags)              # caller down-weights flags

    fk, tk = _real_kernels()                                               # real cores -> real path
    out_shared = inpaint(data, shared_flags, t, nu, noise=shared_noise, mode="joint",
                         freq_kernel=fk, time_kernel=tk, unpack_complex=True, fit=False)
    # the same mask/noise broadcast to per-baseline -> still one batched solve, not a loop
    out_perbl = inpaint(data, shared_flags.expand(Nbls, -1, -1).contiguous(), t, nu,
                        noise=shared_noise.expand(Nbls, -1, -1).contiguous(),
                        freq_kernel=fk, time_kernel=tk, mode="joint", unpack_complex=True, fit=False)

    assert torch.isfinite(out_shared).all()
    assert (out_shared - out_perbl).abs().max() < 1e-9


def test_joint_real_cov_cg_batches_no_loop():
    """Real covariance + complex data in joint 'cg' mode batches all (real/imag x baseline)
    RHS through one kron_wiener_cg -- no per-baseline loop, even with PER-baseline flags &
    noise (which used to trigger the loop). The fill matches a dense per-baseline Wiener
    solve, and n_jobs only changes parallelism, not the answer."""
    torch.manual_seed(0)
    Nbls, Nt, Nf = 6, 10, 14
    times = torch.linspace(0, 50, Nt, dtype=torch.float64)
    freqs = torch.linspace(120, 180, Nf, dtype=torch.float64)
    kt = kernels.ScaleKernel(kernels.RBFKernel()).double(); kt.base_kernel.lengthscale = 15.0
    kf = kernels.ScaleKernel(kernels.SincKernel()).double(); kf.base_kernel.lengthscale = 2.0
    data = torch.randn(Nbls, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(Nbls, Nt, Nf, dtype=bool)
    for b in range(Nbls):
        flags[b, :, 3 + b % 4] = True                  # per-baseline (non-shared) flags
    noise = _flagged_noise((0.02 + 0.01 * torch.rand(Nbls, Nt, Nf)).double(), flags)

    kw = dict(noise=noise, freq_kernel=kf, time_kernel=kt, mode='joint', method='cg',
              unpack_complex=True, fit=False, center=False, cg_tol=1e-10, cg_max_iter=3000)
    out = inpaint(data, flags, times, freqs, n_jobs=1, **kw)
    out4 = inpaint(data, flags, times, freqs, n_jobs=4, **kw)

    # dense per-baseline reference (a real cov acts identically on real & imag)
    Ct = kt(times[:, None]).to_dense().detach(); Cf = kf(freqs[:, None]).to_dense().detach()
    Ks = torch.kron(Ct, Cf)
    ref = data.clone()
    for b in range(Nbls):
        A = Ks + torch.diag(noise[b].reshape(-1))
        mr = (Ks @ torch.linalg.solve(A, data[b].real.reshape(-1))).reshape(Nt, Nf)
        mi = (Ks @ torch.linalg.solve(A, data[b].imag.reshape(-1))).reshape(Nt, Nf)
        ref[b] = torch.where(flags[b], torch.complex(mr, mi), data[b])

    assert out.dtype == torch.cdouble
    assert torch.allclose(out, ref, atol=1e-7)
    assert torch.allclose(out4, ref, atol=1e-7)            # n_jobs is just parallelism
    assert torch.allclose(out[~flags], data[~flags], atol=1e-8)


def test_kron_woodbury_selftest():
    """Synthetic self-test (no data file): the Kronecker-Woodbury solver matches a dense
    per-baseline reference for full-rank and low-rank cores, and the inpaint wrapper fills
    flagged pixels / leaves good pixels untouched on complex data."""
    torch.manual_seed(0)
    Nt, Nf, b = 6, 5, 4
    f64 = torch.float64

    def dense_ref(P, F, noise, y):
        Ks = torch.kron(P, F)
        return torch.stack([
            (Ks @ torch.linalg.solve(Ks + torch.diag(noise[i].reshape(-1)),
                                     y[i].reshape(-1))).reshape(Nt, Nf)
            for i in range(b)
        ])

    A = torch.randn(Nt, Nt, dtype=f64); P = A @ A.T / Nt + 0.3 * torch.eye(Nt, dtype=f64)
    B = torch.randn(Nf, Nf, dtype=f64); F = B @ B.T / Nf + 0.3 * torch.eye(Nf, dtype=f64)
    noise = 0.1 + torch.rand(b, Nt, Nf, dtype=f64)
    y = torch.randn(b, Nt, Nf, dtype=f64)

    # 1. full-rank cores -> exact match to the dense Kronecker solve
    m = kron_woodbury_predict(P, F, noise, y, rcond=1e-15)
    assert torch.isfinite(m).all()
    assert (m - dense_ref(P, F, noise, y)).abs().max() < 1e-9

    # 2. low-rank cores (rank 2 x rank 3) -> still exact for the low-rank Ks
    Plr = torch.randn(Nt, 2, dtype=f64); Plr = Plr @ Plr.T
    Flr = torch.randn(Nf, 3, dtype=f64); Flr = Flr @ Flr.T
    m2 = kron_woodbury_predict(Plr, Flr, noise, y, rcond=1e-12)
    assert (m2 - dense_ref(Plr, Flr, noise, y)).abs().max() < 1e-7

    # 3. complex inpaint wrapper: fills flags, matches the dense joint solve there
    data = torch.randn(b, Nt, Nf, dtype=torch.cdouble)
    flags = torch.rand(b, Nt, Nf) < 0.25
    flags[:, :, 2] = True                                            # all-times-flagged channel
    var = 0.05 ** 2 * torch.ones(b, Nt, Nf, dtype=f64)
    out = kron_woodbury_inpaint(data, flags, P, F, var, rcond=1e-15)
    nz = var.clone(); nz[flags] = 1e12
    ref = kron_woodbury_predict(P, F, torch.cat([nz, nz]),
                                torch.cat([data.real, data.imag]), rcond=1e-15)
    filled = torch.where(flags, torch.complex(ref[:b], ref[b:]), data)
    assert torch.allclose(out, filled)
    assert torch.allclose(out[~flags], data[~flags])                 # good pixels untouched
    assert (out[:, :, 2].abs() > 1e-9).all()                         # all-flagged channel filled


def test_freq_accepts_shared_flags():
    """flags/noise given as (1, Nt, Nf) (shared across baselines) broadcast correctly in
    freq mode: right shape, finite, good pixels untouched on every baseline."""
    data, flags, t, nu = _load(nbls=3, ntimes=40)
    shared_flags = flags[:1]
    shared_noise = 0.05 ** 2 * torch.ones(1, *data.shape[1:], dtype=data.real.dtype)
    shared_noise = _flagged_noise(shared_noise, shared_flags)             # caller down-weights flags

    out = inpaint(data, shared_flags, t, nu, noise=shared_noise, bl_vecs=BL_VECS, fit=True,
                  fit_iter=15, fit_nsamp=128)

    assert out.shape == data.shape and torch.isfinite(out).all()
    keep = ~shared_flags.expand_as(data)
    assert torch.allclose(out[keep], data[keep], atol=1e-4)


def test_inpaint_joint_woodbury_complex():
    """Production joint-mode path on a COMPLEX covariance (no data file): mode='joint',
    method='woodbury' is the no-densify Kronecker-Woodbury solve. It must route to
    solvers.kron_woodbury_predict (match a direct call) and leave good pixels untouched."""
    from gprlim import kernels
    torch.manual_seed(0)
    Nt, Nf = 10, 12
    times = torch.linspace(0., 50., Nt)
    freqs = torch.linspace(120., 180., Nf)
    # complex time core (CarrierKernel) (x) real freq core
    kt = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    kt.base_kernel.base_kernel.lengthscale = 8.0
    kf = kernels.ScaleKernel(kernels.SincKernel()).double()
    kf.base_kernel.lengthscale = 2.0

    data = torch.randn(2, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(2, Nt, Nf, dtype=bool)
    flags[:, 4, :] = True            # a fully-flagged time integration
    flags[:, :, 5] = True            # a fully-flagged channel
    noise = 0.05 ** 2 * torch.ones(2, Nt, Nf, dtype=torch.float64)
    noise = _flagged_noise(noise, flags)                          # caller down-weights flags

    out = inpaint(data, flags, times, freqs, noise=noise, freq_kernel=kf, time_kernel=kt,
                  mode="joint", method="woodbury", fit=False, center=False, rcond=1e-12)

    # reference: a direct Kronecker-Woodbury fill with the same cores / noise
    P = kt(times[:, None]).to_dense().detach()
    F = kf(freqs[:, None]).to_dense().detach()
    m = kron_woodbury_predict(P, F, noise, data, rcond=1e-12)
    ref = torch.where(flags, m, data)

    assert out.dtype == torch.cdouble
    assert torch.allclose(out[~flags], data[~flags], atol=1e-6)   # good pixels untouched
    assert torch.allclose(out, ref, atol=1e-7)                    # wired to the woodbury solver


def test_inpaint_joint_cg_complex():
    """Production joint-mode path on a COMPLEX covariance with method='cg': the
    preconditioned complex CG solver (no densify, no truncation). It must route to
    solvers.kron_wiener_cg (match a direct call) and leave good pixels untouched."""
    from gprlim import kernels
    torch.manual_seed(0)
    Nt, Nf = 10, 12
    times = torch.linspace(0., 50., Nt)
    freqs = torch.linspace(120., 180., Nf)
    kt = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    kt.base_kernel.base_kernel.lengthscale = 8.0
    kf = kernels.ScaleKernel(kernels.SincKernel()).double()
    kf.base_kernel.lengthscale = 2.0

    data = torch.randn(2, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(2, Nt, Nf, dtype=bool)
    flags[:, 4, :] = True
    flags[:, :, 5] = True
    noise = 0.05 ** 2 * torch.ones(2, Nt, Nf, dtype=torch.float64)
    noise = _flagged_noise(noise, flags)                          # caller down-weights flags

    out = inpaint(data, flags, times, freqs, noise=noise, freq_kernel=kf, time_kernel=kt,
                  mode="joint", method="cg", cg_tol=1e-10, cg_max_iter=2000,
                  fit=False, center=False)

    # reference: a direct preconditioned-CG fill with the same kernels / noise / tol
    Ct = kt(times[:, None]).to_dense().detach()
    Cf = kf(freqs[:, None]).to_dense().detach()
    m, _ = kron_wiener_cg(Ct, Cf, noise, data, tol=1e-10, max_iter=2000)
    ref = torch.where(flags, m, data)

    assert out.dtype == torch.cdouble
    assert torch.allclose(out[~flags], data[~flags], atol=1e-6)   # good pixels untouched
    assert torch.allclose(out, ref, atol=1e-7)                    # wired to the CG solver


def test_inpaint_joint_eta_time_decorrelation():
    """The joint-mode eta knob shrinks Ct toward its diagonal. eta=0 is the unchanged full
    joint; eta=1 decouples the solve into per-time freq inpaints, so (with a unit-amplitude
    time kernel, Ct[0,0]=1) joint(eta=1) must equal mode='freq'."""
    torch.manual_seed(0)
    Nt, Nf = 8, 12
    times = torch.linspace(0., 50., Nt, dtype=torch.float64)
    freqs = torch.linspace(120., 180., Nf, dtype=torch.float64)
    # unit-amplitude complex time kernel so eta=1 (Ct -> Ct[0,0]*I = I) reduces exactly to
    # a per-time freq solve with Cf
    kt = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.05).double()
    kt.base_kernel.base_kernel.lengthscale = 8.0
    kt.base_kernel.outputscale = 1.0
    kf = kernels.ScaleKernel(kernels.SincKernel()).double()
    kf.base_kernel.lengthscale = 2.0

    data = torch.randn(2, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(2, Nt, Nf, dtype=bool)
    flags[:, :, 5] = True            # fully-flagged channel (freq-fillable)
    flags[0, 3, 2] = True            # a scattered flag
    noise = 0.05 ** 2 * torch.ones(2, Nt, Nf, dtype=torch.float64)
    noise = _flagged_noise(noise, flags)

    kw = dict(noise=noise, freq_kernel=kf, fit=False, center=False)
    # eta=0 is a no-op (identical to the default full joint)
    base = inpaint(data, flags, times, freqs, time_kernel=kt, mode="joint", method="woodbury", **kw)
    eta0 = inpaint(data, flags, times, freqs, time_kernel=kt, mode="joint", method="woodbury", eta=0.0, **kw)
    assert torch.equal(base, eta0)

    # eta=1 decouples -> equals plain freq inpaint (exact via the cholesky/densify path)
    j1 = inpaint(data, flags, times, freqs, time_kernel=kt, mode="joint", method="cholesky", eta=1.0, **kw)
    fr = inpaint(data, flags, times, freqs, mode="freq", method="cholesky", **kw)
    assert torch.allclose(j1, fr, atol=1e-8)


def test_axis_inpaint_complex_kernel_ignores_unpack_complex():
    """A complex covariance couples real & imag, so unpack_complex=True is invalid for the
    axis (freq/time) modes: it must be ignored (warn + fall back to the direct complex
    solve) rather than stacking real/imag against a complex covariance, which used to raise
    a real-vs-complex dtype error. The fallback must match unpack_complex=False."""
    import warnings
    torch.manual_seed(0)
    Nt, Nf = 12, 10
    times = torch.linspace(0., 100., Nt, dtype=torch.float64)
    freqs = torch.linspace(120., 180., Nf, dtype=torch.float64)
    # explicit COMPLEX time kernel (CarrierKernel) -> the time-axis path sees a complex cov
    kt = kernels.CarrierKernel(kernels.ScaleKernel(kernels.RBFKernel()), tau=0.02).double()
    kt.base_kernel.base_kernel.lengthscale = 20.0

    data = torch.randn(2, Nt, Nf, dtype=torch.cdouble)
    flags = torch.zeros(2, Nt, Nf, dtype=bool)
    flags[:, 4, :] = True            # fully-flagged time integration (filled via time corr)
    flags[0, 7, 2] = True            # a scattered flag
    noise = 0.05 ** 2 * torch.ones(2, Nt, Nf, dtype=torch.float64)
    noise = _flagged_noise(noise, flags)                          # caller down-weights flags

    kw = dict(noise=noise, time_kernel=kt, mode="time", method="woodbury", fit=False)
    out_direct = inpaint(data, flags, times, freqs, unpack_complex=False, **kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out_unpack = inpaint(data, flags, times, freqs, unpack_complex=True, **kw)

    assert any("unpack_complex=True ignored" in str(x.message) for x in w)   # warned, didn't crash
    assert out_unpack.dtype == torch.cdouble and torch.isfinite(out_unpack).all()
    assert torch.allclose(out_unpack, out_direct, atol=1e-10)                # fell back -> same answer
    assert torch.allclose(out_unpack[~flags], data[~flags], atol=1e-6)       # good pixels untouched
