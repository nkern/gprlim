import torch

from gprlim.utils import empirical_cov


def _pairwise_reference(X, flags, mean_subtract=False):
    """Brute-force pairwise-complete covariance, looping over feature pairs."""
    Nf = X.shape[1]
    W = ~flags
    Xc = X
    if mean_subtract:
        w = W.to(X.real.dtype)
        Xc = X - (w * X).sum(0) / w.sum(0).clamp_min(1e-30)
    C = torch.zeros(Nf, Nf, dtype=X.dtype)
    for i in range(Nf):
        for j in range(Nf):
            m = W[:, i] & W[:, j]
            if m.sum() > 0:
                C[i, j] = (Xc[m, i].conj() * Xc[m, j]).sum() / m.sum()
    return C


def test_empirical_cov_matches_pairwise_reference():
    torch.manual_seed(0)
    X = torch.randn(300, 16, dtype=torch.cdouble)
    flags = torch.rand(300, 16) < 0.2
    for mean_subtract in (False, True):
        C = empirical_cov(X, flags, mean_subtract=mean_subtract)
        ref = _pairwise_reference(X, flags, mean_subtract=mean_subtract)
        assert torch.allclose(C, ref), mean_subtract
        assert torch.allclose(C, C.conj().T)            # Hermitian


def test_empirical_cov_respects_flags():
    torch.manual_seed(1)
    X = torch.randn(200, 12, dtype=torch.cdouble)
    flags = torch.rand(200, 12) < 0.3
    C = empirical_cov(X, flags, mean_subtract=False)

    # flagged pixels carry zero weight: corrupting them must not change the result
    Xbad = X.clone()
    Xbad[flags] = 1e6 + 1e6j
    assert torch.allclose(C, empirical_cov(Xbad, flags, mean_subtract=False))

    # min_weight zeros entries with too few jointly-valid samples
    flags2 = flags.clone()
    flags2[1:, 0] = True                                 # feature 0 valid in <=1 sample
    Cz = empirical_cov(X, flags2, mean_subtract=False, min_weight=5)
    assert (Cz[0, :] == 0).all() and (Cz[:, 0] == 0).all()


def test_empirical_cov_psd_projection():
    # low-rank signal + heavy flagging -> the pairwise estimate is not PSD
    torch.manual_seed(0)
    F = torch.randn(20, 3, dtype=torch.cdouble)
    X = torch.randn(150, 3, dtype=torch.cdouble) @ F.conj().T
    flags = torch.rand(150, 20) < 0.4
    C = empirical_cov(X, flags, mean_subtract=False)
    assert torch.linalg.eigvalsh(C).min() < 0           # genuinely indefinite

    # the documented eigenvalue-clip projection restores positive semi-definiteness
    w, V = torch.linalg.eigh(C)
    Cpsd = (V * w.clamp_min(0)) @ V.conj().T
    assert torch.linalg.eigvalsh(Cpsd).min() > -1e-10
    assert torch.allclose(Cpsd, Cpsd.conj().T)
