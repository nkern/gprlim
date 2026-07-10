import os

import numpy as np
import pytest
import torch

from gprlim import pipelines

# reuse the real-data loader + array geometry from the models tests (pytest puts the tests
# directory on sys.path, so a bare import resolves the sibling module)
from test_models import _load, BL_VECS, LAT


@pytest.mark.parametrize("mode", ['2d', '2d_1d', '1d_1d', 'time', 'freq'])
def test_hera_inpaint_modes(mode):
    """hera_inpaint runs each inpaint mode end-to-end on real HERA data: it fills the flagged
    pixels, leaves good pixels untouched, and returns a finite, bounded, complex model of the
    right shape."""
    data, flags, t, nu = _load(nbls=3, ntimes=40, fslice=slice(60, 140))
    noise = 0.05 ** 2 * torch.ones_like(data.real)
    noise = noise.clone(); noise[flags] = 1e12                       # caller down-weights flags

    inp_y, mdl = pipelines.hera_inpaint(
        data, noise, flags, t, nu, inpaint=mode, n_threads=1, cg_tol=1e-3,
        bl_vec=BL_VECS, lat=LAT, 
    )

    assert inp_y.shape == data.shape and inp_y.is_complex()
    assert torch.isfinite(inp_y).all() and torch.isfinite(mdl).all()
    assert torch.allclose(inp_y[~flags], data[~flags], atol=1e-10, rtol=1e-8)  # good pixels untouched
    assert not torch.equal(inp_y[flags], data[flags])                # flagged pixels filled (were 0)
    assert inp_y[flags].abs().max() < 50 * data[~flags].abs().max()  # bounded, no blow-up
