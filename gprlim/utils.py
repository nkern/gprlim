import numpy as np
import torch


_SPEED_OF_LIGHT = 299_792_458.0      # m / s
_SDAY_SEC       = 86_164.0905        # sidereal day [s]  (astropy: units.sday.to("s"))


def detect_contiguous(flags, width, axis):
    """
    Detect a contiguous block of flags
    
    Parameters
    ----------
    flags : ndarray
        Boolean flags
    width : int
        Minimum width of contiguous flags to detect
    axis : int
        Axis of flags to detect contiguous flags over
        
    Returns
    -------
    ndarray
    """
    f = np.asarray(np.take(flags, 0, axis=axis), dtype=int)
    wide_flags = np.zeros_like(flags)
    idx = [slice(None) for i in range(flags.ndim)]
    for i in range(1, flags.shape[axis]):
        # add current flags to counter
        f += np.take(flags, i, axis=axis)
        
        # if unflagged, restart counter
        f *= np.take(flags, i, axis=axis)
        
        # check if counter is above width, update wide_flags
        if i < width - 1:
            pass
        else:
            for j in range(0, width):
                idx[axis] = i - j
                wide_flags[*idx] += f >= width

    return wide_flags


def empirical_cov(X, flags=None, weights=None, mean_subtract=True, min_weight=1.0):
    """
    Weighted empirical (Hermitian) covariance over the sample axis, with flagged
    pixels given zero weight.

    Treats rows of ``X`` as samples and columns as features, returning the
    (Nfeatures, Nfeatures) covariance. Each entry ``C[i, j]`` is the weighted
    average of ``conj(x_i) * x_j`` over the samples where both feature ``i`` and
    feature ``j`` are unflagged: the normalization is the *pairwise* valid-sample
    count ``(W.T @ W)[i, j]``, not the total number of samples. For data of shape
    (Ntimes, Nfreqs) this yields the frequency-frequency covariance; transpose
    ``X`` for the time-time covariance.

    Parameters
    ----------
    X : tensor
        Data of shape (Nsamples, Nfeatures), real or complex.
    flags : tensor, optional
        Boolean array of shape (Nsamples, Nfeatures), True where a pixel is
        flagged (excluded). Ignored if ``weights`` is provided.
    weights : tensor, optional
        Per-pixel weights of shape (Nsamples, Nfeatures). Overrides ``flags``
        (flagged pixels should carry weight zero); pass e.g. inverse noise
        variance for soft weighting. Defaults to ``(~flags)`` (1 for good, 0 for
        flagged), or all ones if neither ``flags`` nor ``weights`` is given.
    mean_subtract : bool
        If True, subtract the weighted per-feature mean before forming products.
    min_weight : float
        Entries whose pairwise weight sum ``(W.T @ W)[i, j]`` falls below this are
        set to zero (too few jointly-valid samples to estimate reliably).

    Returns
    -------
    tensor
        Hermitian covariance of shape (Nfeatures, Nfeatures). For complex ``X``,
        ``C[i, j] = <conj(x_i) x_j>``.

    Notes
    -----
    The pairwise normalization makes each entry an unbiased average over the
    samples that observed both features, but it does *not* guarantee a positive
    semi-definite result: different entries are estimated from different sample
    subsets, so the matrix can have negative eigenvalues. If the covariance will
    be used as a GP kernel, project it onto the nearest PSD matrix by clipping the
    negative eigenvalues::

        w, V = torch.linalg.eigh(C)
        C = (V * w.clamp_min(0)) @ V.conj().T

    Alternatives that are PSD by construction: normalize by a constant
    (``Nsamples``) instead of the pairwise counts (biases heavily-flagged pairs
    toward zero), or keep only samples with no flags at all (complete-case, which
    discards data when flags are scattered).
    """
    if weights is None:
        weights = torch.ones_like(X.real) if flags is None else (~flags).to(X.real.dtype)
    W = weights

    # weighted per-feature mean, then zero out the flagged pixels
    if mean_subtract:
        mean = (W * X).sum(0) / W.sum(0).clamp_min(1e-30)
        X = X - mean
    Xw = X * W

    # numerator: sum_s conj(Xw_si) Xw_sj ;  denominator: pairwise valid-sample counts
    num = Xw.conj().transpose(-2, -1) @ Xw
    den = W.transpose(-2, -1) @ W
    C = num / den.clamp_min(1e-30)

    # zero entries with too few jointly-valid samples
    return torch.where(den >= min_weight, C, torch.zeros_like(C))


def sky_frate_range(freqs, bl_vecs, lat):
    """
    Range [min, max] of sky fringe rates (mHz) for one or more baselines.

    Closed form from Parsons et al. (2016): the fringe rates produced by sky emission
    across the visible hemisphere for ENU baselines at a given latitude and frequency.

    Parameters
    ----------
    freqs : float or array-like
        Observing frequency/frequencies in Hz, shape (Nfreqs,).
    bl_vecs : array-like
        ENU (East, North, Up) baseline vector(s) in meters, shape (Nbls, 3); a single
        length-3 vector is promoted to (1, 3).
    lat : float
        Observer latitude in degrees.

    Returns
    -------
    min_frate, max_frate : torch.Tensor
        Lower / upper sky fringe rate, shape (Nbls, Nfreqs), in mHz.
    """
    freqs  = torch.atleast_1d(torch.as_tensor(freqs))   # (Nfreqs,)
    bl_vecs = torch.atleast_2d(torch.as_tensor(bl_vecs))   # (Nbls, 3)

    bl_en  = bl_vecs[:, :2].norm(dim=-1)                          # (Nbls,) East-North length [m]
    sinlat = np.sin(np.abs(np.radians(lat)))               # scalar
    blcos  = bl_vecs[:, 0] / bl_en.clamp_min(1e-30)              # (Nbls,) cos(angle from East)
    amp    = bl_en * 2 * np.pi / (_SDAY_SEC * 1e-3) / _SPEED_OF_LIGHT   # (Nbls,) per-Hz; 1e-3 -> mHz

    big   = amp * torch.sqrt(sinlat**2 + blcos**2 * (1 - sinlat**2))   # (Nbls,)
    small = amp * sinlat                                                # (Nbls,)

    # per-baseline orientation: East-leaning (blcos>=0) vs West-leaning
    max_df = torch.where(blcos >= 0,  big, small)               # (Nbls,)
    min_df = torch.where(blcos >= 0, -small, -big)              # (Nbls,)

    return min_df[:, None] * freqs, max_df[:, None] * freqs     # (Nbls, Nfreqs) each


def sky_frates(freqs, bl_vecs, lat, buffer=None, hw_mult=1.0, min_hw=0.5):
    """
    Return sky fringe rate centers and half widths.

    Parameters
    ----------
    freqs : float or array-like
        Observing frequency/frequencies in Hz, shape (Nfreqs,).
    bl_vecs : array-like
        ENU (East, North, Up) baseline vector(s) in meters, shape (Nbls, 3); a single
        length-3 vector is promoted to (1, 3).
    lat : float
        Observer latitude in degrees.
    buffer : float
        Additive frate [mHz] to min/max range.
    hw_mult : float 
        Multiplier of computed half width
    min_hw : float
        Minimum threshold for computed half widths

    Returns
    -------
    frate_centers : float
        Fringe rate centers [mHz]
    frate_half_widths : float
        Fringe rate half widths [mHz]
    """
    # get (Nbls, Nfreqs) range of sky frates
    min_fr, max_fr = sky_frate_range(freqs, bl_vecs, lat)

    # add buffer
    if buffer is not None:
        # check frate conjugation: True/False for +/- frate_center
        sign = min_fr[:, :1] < max_fr[:, :1]
        sign = torch.as_tensor(sign, dtype=min_fr.dtype) * 2 - 1

        # add buffer to frate ranges
        min_fr -= buffer * sign
        max_fr += buffer * sign

    # compute centers and half widths
    frate_centers = (max_fr.amax(dim=-1) + min_fr.amin(dim=-1)) / 2
    frate_half_widths = abs(max_fr.amax(dim=-1) - min_fr.amin(dim=-1)) / 2 * hw_mult
    frate_half_widths = frate_half_widths.clamp_min(min_hw)

    return frate_centers, frate_half_widths


def zenith_frate(freqs, bl_vecs, latitude):
    """
    Fringe rate of a source at zenith (mHz) for one or more baselines.

    For a zenith-pointing (drift-scan) array this is the main-lobe peak fringe rate: the
    primary beam is centered on zenith, so most main-lobe sky power arrives at the zenith
    fringe rate. Only the East baseline component fringes at zenith, at
    ``f = (nu/c) * Omega_earth * cos(lat) * b_E``. Same convention/units as
    ``sky_frate_range``. (The main-lobe *width* needs a primary-beam model; this is the
    center only.)

    Parameters
    ----------
    freqs : float or array-like
        Observing frequency/frequencies in Hz, shape (Nfreqs,).
    bl_vecs : array-like
        ENU baseline vector(s) in meters, shape (Nbls, 3); a length-3 vector -> (1, 3).
    latitude : float
        Array latitude in degrees.

    Returns
    -------
    torch.Tensor
        Zenith fringe rate, shape (Nbls, Nfreqs), in mHz.
    """
    freqs  = torch.atleast_1d(torch.as_tensor(freqs))
    bl_vecs = torch.atleast_2d(torch.as_tensor(bl_vecs))
    b_E = bl_vecs[:, 0]                                    # only East fringes at zenith
    df = b_E * np.cos(np.radians(latitude)) * 2 * np.pi / (_SDAY_SEC * 1e-3) / _SPEED_OF_LIGHT

    return df[:, None] * freqs


def sky_delay(bl_vecs, theta=0.0, buffer=0.0, min_delay=50):
    """
    Sky (horizon) delay-filter half-width in ns, for one or more baselines.

    The delay filter is centered at delay = 0, so this single value is the half-width:
    the filter spans ``[-delays, +delays]``. The largest geometric delay that sky
    emission can produce is the baseline "horizon" delay ``|b| / c`` (the baseline
    light-crossing time); restricting the sky to elevations at least ``theta`` above the
    horizon scales that by ``cos(theta)``. A ``buffer`` (ns) is then added beyond the
    horizon and the result is floored at ``min_delay`` (ns) for short baselines.

    Unlike fringe rate, the geometric delay is independent of frequency, so ``freqs`` is
    accepted only for signature symmetry with :func:`sky_frate_range` and is not used.

    Parameters
    ----------
    bl_vecs : array-like
        ENU baseline vector(s) in meters, shape (Nbls, 3); a length-3 vector -> (1, 3).
    theta : float, optional
        Elevation cutoff in degrees, measured up from the horizon: 0 = full horizon
        (largest delay), 90 = zenith only (zero delay). Default 0.
    buffer : float, optional
        Extra half-width added beyond the horizon delay, in ns. Default 0.
    min_delay : float, optional
        Minimum half-width in ns (floor for short baselines). Default 50.

    Returns
    -------
    torch.Tensor
        Delay-filter half-width per baseline, shape (Nbls,), in ns.
    """
    bl_vecs = torch.atleast_2d(torch.as_tensor(bl_vecs))
    horizon = bl_vecs.norm(dim=-1) / _SPEED_OF_LIGHT * 1e9        # |b| / c, in ns
    delays = horizon * np.cos(np.radians(theta)) + buffer

    return delays.clamp(min=min_delay)

