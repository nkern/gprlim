"""
Implementation of GP inpainting pipeline for radio data sets
"""
import numpy as np
import torch
from copy import deepcopy

from . import models, kernels, solvers, utils


def hera_inpaint(
    data,
    noise_var,
    flags,
    times,
    freqs,
    inpaint='2d_1d',
    norm_freq_alpha=-2.0,
    time_kernel=None,
    freq_kernel=None,
    bl_vec=None,
    lat=None,
    fr_buffer=1.0,
    fr_scale=3e-3,
    fz_scale=1e-1,
    pf_scale=1e-1,
    wd_scale=1e-3,
    lk_scale=1e-3,
    lk_buffer=150.0,
    kernel_var_mult=3,
    noise_mult=100,
    precond_2d='sparse_separable',
    sparse_rcond=1e-12,
    red_avg_2d=True,
    cg_tol=1e-3,
    n_threads=16,
    rcond_1d=1e-12,
    flag_Ntimes=50,
    rf_scale=None,
    rf_tau=None,
    rf_width=None,
    ):
    """
    All purpose inpainting function for drift-scan, redundant radio visibilities.

    Builds the time and frequency kernels (via :func:`build_kernels`, unless pre-built ones
    are passed), optionally normalizes by a spectral powerlaw, runs the chosen inpaint mode,
    and returns the inpainted data and model.

    Parameters
    ----------
    data : tensor
        Complex visibility data of shape (Nbls, Ntimes, Nfreqs).
        Baseline axis should hold redundant baselines.
    noise_var : tensor
        Visibility noise variance broadcastable with data.
        Flagged pixels must have a large noise variance.
    flags : tensor
        Visibility flags to inpaint over.
    times : tensor
        Time integration of visibilities along Ntimes [seconds]
    freqs : tensor
        Frequency bins of visibilities along Nfreqs [MHz]
    inpaint : str, optional
        Inpainting method. ['2d', '2d_1d', '1d_1d', 'time', 'freq'].
            '2d'    : time + freq 2d inpaint
            '2d_1d' : 2d inpaint, then 1d freq inpaint
            '1d_1d' : 1d time inpaint, then 1d freq inpaint
            'time'  : 1d time inpaint
            'freq'  : 1d freq inpaint
    norm_freq_alpha : float, optional
        If provided, normalize input data and noise amplitude
        by a powerlaw with this spectral index (default=-2.0)
        before inpainting, un-normalize before returning.
        Pass as None or 1.0 for no effect.
    time_kernel : Kernel
        Pre-built time kernel. Supersedes build_kernels() parameters.
    freq_kernel : Kernel
        Pre-built freq kernel. Supersedes build_kernels() parameters.
    bl_vec : tensor
        Baseline vector (E, N, U) of this redundnat set [meters].
    lat : float
        Latitude of observer in degrees
    fr_buffer, fr_scale, fz_scale, pf_scale, wd_scale, lk_scale, lk_buffer, kernel_var_mult : optional
        Time / frequency kernel shape amplitudes and the fit-variance multiple, forwarded to
        :func:`build_kernels` (used only when ``time_kernel`` / ``freq_kernel`` are not given).
    red_avg_2d : bool, optional
        For the 2D modes, run the 2D stage on the gain-normalized redundant-baseline average
        (default True). See :func:`inpaint_2d_then_1d_freq`.
    noise_mult : float, optional
        Noise-inflation factor applied to first-guess-filled pixels so they act as a soft
        prior in the following 1D stage. Default 100.
    precond_2d : str, optional
        CG preconditioner for the 2D stage: 'sparse_separable' (default), 'separable' or 'scalar'.
    sparse_rcond : float, optional
        Eigenvalue cutoff for ``precond_2d='sparse_separable'``. Default 1e-12.
    cg_tol : float, optional
        CG tolerance for the 2D stage. Default 1e-3.
    n_threads : int, optional
        Threads for the batched 2D CG solve. Default 16.
    rcond_1d : float, optional
        Low-rank / truncation cutoff for the 1D solvers. Default 1e-12.
    flag_Ntimes : int, optional
        Channels with a contiguous flagged-time run of at least this many samples are fully
        flagged before a 1D time inpaint (``'1d_1d'`` / ``'time'``). Default 50.
    rf_scale : float or list, optional
        Reflection term amplitude(s)
    rf_tau : float or list, optional
        Reflection term delay(s) [micro-sec]
    rf_width : float or list, optional
        Reflection term delay width(s) [micro-sec]

    Returns
    -------
    inp_y : tensor
        ``data`` with flagged pixels replaced by the inpaint model, good pixels untouched;
        same shape as ``data``.
    mdl : tensor
        The full posterior-mean model from the final inpaint stage.
    """
    ## first normalize data if needed
    if norm_freq_alpha is not None:
        scale = (freqs / freqs.mean())**norm_freq_alpha
        data = data / scale
        noise_var = noise_var / scale**2

    if (freq_kernel is None) or (time_kernel is None):
        # build kernels
        time_kernel, freq_kernel = build_kernels(
            data, noise_var, flags, bl_vec, lat, times, freqs,
            inpaint=inpaint, fr_buffer=fr_buffer, fr_scale=fr_scale, fz_scale=fz_scale,
            pf_scale=pf_scale, wd_scale=wd_scale, lk_scale=lk_scale, lk_buffer=lk_buffer,
            kernel_var_mult=kernel_var_mult, norm_freq_alpha=None,
            rf_scale=rf_scale, rf_tau=rf_tau, rf_width=rf_width,
        )

    ## now run the inpainting
    if inpaint in ['2d', '2d_1d']:
        freq_1d = inpaint == '2d_1d'

        inp_y, mdl = inpaint_2d_then_1d_freq(
            time_kernel, freq_kernel, times, freqs, data, noise_var, flags,
            method_2d='cg', precond=precond_2d, sparse_rcond=sparse_rcond,
            red_avg_2d=red_avg_2d, cg_tol=cg_tol, cg_max_iter=10000,
            n_threads=n_threads, method_1d='woodbury', rcond_1d=rcond_1d,
            freq_1d=freq_1d, noise_mult=noise_mult,
            )

    elif inpaint in ['1d_1d', 'time', 'freq']:
        freq_1d = inpaint in ['1d_1d', 'freq']
        time_1d = inpaint in ['1d_1d', 'time']

        inp_y, mdl = inpaint_time_1d_then_freq_1d(
            time_kernel, freq_kernel, times, freqs, data, noise_var, flags,
            method='woodbury', rcond=rcond_1d, noise_mult=noise_mult,
            flag_Ntimes=flag_Ntimes, freq_1d=freq_1d, time_1d=time_1d
            )

    # re-normalize data if needed
    if norm_freq_alpha is not None:
        inp_y *= scale
        mdl *= scale

    return inp_y, mdl


def build_kernels(
    data,
    noise_var,
    flags,
    bl_vec,
    lat,
    times,
    freqs,
    inpaint='2d_1d',
    fr_buffer=1.0,
    fr_scale=3e-3,
    fz_scale=1e-1,
    pf_scale=1e-1,
    wd_scale=1e-3,
    lk_scale=1e-3,
    lk_buffer=150.0,
    iters=0,
    kernel_var_mult=3,
    flip_sign=False,
    norm_freq_alpha=None,
    only_amp=True,
    parameter=False,
    rf_scale=None,
    rf_tau=None,
    rf_width=None
    ):
    """
    Build the time (fringe-rate) and frequency (delay) kernels for drift-scan radio
    visibilities and fit their amplitude (and, with ``iters > 0``, shape) to the data.

    The kernels are :func:`gprlim.kernels.default_time_kernel` and
    :func:`gprlim.kernels.default_freq_kernel`; their variance is matched to the data variance
    with :func:`gprlim.models.fit_axis_kernel_2d` (a composite 2D fit, for the ``'2d'`` /
    ``'2d_1d'`` modes) or two 1D :func:`gprlim.models.fit_axis_kernel` fits (otherwise).

    Parameters
    ----------
    data : tensor
        Complex visibility data, shape (Nbls, Ntimes, Nfreqs).
    noise_var : tensor
        Visibility noise variance, broadcastable with ``data`` (large on flagged pixels).
    flags : tensor
        Boolean flags (True where flagged); used only to exclude pixels from the fit.
    bl_vec : tensor
        ENU baseline vector (E, N, U) of the redundant set [meters].
    lat : float
        Observer latitude [degrees].
    times : tensor
        Time grid along Ntimes [seconds].
    freqs : tensor
        Frequency grid along Nfreqs [MHz].
    inpaint : str, optional
        Which inpaint mode the kernels are for -- selects a composite 2D fit
        (``'2d'`` / ``'2d_1d'``) or two 1D fits (``'1d_1d'`` / ``'time'`` / ``'freq'``).
        Default '2d_1d'.
    fr_buffer : float, optional
        Extra fringe-rate half-width [mHz] added to the time kernel's sky-frate band.
    fr_scale, fz_scale : float, optional
        Amplitudes of the time kernel's full-FR-band (Sinc) and FR=0 (RBF) components,
        relative to its main lobe.
    pf_scale, wd_scale, lk_scale : float, optional
        Amplitudes of the frequency kernel's pitchfork, wedge and supra-horizon leakage
        components, relative to its main lobe.
    lk_buffer : float, optional
        Extra delay [ns] beyond the horizon for the frequency kernel's leakage component.
    iters : int, optional
        Optimizer iterations for the kernel fit. Default 0 (amplitude-only via the empirical
        rescale, no shape optimization).
    kernel_var_mult : float, optional
        Multiple of the data variance to put into the fitted kernel amplitude. Default 3.
    flip_sign : bool, optional
        Flip the sign of the time kernel's fringe-rate carriers (opposite baseline
        conjugation convention). Default False.
    norm_freq_alpha : float, optional
        If given, divide ``data`` (and ``noise_var`` by its square) by the spectral powerlaw
        ``(freqs / freqs.mean()) ** norm_freq_alpha`` before fitting. Default None (no effect).
    only_amp : bool, optional
        If True (default) freeze every kernel shape parameter, leaving only the amplitudes
        free (forwarded to the ``default_*_kernel`` builders).
    parameter : bool, optional
        If True keep the kernels' parameters attached; if False (default) detach them.
    rf_scale : float or list, optional
        Reflection term amplitude(s)
    rf_tau : float or list, optional
        Reflection term delay(s) [micro-sec]
    rf_width : float or list, optional
        Reflection term delay width(s) [micro-sec]

    Returns
    -------
    time_kernel : Kernel
    freq_kernel : Kernel
    """
    ## first build the kernels
    time_kernel = kernels.default_time_kernel(
        freqs*1e6, bl_vec, lat, ml_scale=1e0, fz_scale=fz_scale, fr_scale=fr_scale,
        buffer=fr_buffer, only_amp=only_amp, parameter=parameter, flip_sign=flip_sign,
    )

    freq_kernel = kernels.default_freq_kernel(
        bl_vec, ml_scale=1e0, pf_scale=pf_scale, wd_scale=wd_scale, lk_scale=lk_scale, 
        pf_real=True, lk_kern='twinrbf', buffer=lk_buffer, min_delay=50.0,
        only_amp=only_amp, parameter=parameter,
        rf_scale=rf_scale, rf_tau=rf_tau, rf_width=rf_width,
    )

    if norm_freq_alpha is not None:
        scale = (freqs / freqs.mean())**norm_freq_alpha
        data = data / scale
        noise_var = noise_var / scale**2

    ## now fit the overal kernel variance to the data variance
    if inpaint in ['2d', '2d_1d']:
        # do a composite kernel fit
        time_kernel, freq_kernel = models.fit_axis_kernel_2d(
            data, flags, noise_var, times, freqs, time_kernel, freq_kernel, iters=iters,
            prior_draws=100, rescale=True, var_mult=kernel_var_mult,
        )

    elif inpaint in ['1d_1d', 'time', 'freq']:
        # do a 1D fit for each kernel
        time_kernel = models.fit_axis_kernel(
            data, flags, noise_var, times, time_kernel, iters=iters, rescale=True, 
            prior_draws=100, var_mult=kernel_var_mult,
        )
        freq_kernel = models.fit_axis_kernel(
            data, flags, noise_var, freqs, freq_kernel, iters=iters, rescale=True,
            prior_draws=100, var_mult=kernel_var_mult,
        )

    else:
        raise ValueError

    return time_kernel, freq_kernel


def inpaint_2d_then_1d_freq(
    time_kernel,
    freq_kernel,
    times,
    freqs,
    data,
    noise,
    flags,
    method_2d='cg',
    method_1d='woodbury',
    precond='separable',
    red_avg_2d=True,
    noise_mult=100,
    rcond_2d=1e-13,
    rcond_1d=1e-13,
    cg_tol=1e-3,
    cg_max_iter=1000,
    n_threads=1,
    sparse_rcond=1e-12,
    freq_1d=True,
    ):
    """
    Two-stage GP inpaint for one redundant baseline group: a joint 2D (time-frequency)
    inpaint of the group's averaged spectrum to build a baseline model, then a per-baseline
    1D frequency inpaint seeded by that model.

    With ``red_avg_2d`` (the default) the redundant baselines are gain-normalized and
    inverse-variance averaged into a single high-SNR spectrum, the 2D inpaint runs once on
    that average, and its model is re-inflated by each baseline's gain and written into the
    flagged pixels. That 2D result then seeds a per-baseline 1D frequency inpaint: the
    2D-filled pixels enter as a *soft* prior (their noise inflated by ``noise_mult``) while any
    fully-flagged channels are re-filled from the frequency kernel. Without ``red_avg_2d`` the
    2D inpaint is run on ``data`` directly.

    Parameters
    ----------
    time_kernel, freq_kernel : gpytorch.kernels.Kernel
        Factor kernels over the time (outer) and frequency (inner) axes; ``time_kernel`` may
        be complex (e.g. a CarrierKernel). Forwarded to the 2D / 1D inpaint solvers.
    times, freqs : tensor
        1D time and frequency grids, shapes (Ntimes,) and (Nfreqs,).
    data : tensor
        Complex data of shape (Nbls, Ntimes, Nfreqs) for a single redundant set.
    noise : tensor
        Per-pixel noise variance, broadcastable with ``data`` (flagged pixels carry a large
        value); its reciprocal is used as the redundant-average weight.
    flags : tensor
        Boolean flags (True where flagged / to inpaint), broadcastable with ``data``.
    method_2d : str, optional
        Solver for the 2D stage: 'cg' (default), 'woodbury' or 'cholesky' (see
        :func:`gprlim.models.posterior_mean_2d`).
    method_1d : str, optional
        Solver for the 1D frequency stage: 'woodbury' (default) or 'cholesky' (see
        :func:`gprlim.models.posterior_mean_1d`).
    precond : str, optional
        CG preconditioner for the 2D stage: 'separable' (default), 'scalar', or
        'sparse_separable'; used only when ``method_2d='cg'``.
    red_avg_2d : bool, optional
        If True (default), run the 2D stage on the gain-normalized redundant average; if
        False, run it on ``data`` directly.
    noise_mult : float, optional
        Noise-inflation factor for the 2D-filled (partially-flagged) pixels before the 1D
        frequency stage, so the 2D first guess enters as a soft prior rather than hard data.
        Default 100.
    rcond_2d, rcond_1d : float, optional
        Low-rank / truncation cutoff (``rcond``) for the 2D and 1D solvers. Default 1e-13.
    cg_tol, cg_max_iter, n_threads
        CG controls for the 2D stage (``method_2d='cg'``), forwarded to
        :func:`gprlim.models.inpaint_2d`.
    sparse_rcond : float, optional
        Eigenvalue cutoff for ``precond='sparse_separable'`` in the 2D stage (ignored
        otherwise). Default 1e-12.
    freq_1d : bool, optional
        Apply 1D frequency inpainting after 2D inpaint

    Returns
    -------
    inp_y : tensor
        ``data`` with flagged pixels replaced by the inpaint model, good pixels untouched;
        shape (Nbls, Ntimes, Nfreqs).
    mdl : tensor
        The full frequency posterior-mean model from the final 1D stage.
    """
    if red_avg_2d:
        # first get a per-baseline gain
        ## TODO: allow for spectral tilt
        wgts = 1 / noise.clip(1e-40)
        G = (data.abs() * wgts).sum(dim=(-1, -2), keepdim=True)
        G /= wgts.expand(data.shape).sum(dim=(-1, -2), keepdim=True)
        G /= G.mean(dim=-3, keepdim=True)

        # divide by gain and average across baselines
        wsum = wgts.expand(data.shape).sum(dim=-3, keepdim=True)
        y0 = ((data / G) * wgts).sum(dim=-3, keepdim=True) / wsum
        noise0 = 1.0 / wsum                                  # variance of the weighted average
        flags0 = flags.all(dim=-3, keepdim=True)             # flagged in the average iff all-flagged

    else:
        y0, noise0, flags0 = data, noise, flags

    # perform 2D inpainting as a first guess
    inp_y, mdl = models.inpaint_2d(
        time_kernel, freq_kernel, times, freqs, y0, noise0, flags0,
        method=method_2d, rcond=rcond_2d, cg_tol=cg_tol, cg_max_iter=cg_max_iter,
        n_threads=n_threads, precond=precond, sparse_rcond=sparse_rcond,
    )

    if red_avg_2d:
        # re-inflate to data size and gain level
        inp_y = torch.where(flags, inp_y * G, data)

    if not freq_1d:
        return inp_y, mdl

    # now use it as prior for freq only inpaint: keep fully flagged chans flagged
    noise = noise.clone()
    all_flags = (flags.all(dim=-2, keepdim=True)).expand_as(flags)
    noise[flags & ~all_flags] = noise[~flags].mean() * noise_mult

    # inpaint along frequency 
    inp_y, mdl = models.inpaint_1d(
        freq_kernel, freqs, inp_y, noise, flags, dim=-1, method=method_1d, rcond=rcond_1d,
    )

    return inp_y, mdl


def inpaint_time_1d_then_freq_1d(
    time_kernel,
    freq_kernel,
    times,
    freqs,
    data,
    noise,
    flags,
    method='woodbury',
    rcond=1e-13,
    noise_mult=100,
    flag_Ntimes=50,
    time_1d=True,
    freq_1d=True,
    **kwargs
    ):
    """
    Two-stage per-baseline GP inpaint: a 1D inpaint along time to fill the time gaps, then a
    1D inpaint along frequency seeded by the time model.

    Channels with a contiguous run of at least ``flag_Ntimes`` flagged time samples are first
    fully flagged (a long time gap makes the time inpaint unreliable) and are instead filled by
    the frequency stage. The time inpaint fills the remaining time-flagged pixels; those enter
    the frequency inpaint as a *soft* prior (noise inflated by ``noise_mult``), while fully-flagged
    channels are re-filled from the frequency kernel.

    Parameters
    ----------
    time_kernel, freq_kernel : gpytorch.kernels.Kernel
        Factor kernels over the time and frequency axes (forwarded to
        :func:`gprlim.models.inpaint_1d`).
    times, freqs : tensor
        1D time and frequency grids, shapes (Ntimes,) and (Nfreqs,).
    data : tensor
        Complex data of shape (..., Ntimes, Nfreqs) (time at ``dim=-2``, frequency at ``dim=-1``).
    noise : tensor
        Per-pixel noise variance, broadcastable with ``data`` (flagged pixels carry a large value).
    flags : tensor
        Boolean flags (True where flagged), broadcastable with ``data``.
    method : str, optional
        Solver for both 1D inpaints: 'woodbury' (default) or 'cholesky' (see
        :func:`gprlim.models.posterior_mean_1d`).
    rcond : float, optional
        Low-rank / truncation cutoff for the 1D solvers. Default 1e-13.
    noise_mult : float, optional
        Noise-inflation factor for the time-filled (partially-flagged) pixels before the
        frequency stage, so the time first guess enters as a soft prior. Default 100.
    flag_Ntimes : int, optional
        Channels with a contiguous flagged-time run of at least this many samples are fully
        flagged before the time inpaint (and filled by the frequency stage instead). Default 50.
    time_1d : bool, optional
        If True (default) run time 1d inpainting
    freq_1d : bool, optional
        If True (default) run freq 1d inpainting
    **kwargs
        Forwarded to both :func:`gprlim.models.inpaint_1d` calls.

    Returns
    -------
    inp_y : tensor
        ``data`` with flagged pixels replaced by the inpaint model; good pixels untouched.
    mdl : tensor
        The full frequency posterior-mean model from the final 1D stage.
    """
    if time_1d:
        # completely flag channels that have contiguous flags > flag_Ntimes
        # b/c this causes time inpainting to fail
        if flag_Ntimes is not None:
            chan_flags = utils.detect_contiguous(flags, flag_Ntimes, dim=-2)
            flags = flags | chan_flags

        # perform time inpainting as a first guess
        inp_y, mdl = models.inpaint_1d(
            time_kernel, times, data, noise, flags, dim=-2, method=method, rcond=rcond, **kwargs
        )

        # now use it as prior for freq inpaint: keep fully flagged chans flagged
        noise = noise.clone()
        all_flags = (flags.all(dim=-2, keepdim=True)).expand_as(flags)
        noise[flags & ~all_flags] = noise[~flags].mean() * noise_mult

    else:
        inp_y, mdl = data, None

    if not freq_1d:
        return inp_y, mdl

    # inpaint along frequency 
    inp_y, mdl = models.inpaint_1d(
        freq_kernel, freqs, inp_y, noise, flags, dim=-1, method=method, rcond=rcond, **kwargs
    )

    return inp_y, mdl


def plot_kernel_match(
    data, time_kernel, freq_kernel, times, freqs,
    pol=None, lst_range=None, bl_vec=None, axes=None,
    ft_freq=None, ft_time=None,
    ):
    """
    Plot the match between the kernel and (inpainted) data in Fourier space

    Parameters
    ----------
    data : tensor
        Complex (inpainted) data visibilities of shape (Nbls, Ntimes, Nfreqs)
    time_kernel : Kernel
    freq_kernel : Kernel
    times : tensor
        Time array in seconds
    freqs : tensor
        Frequency array [MHz]
    pol : str, optional
        Polarization of data
    lst_range : tuple, optional
        Range of LSTs [hours] (start, stop)
    bl_vec : tensor, optional
        Baseline vector of data in ENU [meters]
    axes : matplotlib.Axes
    ft_freq : bayeslim.fft.FFT
    ft_time : bayeslim.fft.FFT
    """
    import matplotlib.pyplot as plt
    if lst_range is None:
        lst_range = (0, 0)
    if bl_vec is None:
        bl_vec = [0, 0]
    label = f"{pol} pol | {lst_range[0]:.1f}-{lst_range[1]:.1f} hrs LST | "\
            f"{freqs[0]:.1f}-{freqs[-1]:.1f} MHz | {bl_vec[0]:.1f}, {bl_vec[1]:.1f} [m] bl"

    if (ft_freq is None) or (ft_time is None):
        from bayeslim.fft import FFT
        if ft_freq is None:
            ft_freq = FFT(dim=-1, N=len(freqs), ndim=2, window='bh', dx=freqs.diff()[0]/1e3)
        if ft_time is None:
            ft_time = FFT(dim=-2, N=len(times), ndim=2, window='bh', dx=times.diff()[0]/1e3)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    flags = torch.zeros(data.shape, dtype=bool)
    noise = torch.ones(data.shape)

    # rescale each kernel to data variance
    time_kernel = models.fit_axis_kernel(
        data, flags, noise, times, deepcopy(time_kernel), iters=0, rescale=True, 
        prior_draws=100, var_mult=3,
    )
    freq_kernel = models.fit_axis_kernel(
        data, flags, noise, freqs, deepcopy(freq_kernel), iters=0, rescale=True,
        prior_draws=100, var_mult=3,
    )

    # draw 1D samples each
    samples = models.prior_draws_1d(freq_kernel, freqs, size=100, jitter=1e-10)
    axes[0].plot(ft_freq.freqs, ft_freq(data.mean(0)).abs().mean(0), label='data', c='k')
    axes[0].plot(ft_freq.freqs, ft_freq(samples).abs().mean(0), label='samples', c='indianred');
    axes[0].set_xlim(-3000, 3000); axes[0].set_xlabel('delay [nanosec]'); axes[0].set_yscale('log');
    axes[0].grid()
    axes[0].set_title(label, fontsize=9)

    samples = models.prior_draws_1d(time_kernel, times, size=100, jitter=1e-10)[...,None]
    axes[1].plot(ft_time.freqs, ft_time(data.mean(0)).abs().mean(1), label='data', c='k')
    axes[1].plot(ft_time.freqs, ft_time(samples).abs().mean(0)[...,0], label='samples', c='indianred');
    axes[1].set_xlim(-15, 15); axes[1].set_xlabel('fringe-rate [mHz]'); axes[1].set_yscale('log');
    axes[1].grid()
    axes[1].set_title(label, fontsize=9)

    return axes




