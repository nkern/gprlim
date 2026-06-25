import numpy as np
import torch
import gpytorch
import warnings

from .models import fixednoise_gp_1d, optimize_kernel, gpr_invert
from .kernels import default_freq_kernel, default_time_kernel
from .solvers import kron_woodbury_predict, kron_wiener_cg

from gpytorch.means import ZeroMean
from linear_operator import to_linear_operator
from linear_operator.operators import DiagLinearOperator, KroneckerProductLinearOperator

try:
	from pyuvdata import UVData, UVCal, UVFlag, utils as uvutils
except:
	warnings.warn("Couldn't import pyuvdata")

try:
	from hera_cal.frf import sky_frates
	from hera_cal.redcal import get_pos_reds
except:
	warnings.warn("Couldn't import hera_cal")


def hera_freq_inpaint(
	data,
	flags,
	x,
	dim=-1,
	y_scale=None

):
	"""
	Inpaint flagged pixels along a single axis with a fixed-noise GP (work in progress).

	An earlier single-axis sketch, superseded by :func:`inpaint`; retained for reference
	and not currently functional (it references quantities it does not define).

	Parameters
	----------
	data : tensor
	    Complex or real data; the axis to inpaint over is ``dim``.
	flags : tensor
	    Boolean flags matching ``data`` (True where flagged).
	x : tensor
	    Coordinate grid along the inpaint axis, shape (Nx,).
	dim : int, optional
	    Axis of ``data`` to inpaint over.
	y_scale : tensor, optional
	    Pre-scaling applied to the data before fitting and undone afterward.
	"""
	# get data shape
	shape = data.shape
	Nx = shape[dim]
	data = data.moveaxis(dim, -1)  # this is a view
	flags = flags.moveaxis(dim, -1)

	if y_scale is None:
		y_scale = torch.tensor(1.0)

	# get flag info
	all_flags = flags.all(dim=dim, keepdim=True)
	inp_flags = all_flags ^ flags # where to inpaint

	# setup training targets and samples
	if data.is_complex():
		train_y = torch.vstack([data.real.reshape(-1, Nx), data.imag.reshape(-1, Nx)]) / y_scale
		train_flags = torch.vstack([flags.reshape(-1, Nx), flags.reshape(-1, Nx)])
		train_inp_flags = torch.vstack([inp_flags.reshape(-1, Nx), inp_flags.reshape(-1, Nx)])
	else:
		train_y = data.reshape(-1, Nx) / y_scale
		train_flags = flags.reshape(-1, Nx)
		train_inp_flags = inp_flags.reshape(-1, Nx)

	train_x = x[None, :, None]

	# estimate robust signal variance
	y_med = torch.median(train_y[~train_flags])
	y_mad = torch.median((train_y[~train_flags] - dmed).abs()) * 1.48  # robust measure
	y_var = dmad**2

	# subtract off noise variance (1/2 for real/imag)
	y_var -= sigma_N[~flags].pow(2).median() / 2

	# setup noise weights (variance)
	noise_wgts = torch.ones_like(data.real) * sigma_N**2 / pvar
	noise_wgts[flags] = sigma_N.median().pow(2) * 1e8

	noise_wgts[..., 10:20, 85:130] = sigma_N[..., 10:20, 85:130]**2 / pvar * 1

	noise_wgts = torch.vstack([noise_wgts.reshape(-1, Nfreqs), noise_wgts.reshape(-1, Nfreqs)])

	# get GPModel
	model, y_offset = fixednoise_gp_1d(train_x, train_y, mean, covar, inv_wgts=noise_wgts, center_y=True, Ndeg=1)
	y_offset = y_offset if y_offset is not None else 0

	# inpaint
	inp_m = model.inpaint(None, y_offset=y_offset, y_scale=pvar.sqrt(), to_complex=True, return_model=True).reshape(-1, Ntimes, Nfreqs)
	inp_y = model.inpaint(train_inp_flags, y_offset=y_offset, y_scale=pvar.sqrt(), to_complex=True).reshape(-1, Ntimes, Nfreqs)


# ---------------------------------------------------------------------------
# GP inpainting workflow
# ---------------------------------------------------------------------------


def _to_3d(x):
    """
    Promote a 2-D array to 3-D by adding a leading baseline axis.

    Parameters
    ----------
    x : tensor
        Array of shape (Ntimes, Nfreqs) or (Nbls, Ntimes, Nfreqs).

    Returns
    -------
    tensor
        Array of shape (Nbls, Ntimes, Nfreqs); a 2-D input becomes (1, Ntimes, Nfreqs).
    """
    return x.unsqueeze(0) if x.dim() == 2 else x


def _broadcast_noise(noise, Nbls):
    """
    Broadcast a noise (variance) array to one value per baseline.

    Parameters
    ----------
    noise : tensor
        Noise variance of shape (Ntimes, Nfreqs), (1, Ntimes, Nfreqs) or (Nbls, Ntimes, Nfreqs). A 2-D array or
        a leading axis of 1 is treated as shared across baselines.
    Nbls : int
        Number of baselines to broadcast to.

    Returns
    -------
    tensor
        Noise variance of shape (Nbls, Ntimes, Nfreqs).
    """
    noise = _to_3d(torch.as_tensor(noise))
    if noise.shape[0] == 1:
        noise = noise.expand(Nbls, -1, -1)
    return noise.contiguous()


def _stack_ri(x):
    """
    Stack the real and imaginary parts of complex data along the row (batch) axis so a
    real GP can model both; real input is returned unchanged.

    Parameters
    ----------
    x : tensor
        Data of shape (Nrows, Nx), real or complex.

    Returns
    -------
    tensor
        ``x`` if real, else shape (2 * Nrows, Nx) with the real part stacked above the
        imaginary part.
    """
    return torch.cat([x.real, x.imag], 0) if x.is_complex() else x


def fit_axis_kernel(data, flags, noise, x, kernel, nsamp=512, iters=5,
                    opt='LBFGS', method='cholesky'):
    """
    Fit a kernel's hyperparameters by marginal likelihood along one axis.

    Pools the rows (real/imag stacked for a real covariance, kept complex for a complex
    one) and fits the kernel IN PLACE. Rows are drawn from the most-complete ones first so
    a few flagged pixels don't bias the fit; ``nsamp`` caps the fit batch.

    Parameters
    ----------
    data : tensor
        Data rows of shape (Nrows, Nx), real or complex.
    flags : tensor
        Boolean flags of shape (Nrows, Nx) (True where flagged); used only to rank rows by
        completeness for the fit.
    noise : tensor
        Noise variance of shape (Nrows, Nx); flagged pixels are assumed already
        down-weighted (large variance) by the caller.
    x : tensor
        Axis grid of shape (Nx,).
    kernel : gpytorch.kernels.Kernel
        Kernel to fit; modified in place.
    nsamp : int, optional
        Maximum number of (most-complete) rows used in the fit.
    iters : int, optional
        Number of optimizer iterations.
    opt : str or torch.optim.Optimizer, optional
        Optimizer passed to ``optimize_kernel`` (e.g. 'LBFGS', 'Adam').
    method : str, optional
        Marginal-likelihood solver, 'cholesky' (stable) or 'woodbury'.

    Returns
    -------
    gpytorch.kernels.Kernel
        The same ``kernel``, fit in place.
    """
    # rank rows by completeness, keep the most-complete `nsamp` so a few flagged
    # pixels don't bias the fit (noise already down-weights them; flags only rank here)
    good = 1.0 - flags.float().mean(1)
    order = torch.argsort(good, descending=True)[:min(nsamp, data.shape[0])]
    s, nz = data[order], noise[order]

    # a REAL covariance acts identically on real & imag, so stack them as extra rows and
    # fit a single real GP (cheaper). A COMPLEX covariance (e.g. a CarrierKernel axis)
    # couples them, so fit the complex data directly via the complex marginal likelihood.
    cov_complex = kernel(x[:2, None]).to_dense().is_complex()
    if s.is_complex() and not cov_complex:
        ys, nzs = _stack_ri(s), torch.cat([nz, nz], 0)
    else:
        ys, nzs = s, nz

    # fit the hyperparameters IN PLACE by the marginal likelihood
    model, _ = fixednoise_gp_1d(x[None, :, None], ys, ZeroMean(), kernel, inv_wgts=nzs)
    optimize_kernel(model, Niter=iters, opt=opt, batched=method)

    return kernel


def _axis_inpaint(rows, rflags, iwgts, x, kernel, complex_in, center, Ndeg, method, unpack_complex):
    """
    Per-row GP inpaint along one axis.

    Each row of ``rows`` is a 1-D realization (a spectrum or a time series) sharing
    ``kernel`` over the grid ``x``; flagged pixels are filled by the batched Wiener solve.

    By default (``unpack_complex=False``) complex rows are solved directly -- a real
    ``kernel`` covariance is promoted to complex inside :class:`GPModel` (same answer, one
    complex solve). With ``unpack_complex=True`` the real and imaginary parts are instead
    stacked into the batch and solved as a real GP (two real solves), then recombined --
    cheaper, and valid because a real covariance acts identically on each part.

    Parameters
    ----------
    rows : tensor
        Data rows of shape (Nrows, Nx), real or complex.
    rflags : tensor
        Boolean flags of shape (Nrows, Nx) (True where flagged).
    iwgts : tensor
        Noise variance of shape (Nrows, Nx); flagged pixels are assumed already
        down-weighted (large variance) by the caller.
    x : tensor
        Axis grid of shape (Nx,).
    kernel : gpytorch.kernels.Kernel
        Covariance kernel over ``x``.
    complex_in : bool
        Whether ``rows`` is complex.
    center : bool
        If True, subtract a per-row polynomial baseline before the solve.
    Ndeg : int
        Polynomial degree used when ``center`` is True.
    method : str
        Batched solver for the posterior mean, 'cholesky' or 'woodbury'.
    unpack_complex : bool
        If True, stack real/imag of complex rows into the batch (a real-GP solve, then
        recombine); otherwise solve the complex system directly. Ignored (forced False)
        when the covariance is complex, which couples real and imag.

    Returns
    -------
    inp : tensor
        Inpainted rows of shape (Nrows, Nx) (flagged pixels filled, good pixels kept).
    model : GPModel
        The fixed-noise GP model used for the solve.
    """
    # stacking real/imag for a real GP is valid only for a REAL covariance (it acts
    # identically on each part); a complex covariance couples them, so honor
    # unpack_complex only when the kernel is real, else solve the complex system directly.
    cov_complex = kernel(x[:2, None]).to_dense().is_complex()
    if complex_in and unpack_complex and cov_complex:
        warnings.warn("unpack_complex=True ignored: the covariance is complex, so the "
                      "real/imag parts are coupled and the complex system is solved directly.")
    unpack = complex_in and unpack_complex and not cov_complex
    if unpack:
        # stack real/imag into the batch and solve as a real GP, then recombine
        train_y = _stack_ri(rows)
        train_fl = torch.cat([rflags, rflags], 0)
        train_iw = torch.cat([iwgts, iwgts], 0)
    else:
        # solve the data directly (GPModel promotes a real covariance to complex for
        # complex rows -- one complex solve, same result)
        train_y, train_fl, train_iw = rows, rflags, iwgts

    # batched Wiener fill K (K + diag(noise))^-1 y per row; replaces flagged pixels
    model, y_off = fixednoise_gp_1d(x[None, :, None], train_y, ZeroMean(), kernel,
                                    inv_wgts=train_iw, center_y=center, Ndeg=Ndeg)
    inp, _ = model.inpaint(train_fl, y_offset=y_off, unpack_complex=unpack, method=method)

    return inp, model


def inpaint(data, flags, times, freqs, noise, freq_kernel=None, time_kernel=None,
            bl_vecs=None, latitude=None,
            fit=True, mode='freq', center=True, Ndeg=1, fit_nsamp=512, fit_iter=50,
            fit_opt='LBFGS', method='woodbury', cg_tol=1e-8, cg_max_iter=1000, rcond=1e-12,
            unpack_complex=False, return_model=False):
    """
    GP-inpaint complex visibility data over flagged pixels.

    Every baseline is assumed drawn from the *same* covariance: one frequency kernel
    (delay structure) and -- in joint mode -- one time kernel (fringe-rate) are fit by
    marginal likelihood on the pooled, mostly-unflagged data, then applied to each
    baseline with its own noise. The supplied ``noise`` is assumed to already down-weight
    flagged pixels (large variance), so the parametric kernel fills them even where a whole
    channel or time integration is flagged (which a data-driven covariance could not);
    ``flags`` only marks where the inpaint model is written back into the data.

    Parameters
    ----------
    data : tensor
        Complex (or real) visibilities of shape (Ntimes, Nfreqs) or
        (Nbaselines, Ntimes, Nfreqs).
    flags : tensor
        Boolean flags (True where flagged), same trailing shape as ``data``. A 2-D array
        or a leading axis of 1, i.e. (1, Ntimes, Nfreqs), denotes a mask *shared* across
        all baselines; (Nbls, ...) is per-baseline. Used only to mark where the inpaint
        model is written back -- the down-weighting of flagged pixels must be encoded in
        ``noise``.
    times : tensor
        Time grid of shape (Ntimes,). In **seconds** when the default time kernel is built.
    freqs : tensor
        Frequency grid of shape (Nfreqs,). In **MHz** when a default kernel is built.
    noise : tensor
        Noise variance, broadcastable like ``flags`` ((Ntimes, Nfreqs), (1, Ntimes, Nfreqs) or
        (Nbls, Ntimes, Nfreqs); shared if 2-D or leading axis 1). Flagged pixels must
        already carry a large variance so the solve down-weights them.
    freq_kernel : gpytorch.kernels.Kernel, optional
        Frequency (delay) kernel. Default: built from baseline geometry via
        :func:`gprlim.kernels.default_freq_kernel` (needs ``bl_vecs``).
    time_kernel : gpytorch.kernels.Kernel, optional
        Time (fringe-rate) kernel, used by 'time' and 'joint' modes. Default: built from
        baseline geometry via :func:`gprlim.kernels.default_time_kernel` (needs
        ``bl_vecs`` and ``latitude``).
    bl_vecs : tensor, optional
        ENU baseline vector(s) in **meters**, shape (Nbls, 3) (or (3,)). Used only to
        build the default kernel(s) when ``freq_kernel``/``time_kernel`` is None.
    latitude : float, optional
        Array latitude in **degrees**. Used only to build the default time kernel.
    fit : bool, optional
        If True, fit the kernel hyperparameters on the data before inpainting.
    mode : {'freq', 'time', 'joint'}, optional
        Which inpaint to run (see Notes).
    center : bool, optional
        Subtract a per-row polynomial baseline before the 'freq'/'time' GP solve.
    Ndeg : int, optional
        Polynomial degree used when ``center`` is True.
    fit_nsamp : int, optional
        Maximum number of rows used to fit each kernel.
    fit_iter : int, optional
        Optimizer iterations per kernel fit.
    fit_opt : str or torch.optim.Optimizer, optional
        Optimizer for the kernel fit (e.g. 'LBFGS', 'Adam').
    method : str, optional
        Posterior-mean solver. For 'freq'/'time' modes: the batched 'cholesky' or
        'woodbury' solver. For 'joint' mode on a complex covariance, also selects the
        no-densify solver: 'woodbury' (default; direct, rank-truncated at ``rcond``),
        'cg' (preconditioned complex CG, no truncation, accurate to ``cg_tol``), or
        'cholesky' (densifies the full ``(Ntimes*Nfreqs)^2`` covariance -- exact, but
        only when it fits in memory).
    cg_tol : float, optional
        CG tolerance. Used by the joint 'cg' solver, and by the 'joint' structured real
        solve when ``Ntimes * Nfreqs > max_cholesky_size``.
    cg_max_iter : int, optional
        Maximum iterations for the joint 'cg' solver (raise for high-dynamic-range cases).
    rcond : float, optional
        Low-rank eigenvalue cutoff for the joint-mode 'woodbury' solver.
    unpack_complex : bool, optional
        How complex data is handled. If False (default) operate on the complex data
        directly -- a real covariance is promoted to complex (imag = 0) and the complex
        system is solved. If True, stack the real and imaginary parts and solve as a real
        GP (cheaper, but valid only for a real covariance, which acts identically on each).
    return_model : bool, optional
        If True, also return a dict of the fitted pieces.

    Returns
    -------
    out : tensor
        Inpainted data, same shape and dtype as ``data`` (good pixels untouched).
    model : dict
        Only returned if ``return_model`` is True: the fitted kernel(s) and the GP model
        ('freq'/'time') or Kronecker covariance ('joint').

    Notes
    -----
    Modes:

    - ``'freq'`` : per-time frequency inpaint (delay-space path), batched over
      (baseline, time) via the Woodbury/Cholesky solver -- the fast, science-critical
      default; fills RFI channels via frequency correlation.
    - ``'time'`` : per-channel time inpaint over the time axis, batched over
      (baseline, frequency); fills fully-flagged time integrations (which a per-time
      'freq' pass cannot) via time correlation.
    - ``'joint'`` : full 2-D inpaint with the separable Kronecker covariance
      ``K = Ct (x) Cf``, using the structured (never densified) operator. Solved per
      baseline, or -- when flags and noise are shared (passed as (1, Ntimes, Nfreqs)) -- for all
      baselines at once as columns of one batched solve (no per-baseline loop). For the
      frequency-evolving carrier, replace ``Ct (x) Cf`` with a sum over ``Ct_m (x) Cf_m``.
    """
    # ---- normalize everything to (Nbls, Ntimes, Nfreqs) ----
    data = torch.as_tensor(data)
    flags = torch.as_tensor(flags).bool()
    times = torch.as_tensor(times).reshape(-1).to(data.real.dtype)
    freqs = torch.as_tensor(freqs).reshape(-1).to(data.real.dtype)
    twod = data.dim() == 2                                  # remember 2-D input to squeeze on return
    d3 = _to_3d(data)
    Nbls, Ntimes, Nfreqs = d3.shape

    # (1, Ntimes, Nfreqs) or 2-D flags/noise are shared across baselines, (Nbls, ...) per-baseline;
    # record the sharing, then broadcast both to (Nbls, Ntimes, Nfreqs).
    f3 = _to_3d(flags)
    flags_shared = f3.shape[0] == 1
    f3 = f3.expand(Nbls, -1, -1).contiguous() if flags_shared else f3

    noise_in = torch.as_tensor(noise)
    noise_shared = _to_3d(noise_in).shape[0] == 1
    n3 = _broadcast_noise(noise_in, Nbls).to(data.real.dtype)

    shared = flags_shared and noise_shared                 # one joint solve covers all baselines

    # default kernels are built from the baseline geometry (kernels.default_*; needs
    # bl_vecs/latitude, freqs in MHz, times in seconds). A user-supplied kernel is used
    # as-is (not cast), preserving a complex covariance.
    if mode in ('freq', 'joint') and freq_kernel is None:
        if bl_vecs is None:
            raise ValueError("freq_kernel is None -- pass bl_vecs (meters) to build the "
                             "default frequency kernel, or supply freq_kernel.")
        freq_kernel = default_freq_kernel(bl_vecs).to(data.real.dtype)
    if mode in ('time', 'joint') and time_kernel is None:
        if bl_vecs is None or latitude is None:
            raise ValueError("time_kernel is None -- pass bl_vecs (meters) and latitude "
                             "(degrees) to build the default time kernel, or supply time_kernel.")
        time_kernel = default_time_kernel(freqs * 1e6, bl_vecs, latitude).to(data.real.dtype)

    if mode == 'freq':
        # rows = per-(baseline, time) spectra sharing the freq kernel
        rows = d3.reshape(-1, Nfreqs)
        rflags, iwgts = f3.reshape(-1, Nfreqs), n3.reshape(-1, Nfreqs)

        # fit the shared frequency kernel on the pooled, mostly-clean spectra
        if fit:
            fit_axis_kernel(rows, rflags, iwgts, freqs, freq_kernel,
                            nsamp=fit_nsamp, iters=fit_iter, opt=fit_opt)

        inp, model = _axis_inpaint(rows, rflags, iwgts, freqs, freq_kernel,
                                   data.is_complex(), center, Ndeg, method, unpack_complex)
        out = inp.reshape(d3.shape)                         # back to (Nbls, Ntimes, Nfreqs)
        out = out[0] if twod else out                      # drop baseline axis for 2-D input
        return (out, dict(freq_kernel=freq_kernel, model=model)) if return_model else out

    elif mode == 'time':
        # rows = per-(baseline, channel) time series sharing the time kernel; fills
        # fully-flagged integrations via time correlation
        rows = d3.permute(0, 2, 1).reshape(-1, Ntimes)
        rflags = f3.permute(0, 2, 1).reshape(-1, Ntimes)
        iwgts = n3.permute(0, 2, 1).reshape(-1, Ntimes)

        # fit the shared time kernel on the pooled, mostly-clean time series
        if fit:
            fit_axis_kernel(rows, rflags, iwgts, times, time_kernel,
                            nsamp=fit_nsamp, iters=fit_iter, opt=fit_opt)

        inp, model = _axis_inpaint(rows, rflags, iwgts, times, time_kernel,
                                   data.is_complex(), center, Ndeg, method, unpack_complex)
        out = inp.reshape(Nbls, Nfreqs, Ntimes).permute(0, 2, 1)   # (Nbls, Nfreqs, Ntimes) -> (Nbls, Ntimes, Nfreqs)
        out = out[0] if twod else out
        return (out, dict(time_kernel=time_kernel, model=model)) if return_model else out

    elif mode == 'joint':
        # fit both shared kernels: freq on the spectra, time on the transposed series
        if fit:
            fit_axis_kernel(d3.reshape(-1, Nfreqs), f3.reshape(-1, Nfreqs), n3.reshape(-1, Nfreqs),
                            freqs, freq_kernel, nsamp=fit_nsamp, iters=fit_iter, opt=fit_opt)
            fit_axis_kernel(d3.permute(0, 2, 1).reshape(-1, Ntimes),
                            f3.permute(0, 2, 1).reshape(-1, Ntimes),
                            n3.permute(0, 2, 1).reshape(-1, Ntimes), times, time_kernel,
                            nsamp=fit_nsamp, iters=fit_iter, opt=fit_opt)

        # Ks = Ct (x) Cf, structured (never densified; Ct is the slow factor). Promote to a
        # common dtype; unless unpack_complex, also promote a real covariance to complex for
        # complex data (-> complex solve below; else the real path splits real/imag).
        Ct = time_kernel(times[:, None]).to_dense().detach()
        Cf = freq_kernel(freqs[:, None]).to_dense().detach()
        cov_dtype = torch.promote_types(Ct.dtype, Cf.dtype)
        if data.is_complex() and not unpack_complex:
            cov_dtype = torch.promote_types(cov_dtype, data.dtype)
        Ct, Cf = Ct.to(cov_dtype), Cf.to(cov_dtype)
        Ks = KroneckerProductLinearOperator(to_linear_operator(Ct), to_linear_operator(Cf))

        # posterior mean  m = Ks (Ks + diag(noise))^-1 y  per baseline.
        cov_complex = Ks.dtype.is_complex
        with torch.no_grad(), gpytorch.settings.cg_tolerance(cg_tol):
            if cov_complex:
                # complex covariance needs a complex-capable solver (linear_operator's CG
                # is real-only): 'woodbury' (no densify, rank-truncated at rcond), 'cg' (no
                # densify, no truncation, to cg_tol), or 'cholesky' (densifies Ks). See docstring.
                if method == 'woodbury':
                    m = kron_woodbury_predict(Ct, Cf, n3, d3, rcond=rcond)
                elif method == 'cg':
                    m, _ = kron_wiener_cg(Ct, Cf, n3, d3, tol=cg_tol, max_iter=cg_max_iter)
                else:
                    Cs = Ks.to_dense()
                    m = gpr_invert(Cs, n3.reshape(Nbls, -1), B=Cs,
                                   y=d3.reshape(Nbls, -1), method='cholesky').reshape(Nbls, Ntimes, Nfreqs)
                out = torch.where(f3, m.to(d3.dtype), d3)
            else:
                # real covariance, structured solve (never densified; CG for large sizes).
                # Reached for real data, or complex data with unpack_complex=True -- which
                # splits real/imag into independent solves.
                parts = (lambda z: z.real, lambda z: z.imag) if data.is_complex() else (lambda z: z,)
                if shared:
                    A = Ks + DiagLinearOperator(n3[0].reshape(-1))   # one operator for all baselines
                    # columns = parts of every baseline, part-major so the reshape recovers them
                    rhs = torch.cat([p(d3) for p in parts], 0).reshape(len(parts) * Nbls, -1).transpose(0, 1)
                    m = Ks.matmul(A.solve(rhs)).transpose(0, 1).reshape(len(parts), Nbls, Ntimes, Nfreqs)
                    fill = torch.complex(m[0], m[1]) if data.is_complex() else m[0]
                    out = torch.where(f3, fill.to(d3.dtype), d3)
                else:
                    out = d3.clone()
                    for bl in range(Nbls):
                        A = Ks + DiagLinearOperator(n3[bl].reshape(-1))  # this baseline's added diagonal
                        filled = [Ks.matmul(A.solve(p(d3[bl]).reshape(-1, 1))).reshape(Ntimes, Nfreqs)
                                  for p in parts]
                        fillbl = torch.complex(filled[0], filled[1]) if data.is_complex() else filled[0]
                        out[bl] = torch.where(f3[bl], fillbl.to(out.dtype), d3[bl])
        out = out[0] if twod else out
        return (out, dict(freq_kernel=freq_kernel, time_kernel=time_kernel,
                          covariance=Ks)) if return_model else out

    raise ValueError(f"mode must be 'freq', 'time' or 'joint', got {mode!r}")

