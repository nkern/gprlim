import numpy as np


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