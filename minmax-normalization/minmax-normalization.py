import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    x_min = np.min(X, axis=axis)
    x_max = np.max(X, axis=axis)

    if axis:
        x_min_max = (np.transpose(X) - x_min)/(x_max - x_min + eps)
        return x_min_max.T
    else:
        x_min_max = (X - x_min)/(x_max - x_min + eps)
        return x_min_max 