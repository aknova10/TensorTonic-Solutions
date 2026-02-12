import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    np_x = np.array(x)
    sorted_np_x = np.sort(np_x)

    np_q = np.array(q)

    # np.percentile(sorted_np_x, np_q, interpolation="linear")    
    return np.percentile(sorted_np_x, np_q, method="linear")    