import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    product = np.array(x) * np.array(y) # convert to numpy array to perform element-wise multiplication

    dot_prd = sum(product) # sum of all element-wise products

    return dot_prd