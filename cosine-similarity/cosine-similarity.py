import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    np_a = np.array(a)
    np_b = np.array(b)

    euclidean_norm_a = np.linalg.norm(np_a)
    euclidean_norm_b = np.linalg.norm(np_b)

    if euclidean_norm_a == 0 or euclidean_norm_b == 0:
        return 0 

    similarity = (np_a @ np_b)/(euclidean_norm_a*euclidean_norm_b)
    
    return similarity