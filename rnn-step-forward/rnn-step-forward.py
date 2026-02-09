import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x = (x_t @ Wx) + (h_prev @ Wh) + b
    # e = 2.718

    # h_t = (e**x-e**-x)/(e**x+e**-x)
    
    h_t = np.tanh(x)
    return h_t
