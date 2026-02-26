import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.power(base, (2 * (np.arange(d_model) // 2)) / d_model) # (np.arange(d_model) // 2) => [0 0 1 1 2 2 ...] to get same value of i for each sin and cos when skipping to match shapes

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(pos/div[0::2]) #div[0::2] to match the shape of pe[:, 0::2] while broadcasting
    pe[:, 1::2] = np.cos(pos/div[1::2]) #same as above
    
    return pe
    