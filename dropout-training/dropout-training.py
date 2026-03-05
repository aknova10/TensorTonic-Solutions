import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if not rng:
        dropout_pattern = np.random.random(size=x.shape) #create a map of random values the same shape as input
    else:
        dropout_pattern = rng.random(size=x.shape)
    dropout_pattern = dropout_pattern < (1-p) # values greater than 1-p are False (i.e. 0) and values less than 1-p are True (i.e. 1)
    dropout_pattern = dropout_pattern * (1/(1-p)) # inverted scaling applied of unmasked values

    output = x * dropout_pattern # scales values based on mask pattern generated
    
    return (output, dropout_pattern)