def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    n = len(X)
    in_dim = len(X[0])
    out_dim = len(W[0])
    
    Y = [[0 for _ in range(out_dim)] for _ in range(n)]
    
    for i in range(n):
        for j in range(out_dim):
            Y[i][j] = sum(X[i][k] * W[k][j] for k in range(in_dim)) + b[j]
    
    return Y