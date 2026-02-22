def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    output_len_i = ((len(X)-pool_size)//stride)+1
    output_len_j = ((len(X[0])-pool_size)//stride)+1 #account for NxM sized matrix
    output = [[] for i in range(output_len_i)]
    # print(output)
    for i in range(output_len_i):
        for j in range(output_len_j):
            row = []
            for a in range(pool_size):
                for b in range(pool_size):
                    # output[i][j] = X[(i*stride)+a][(j*stride)+b]
                        row.append(X[(i*stride)+a][(j*stride)+b])
            output[i].append(max(row))
    
    # print(output)
    
    return output