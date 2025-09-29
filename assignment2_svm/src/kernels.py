import numpy as np

def linear_kernel(X, Y):
    #K(x,y) = x . y
    return X @ Y.T #matrix multiply X with Y transpose to compute all the dot products at once

#radial basis function kernal
def rbf_kernel(X, Y, gamma):
    #K(x,y) = exp(-gamma * ||x-y||^2)

    #compute squared norms of each row in X and Y
    X_norm = np.sum(X**2, axis = 1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]

    #use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
    #and then apply the exponential function to get the kernel matrix
    dists = X_norm + Y_norm - 2 * (X @ Y.T)
    return np.exp(-gamma * dists)


#polynomial kernel function
def poly_kernel(X, Y, degree=3, coef0=1.0):
    #K(x,y) = (x · y + coef0)^degree
    return (X @ Y.T + coef0) ** degree