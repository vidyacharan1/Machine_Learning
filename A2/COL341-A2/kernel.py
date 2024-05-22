import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    kernel_matrix = X @ X.T
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    kernel_matrix = (gamma * (X @ X.T) + coeff) ** degree
    return kernel_matrix
    pass

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = np.exp(-gamma * (np.linalg.norm(X[i] - X[j]) ** 2))
    return kernel_matrix
    pass

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    kernel_matrix = np.tanh(gamma*(X @ X.T) + coeff)
    return kernel_matrix
    pass

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = np.exp(-gamma* np.linalg.norm(X[i] - X[j]))
    return kernel_matrix
    pass



