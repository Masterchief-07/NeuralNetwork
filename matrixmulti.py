import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

def derive(func:Callable[[ndarray], ndarray], X:ndarray, delta:float=0.001) -> ndarray:
    return (func(X+delta) - func(X-delta)) / (2*delta)

def sigma(X:ndarray) ->ndarray:
    return 1 / (1 + np.exp(-X))

def matmul_forward(x:ndarray, w:ndarray) -> ndarray:
    assert x.shape[1] == y.shape[0] , "matrix multiplaction error"

    N  = np.dot(x, w)

    return N

def matmul_backward_first(x:ndarray, w:ndarray) -> ndarray:
    dNdX = np.transpose(w, (1,0))

    return dNdX

def matrix_function_forward(X:ndarray, W:ndarray, func:Callable[[ndarray], ndarray]) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    
    S = sigma(N)

    return S

def matrix_function_forward_sum(X:ndarray, W:ndarray, func:Callable[[ndarray], ndarray]) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    
    S = sigma(N)

    L = np.sum(S)
    return L

def matrix_function_backward(X:ndarray, W:ndarray, func:Callable[[ndarray], ndarray]) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    
    S = func(N)

    dSdN = derive(func, N)

    dNdX = np.transpose(W, (1,0))

    return np.dot(dSdN, dNdX)

def matrix_function_backward_sum(X:ndarray, W:ndarray, func:Callable[[ndarray], ndarray]) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    
    S = func(N)

    L = np.sum(S)

    dLdS = np.ones_like(S)

    dSdN = derive(func, N)

    dLdN = dLdS * dSdN

    dNdX = np.transpose(W, (1,0))
    
    dLdX = np.dot(dSdN, dNdX)

    return dLdX

np.random.seed(190204)
X = np.random.randn(3,3)
W = np.random.randn(3,2)
print(X.shape, "\n", W.shape, "\n")
print("X: ")
print(X,"\n")
print("W: ")
print(W,"\n")
print("sigma forward: \n")
print(matrix_function_forward(X,W,sigma), "\n")

print(matrix_function_backward(X,W,sigma), "\n")

print("sum forward: \n")
print(matrix_function_forward_sum(X, W, sigma), "\n")
print(matrix_function_backward_sum(X, W, sigma), "\n")



