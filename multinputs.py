import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def derive(func:Array_Function, x:ndarray, delta:float=0.001) -> ndarray:
    return (func(x+delta) - func(x-delta)) / (2*delta)

def sigma(x:ndarray)->ndarray:
    return 1 / (1 + np.exp(-x))

def multiple_inputs_add(x:ndarray, y:ndarray, sigma: Array_Function) -> float:
    #functions with multiples inputs and addition, forward pass
    assert x.shape == y.shape
    
    a = x + y

    return sigma(a)

def multiple_inputs_add_backward(x:ndarray, y:ndarray, sigma:Array_Function) -> float:
    assert x.shape == y.shape
    a = x + y
    dsda = deriv(sigma, a)
    dadx, dady = 1, 1
    return dsda * dadx, dsda * dady




