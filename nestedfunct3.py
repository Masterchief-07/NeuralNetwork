import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

#set callable type function
Functs = Callable[[ndarray], ndarray]

#set List of callable functions
Chain = List[Functs]

def derive(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_+delta) - func(input_-delta)) / (2*delta)

def chain_3(chain: Chain, x: ndarray) -> ndarray:
    assert len(chain) == 3, "chain must have 3 functions"
    assert x.ndim == 1, "1dim array as input"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))

def chain_derive_3(chain: Chain, x: ndarray) -> ndarray:
    assert len(chain) == 3, "chain must have 3 functions"
    assert x.ndim == 1, "1dim array as input"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    #f1(x)
    f1_x = f1(x)
    #df1/dx
    df1 = derive(f1, x)
    #f2(f1)
    f2_f1 = f2(f1_x)
    #df2/du
    df2 = derive(f2, f1_x)
    #df3/du
    df3 = derive(f3, f2_f1)

    return df3*df2*df1

#define function
def square(input_:ndarray) -> ndarray:
    return np.power(input_, 2)

def sigmoid(input_:ndarray) ->ndarray:
    return 1 / (1 + np.exp(-input_))

def leaky_relu(x:ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)

#plot
PLOT_RANGE = np.arange(-3, 3, 0.01)
#chain_1 = [sigmoid,square,leaky_relu]
chain_1 = [leaky_relu, sigmoid, square]

fchain1 = chain_3(chain_1, PLOT_RANGE)
dchain1 = chain_derive_3(chain_1, PLOT_RANGE)

plt.plot(PLOT_RANGE, fchain1)
plt.plot(PLOT_RANGE, dchain1)
plt.show()



