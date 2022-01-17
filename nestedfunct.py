import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List
from typing import Callable

#derive function
def derive(func:Callable[[ndarray], ndarray], input_:ndarray, delta:float = 0.001) -> ndarray:
    return (func(input_+delta) - func(input_ - delta)) / (2*delta)

#define the type of function
Array_function = Callable[[ndarray], ndarray]

#creation of the chain of functions
Chain = List[Array_function]

#define chain of lenght 2
def chain_length_2(chain: Chain, a:ndarray) -> ndarray:
    assert len(chain) == 2, "length of imput 'chain' should be 2"

    assert a.ndim == 1, "require 1dim array as input"

    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))

#define the derive of chain length 2
def chain_derive_2(chain: Chain, x : ndarray) -> ndarray:
    assert len(chain) == 2, "length of imput 'chain' should be 2"

    assert x.ndim == 1, "require 1dim array as input"

    f1 = chain[0]
    f2 = chain[1]

    #f1(x)
    f1x = f1(x)
    #df1/du
    df1dx = derive(f1, x)

    #df2/du
    df2du = derive(f2, f1x)

    return df1dx * df2du


#creation of the square function
def square(x: ndarray) -> ndarray:
    return np.power(x, 2);

#creation of the sigmoid function
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

#plot
PLOT_RANGE = np.arange(-3, 3, 0.01)
chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

fchain1 = chain_length_2(chain_1, PLOT_RANGE)
dchain1 = chain_derive_2(chain_1, PLOT_RANGE)
fchain2 = chain_length_2(chain_2, PLOT_RANGE)
dchain2 = chain_derive_2(chain_2, PLOT_RANGE)

plt.plot(PLOT_RANGE, fchain1)
plt.plot(PLOT_RANGE, dchain1)
plt.show()

plt.plot(PLOT_RANGE, fchain2)
plt.plot(PLOT_RANGE, dchain2)

plt.show()


