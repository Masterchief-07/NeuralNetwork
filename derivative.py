from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_+delta) - func(input_ - delta)) / (2*delta)

def square(input_ : ndarray) -> ndarray:
    return np.power(input_, 2)

input_ = np.array([1,2,3,4,5,6,7,8,9,0])

square_result  = square(input_)

deriv_result = deriv(square, input_)

print(f"{input_} \n {square_result} \n {deriv_result}")

plt.plot(input_, square_result, label = 'square')

plt.plot(input_, deriv_result, label = 'derivative')

plt.show()

