import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List



def forward_linear_regression(X_batch: ndarray, y_batch: ndarray, weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:
    assert X_batch.shape[0] == y.shape[0]

    assert X_batch.shape[1] == weights['W'].shape[0]

    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])

    P = N + weights['b']

    loss = np.mean(np.power(y_vatch - P, 2))

    #save information
    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

def loss_gradients(forward_info:Dict[str, ndarray], weights:Dict[str, ndarray]) -> Dict[str, ndarray]:
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    
    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1,0))

    dLdW = np.dot(dNdW, dLdN)

    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradient:Dict[str, ndarray] = {}
    loss_gradient['W'] = dLdW
    loss_gradient['B'] = dLdB

    return loss_gradient


