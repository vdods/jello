import numpy as np
import num_dual as nd

from .j1 import J1, DJ1
from numpy import array, ndarray
from typing import Any

def J1_otimes_J1(corner_v: ndarray, T: Any) -> ndarray:
    """Indexed as [i,j,k,l], where i and j are the endpoint indices and k and l are the jet indices."""
    assert corner_v.shape == (2, 2)
    assert T.shape == (2,)
    size = corner_v[1,:] - corner_v[0,:]
    assert np.all(size > 0)
    x0_j1 = J1(T[0], corner_v[:,0])
    x1_j1 = J1(T[1], corner_v[:,1])
    return np.outer(x0_j1, x1_j1).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)

def D_J1_otimes_J1(corner_v: ndarray, T: Any) -> ndarray:
    """Indexed as [h,i,j,k,l], where h is the D derivative index, i and j are the endpoint indices, and k and l are the jet indices"""
    assert corner_v.shape == (2, 2)
    assert T.shape == (2,)
    size = corner_v[1,:] - corner_v[0,:]
    assert np.all(size > 0)
    x0_j1 = J1(T[0], corner_v[:,0])
    x1_j1 = J1(T[1], corner_v[:,1])
    dx0_j1 = DJ1(T[0], corner_v[:,0])
    dx1_j1 = DJ1(T[1], corner_v[:,1])
    return array([np.outer(dx0_j1, x1_j1).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3), np.outer(x0_j1, dx1_j1).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)])

def test():
    for corner_0 in [array([0.0, 0.0]), array([1.0, 0.5]), array([0.45325, 0.5])]:
        for size in [array([0.75, 1.0]), array([1.0, 1.0]), array([0.55432, 1.1320])]:
            corner_1 = corner_0 + size
            # print(f'corner_0 = {corner_0}, corner_1 = {corner_1} -------------------------')
            corner_v = array([corner_0, corner_1])
            for t0_real in np.linspace(corner_0[0], corner_1[0], 23, endpoint=True):
                for t1_real in np.linspace(corner_0[1], corner_1[1], 23, endpoint=True):
                    T = array([t0_real, t1_real])
                    t_dx0 = array([nd.Dual64(t0_real, 1.0), nd.Dual64(t1_real, 0.0)])
                    d_j1_otimes_j1 = D_J1_otimes_J1(corner_v, T)
                    expected_d_j1_otimes_j1_dx0 = array([[[[nd.first_derivative(lambda t: J1_otimes_J1(corner_v, array([t, t1_real]))[i,j,k,l], t0_real)[1] for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)])
                    expected_d_j1_otimes_j1_dx1 = array([[[[nd.first_derivative(lambda t: J1_otimes_J1(corner_v, array([t0_real, t]))[i,j,k,l], t1_real)[1] for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)])
                    # print(f'd_j1_otimes_j1 = {d_j1_otimes_j1}')
                    # print(f'expected_d_j1_otimes_j1_dx0 = {expected_d_j1_otimes_j1_dx0}')
                    # print(f'expected_d_j1_otimes_j1_dx1 = {expected_d_j1_otimes_j1_dx1}')
                    assert np.allclose(d_j1_otimes_j1[0], expected_d_j1_otimes_j1_dx0)
                    assert np.allclose(d_j1_otimes_j1[1], expected_d_j1_otimes_j1_dx1)
    
    print('jello.j1_otimes_j1.test passed')
