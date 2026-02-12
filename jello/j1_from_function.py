import numpy as np
import num_dual as nd

from numpy import array, ndarray
from typing import Any, Callable

def J1_from_function(
    domain_corner_v: ndarray,
    mesh_vertex_count_v: ndarray,
    function_shape: ndarray,
    function: Callable[[Any], Any]
) -> ndarray:
    assert domain_corner_v.shape == (2, 1)
    if not isinstance(mesh_vertex_count_v, ndarray):
        mesh_vertex_count_v = array(mesh_vertex_count_v)
    assert mesh_vertex_count_v.shape == (1,)
    assert mesh_vertex_count_v.dtype == int
    if not isinstance(function_shape, ndarray):
        function_shape = array(function_shape)
    assert np.all(mesh_vertex_count_v >= 2)

    mesh_x = np.linspace(domain_corner_v[0,0], domain_corner_v[1,0], mesh_vertex_count_v[0], endpoint=True)

    retval = ndarray(function_shape.tolist() + [mesh_vertex_count_v[0], 2], dtype=np.float64)
    for i, x in enumerate(mesh_x):
        X = array([nd.Dual64(x, 1.0)])
        F_X = function(X)
        assert np.all(np.shape(F_X) == function_shape)
        retval[...,i,0] = np.vectorize(lambda x: x.value)(F_X)
        retval[...,i,1] = np.vectorize(lambda x: x.first_derivative)(F_X)
    return retval

def test():
    from .j1 import J1

    # TODO: Have it test random cubic polynomials.
    domain_corner_v = array([[0.0], [3.0]])
    mesh_vertex_count = 2
    mesh_vertex_count_v = array([2])
    function = lambda X: X[0]**3 + 2.0*X[0]**2 + 3.0*X[0] + 4.0
    j1_data = J1_from_function(domain_corner_v, mesh_vertex_count_v, (), function)
    for x0 in np.linspace(domain_corner_v[0,0], domain_corner_v[1,0], mesh_vertex_count*7, endpoint=True):
        X = array([x0])
        F_X = J1(X[0], domain_corner_v[:,0]).flatten().dot(j1_data.flatten())
        expected_F_X = function(X)
        assert np.allclose(F_X, expected_F_X), f'F_X = {F_X}, expected_F_X = {expected_F_X}'
    print('jello.j1_from_function.test passed')

if __name__ == '__main__':
    test()
