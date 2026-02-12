import numpy as np
import num_dual as nd

from numpy import array, ndarray
from typing import Any, Callable

def J1_otimes_J1_from_function(
    domain_corner_v: ndarray,
    mesh_vertex_count_v: ndarray,
    function_shape: ndarray,
    function: Callable[[Any], Any]
) -> ndarray:
    assert domain_corner_v.shape == (2, 2)
    if not isinstance(mesh_vertex_count_v, ndarray):
        mesh_vertex_count_v = array(mesh_vertex_count_v)
    assert mesh_vertex_count_v.shape == (2,)
    assert mesh_vertex_count_v.dtype == int
    if not isinstance(function_shape, ndarray):
        function_shape = array(function_shape)
    assert function_shape.dtype == int
    assert np.all(mesh_vertex_count_v >= 2)

    mesh_x = np.linspace(domain_corner_v[0,0], domain_corner_v[1,0], mesh_vertex_count_v[0], endpoint=True)
    mesh_y = np.linspace(domain_corner_v[0,1], domain_corner_v[1,1], mesh_vertex_count_v[1], endpoint=True)

    retval = ndarray(function_shape.tolist() + [mesh_vertex_count_v[0], mesh_vertex_count_v[1], 2, 2], dtype=np.float64)
    for i, x in enumerate(mesh_x):
        for j, y in enumerate(mesh_y):
            X = array([nd.HyperDual64(x, 1.0, 0.0, 0.0), nd.HyperDual64(y, 0.0, 1.0, 0.0)])
            F_X = function(X)
            assert np.all(np.shape(F_X) == function_shape)
            retval[...,i,j,0,0] = np.vectorize(lambda x: x.value)(F_X)
            retval[...,i,j,1,0] = np.vectorize(lambda x: x.first_derivative[0])(F_X)
            retval[...,i,j,0,1] = np.vectorize(lambda x: x.first_derivative[1])(F_X)
            retval[...,i,j,1,1] = np.vectorize(lambda x: x.second_derivative)(F_X)
    return retval

# def test():
#     # TODO: Have it test random bicubic polynomials.
#     domain_corner_v = array([[0.0, 0.0], [2.0, 3.0]])
#     mesh_vertex_count_v = array([4, 5])
#     function = lambda X: np.outer(X, X)*X[0] + np.diag([X[0], X[1]**3])
#     j1_otimes_j1_data = J1_otimes_J1_from_function(domain_corner_v, mesh_vertex_count_v, (2,2), function)
#     print(f'j1_otimes_j1_data = {j1_otimes_j1_data}')
#     # TODO: Implement this test.
#     print('jello.j1_otimes_j1_from_function.test passed')
