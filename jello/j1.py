import numpy as np

from numpy import array, ndarray
from typing import Any

#
# Finite-element-specific code -- J1 control functions and their derivatives.
#

def J1(x: Any, endpoints: ndarray) -> ndarray:
    """
    1-jet control functions on the given interval, each evaluated at x.
    Indexed as [i,j], where i is the endpoint index (0 or 1) and j is the jet index (0 or 1).
    """
    assert endpoints.shape == (2,), f'endpoints.shape = {endpoints.shape}'
    length = endpoints[1] - endpoints[0]
    t = (x - endpoints[0]) / length
    s = 1 - t
    # print(f'J1; x = {x}, endpoints = {endpoints}, length = {length}, s = {s}, t = {t}')
    retval = array([[s**3 + 3*s**2*t, s**2*t*length], [3*s*t**2 + t**3, -s*t**2*length]])
    return retval

def DJ1(x: Any, endpoints: ndarray) -> ndarray:
    """
    Derivative of the 1-jet control functions on the given interval, each evaluated at x.
    Indexed as [i,j], where i is the endpoint index (0 or 1) and j is the jet index (0 or 1).
    """
    assert endpoints.shape == (2,), f'endpoints.shape = {endpoints.shape}'
    length = endpoints[1] - endpoints[0]
    t = (x - endpoints[0]) / length
    s = 1 - t
    # print(f'DJ1; x = {x}, endpoints = {endpoints}, length = {length}, s = {s}, t = {t}')
    retval = array([[-6*s*t/length, s**2 - 2*s*t], [6*s*t/length, -2*s*t + t**2]])
    # print(f'DJ1 = {retval}')
    return retval

def test():
    import num_dual as nd
    import sympy as sp

    from .sym import sym_eval, sym_D, sym_simplify

    x = sp.var('x')
    for endpoints in [array([0.0, 1.0]), array([1.0, 2.0]), array([0.0, 0.5]), array([0.5, 1.0]), array([0.32532, 1.253216542])]:
        # Verify J1 symbolically.
        sym_endpoints = np.vectorize(sp.Rational)(endpoints)
        j1 = J1(x, sym_endpoints)
        # print(f'j1 = {j1}')
        dj1 = sym_D(j1, x)
        # print(f'dj1 = {dj1}')
        assert np.all(sym_eval(j1, x, sym_endpoints[0]) == array([[1, 0], [0, 0]]))
        assert np.all(sym_eval(dj1, x, sym_endpoints[0]) == array([[0, 1], [0, 0]]))
        assert np.all(sym_eval(j1, x, sym_endpoints[1]) == array([[0, 0], [1, 0]]))
        assert np.all(sym_eval(dj1, x, sym_endpoints[1]) == array([[0, 0], [0, 1]]))

        # Also verify DJ1 symbolically.
        sym_dj1 = DJ1(x, sym_endpoints)
        assert np.all(sym_simplify(sym_dj1 - dj1) == 0)
        # print(f'sym_dj1 = {sym_dj1}')
        assert np.all(sym_eval(sym_dj1, x, sym_endpoints[0]) == array([[0, 1], [0, 0]]))
        assert np.all(sym_eval(sym_dj1, x, sym_endpoints[1]) == array([[0, 0], [0, 1]]))

        # Ensure that the 1-jet control criteria are satisfied at each endpoint via numerical differentiation.
        for i in range(2):
            for j in range(2):
                # print('------------------------------------')
                # print(f'i = {i}, j = {j}')

                for k, endpoint_k in enumerate(endpoints):
                    # print('------------------------------------')
                    # print(f'i = {i}, j = {j}, k = {k}, endpoint_k = {endpoint_k}')
                    j1_i_j = array(nd.first_derivative(lambda t: J1(t, endpoints)[i,j], endpoint_k))
                    dj1_i_j = DJ1(endpoint_k, endpoints)[i,j]
                    expected_j1_i_j = array([1.0 if (i == k) and (j == l) else 0.0 for l in range(2)])
                    # print(f'j1_i_j = {j1_i_j}')
                    # print(f'dj1_i_j = {dj1_i_j}')
                    # print(f'expected_j1_i_j = {expected_j1_i_j}')
                    assert np.allclose(j1_i_j, expected_j1_i_j)

        # print(f'endpoints = {endpoints} ------------------------------')
        for t_real in np.linspace(endpoints[0], endpoints[1], 234, endpoint=True):
            # print(f't_real = {t_real}')
            t = nd.Dual64(t_real, 1.0)
            dj1 = DJ1(t_real, endpoints)
            expected_dj1 = array([[nd.first_derivative(lambda t: J1(t, endpoints)[i,j], t_real)[1] for j in range(2)] for i in range(2)])
            # print(f'dj1 = {dj1}')
            # print(f'expected_dj1 = {expected_dj1}')
            assert np.allclose(dj1, expected_dj1)

    print('jello.j1.test passed')
