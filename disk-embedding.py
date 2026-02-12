import numpy as np
import sympy as sp

from numpy import array, ndarray
from sympy import Integer
from typing import Any, Callable

def g(x: ndarray) -> ndarray:
    """Euclidean metric in polar coordinates."""
    r = x[0]
    return array([[1, 0], [0, r**2]])

def g_inv(x: ndarray) -> ndarray:
    """Inverse of g(x)."""
    r = x[0]
    return array([[1, 0], [0, 1 / r**2]])

def h(k: Any, y: ndarray) -> ndarray:
    """Poincare disk metric in polar coordinates, where the radial coordinate is in [0, 1)."""
    rho = y[0]
    return -4 / (k * (1 - rho**2)**2) * array([[1, 0], [0, rho**2]])

def C(k: Any, x: ndarray, y: ndarray, F: ndarray) -> ndarray:
    """Cauchy strain tensor."""
    assert x.shape == (2,)
    assert y.shape == (2,)
    assert F.shape == (2, 2)
    retval = np.ndarray((2, 2), dtype=F.dtype)
    retval[...] = g_inv(x)
    retval @= F.T
    retval @= h(k, y)
    retval @= F
    return retval

def det(M: ndarray) -> Any:
    """Determinant of a 2x2 matrix."""
    assert M.shape == (2, 2)
    return M[0,0] * M[1,1] - M[0,1] * M[1,0]

def W(k: Any, alpha: Any, x: ndarray, y: ndarray, F: ndarray) -> Any:
    """Stored energy density."""
    assert x.shape == (2,)
    assert y.shape == (2,)
    assert F.shape == (2, 2)
    c = C(k, x, y, F)
    return alpha * (c.trace() - 2 - sp.log(det(c)))

def gauss_legendre_quadrature(*, n: int, interval: ndarray, dtype: Any) -> Any:
    """Gauss-Legendre quadrature."""
    assert n >= 1
    assert interval.shape == (2,)
    interval_length = interval[1] - interval[0]
    print(f'interval.dtype = {interval.dtype}')
    print(f'interval_length = {interval_length}')

    # Quadrature points and weights for interval [-1, 1].
    point_v = np.ndarray((n,), dtype=dtype)
    weight_v = np.ndarray((n,), dtype=dtype)
    print(f'dtype = {dtype}')
    print(f'dtype(3)/dtype(5) = {dtype(3)/dtype(5)}')
    if n == 1:
        point_v[0] = dtype(0)
        weight_v[0] = dtype(2)
    elif n == 2:
        sqrt_1_over_3 = dtype(3)**(-dtype(1)/dtype(2))
        print(f'sqrt_1_over_3 = {sqrt_1_over_3}')
        point_v[0] = -sqrt_1_over_3
        point_v[1] = sqrt_1_over_3
        weight_v[0] = dtype(1)
        weight_v[1] = dtype(1)
    elif n == 3:
        sqrt_3_over_5 = (dtype(3)/dtype(5))**(dtype(1)/dtype(2))
        print(f'sqrt_3_over_5 = {sqrt_3_over_5}')
        point_v[0] = -sqrt_3_over_5
        point_v[1] = dtype(0)
        point_v[2] = sqrt_3_over_5
        weight_v[0] = dtype(5)/dtype(9)
        weight_v[1] = dtype(8)/dtype(9)
        weight_v[2] = dtype(5)/dtype(9)
    else:
        raise ValueError(f'n = {n} is not supported (yet)')

    # Shift interval to [interval[0], interval[1]].
    point_v -= 1
    point_v *= interval_length / dtype(2)
    point_v += interval[0]
    weight_v *= interval_length / dtype(2)

    return point_v, weight_v

def compute_stored_energy_by_quadrature():
    k = sp.var('k', real=True, negative=True)
    alpha = sp.var('alpha', real=True, positive=True)
    r = sp.var('r', real=True, positive=True)
    theta = sp.var('theta', real=True)
    x = array([r, theta])
    a = sp.var('a', real=True, positive=True)
    b = sp.var('b', real=True)
    f = a*r + b*r**2 / 2
    phi = array([f, theta])
    f_prime = sp.diff(f, r)
    Dphi = array([[f_prime, 0], [0, 1]])
    point_v, weight_v = gauss_legendre_quadrature(n=3, interval=array([0, 1], dtype=sp.Integer), dtype=sp.Integer)
    print(f'point_v = {point_v}')
    print(f'weight_v = {weight_v}')
    w = np.sum(weight_v * array([W(k, alpha, x, phi, Dphi).subs(r, point).simplify() for point in point_v]))
    print(f'stored_energy = {sp.latex(w)}')
    print()

    dw_da = sp.diff(w, a).simplify()
    print(f'dw_da = {sp.latex(dw_da)}')
    print()
    dw_db = sp.diff(w, b).simplify()
    print(f'dw_db = {sp.latex(dw_db)}')
    print()
    return 0

def calculate_stuff():
    k = sp.var('k', real=True, negative=True)
    alpha = sp.var('alpha', real=True, positive=True)
    r = sp.var('r', real=True, positive=True)
    theta = sp.var('theta', real=True)
    x = array([r, theta])
    f = sp.var('f', real=True, positive=True)
    f_prime = sp.var('f^{\\prime}', real=True, positive=True)
    f_prime_prime = sp.var('f^{\\prime\\prime}', real=True)

    phi = array([f, theta])
    Dphi = array([[f_prime, 0], [0, 1]])

    w = W(k, alpha, x, phi, Dphi)
    # This is the integrand of the stored energy functional for the hyperelatic body embedded in the Poincare disk.
    w_bar = 2*sp.pi*w*r
    print(f'w_bar = {sp.latex(w_bar)}')
    print()

    # Derive the Euler-Lagrange equations for this functional.
    F = sp.Function('F')
    F_r = F(r)
    DF_r = sp.diff(F_r, r)
    D2F_r = sp.diff(DF_r, r)
    subs_d = {f: F_r, f_prime: DF_r}
    euler_lagrange_form = sp.diff(w_bar, f).subs(subs_d) - sp.diff(sp.diff(w_bar, f_prime).subs(subs_d), r)
    print(f'euler_lagrange_form = {sp.latex(euler_lagrange_form.simplify())}')
    print()
    print(f'euler_lagrange_form (collected) = {sp.latex(euler_lagrange_form.collect(D2F_r).simplify())}')
    print()

    el2 = euler_lagrange_form.subs(D2F_r, f_prime_prime).subs(DF_r, f_prime).subs(F_r, f) * (k*r*(f**2 - 1)**3 * f * (f_prime**2)) / (2*sp.pi*alpha)
    el2 = el2.expand().simplify().collect(f_prime_prime)
    el2.simplify()
    # el2 *= -k * (1 - f**2)**3 * f_prime**2
    # el2 = el2.subs(DF_r, f_prime)
    # el2 = el2.subs(D2F_r, f_prime_prime)
    # el2 = el2.simplify()
    print(f'el2 = {sp.latex(el2)}')
    print()
    # print(f'el2 (collected) = {sp.latex(el2.collect(f_prime_prime).simplify())}')
    # print()

    # euler_lagrange_form_2 = euler_lagrange_form * (-k * (1 - F_r**2)**3 * DF_r**2) / (4*sp.pi*alpha)
    # euler_lagrange_form_2 = euler_lagrange_form_2.simplify()
    # euler_lagrange_form_2 
    # print(f'euler_lagrange_form_2 = {sp.latex(euler_lagrange_form_2)}')
    # print()
    # print(f'euler_lagrange_form_2 (collected) = {sp.latex(euler_lagrange_form_2.collect(D2F_r).simplify())}')
    # print()


    # h = sp.var('h')
    # h_prime = sp.var('h^{\'}')
    # w_bar_h = w_bar.subs({f: r*h, f_prime: h + r*h_prime})
    # print(f'w_bar_h = {sp.latex(w_bar_h)}')

    # # Derive the Euler-Lagrange equations for this functional.
    # H = sp.Function('H')
    # H_r = H(r)
    # DH_r = sp.diff(H_r, r)
    # D2H_r = sp.diff(DH_r, r)
    # subs_d = {h: H_r, h_prime: DH_r}
    # euler_lagrange_form = sp.diff(w_bar_h, h).subs(subs_d) - sp.diff(sp.diff(w_bar_h, h_prime).subs(subs_d), r)
    # print(f'euler_lagrange_form = {sp.latex(euler_lagrange_form.simplify())}')
    # print(f'euler_lagrange_form (collected) = {sp.latex(euler_lagrange_form.collect(D2H_r).simplify())}')

    # p = sp.var('p')
    # p_prime = sp.var('p^{\'}')
    # w_bar_p = w_bar.subs({f: p/r, f_prime: (r*p_prime - p)/r**2})
    # print(f'w_bar_p = {sp.latex(w_bar_p)}')

    # # Derive the Euler-Lagrange equations for this functional.
    # P = sp.Function('P')
    # P_r = P(r)
    # DP_r = sp.diff(P_r, r)
    # D2P_r = sp.diff(DP_r, r)
    # subs_d = {p: P_r, p_prime: DP_r}
    # euler_lagrange_form = sp.diff(w_bar_p, p).subs(subs_d) - sp.diff(sp.diff(w_bar_p, p_prime).subs(subs_d), r)
    # print(f'euler_lagrange_form = {sp.latex(euler_lagrange_form.simplify())}')
    # print(f'euler_lagrange_form (collected) = {sp.latex(euler_lagrange_form.collect(D2P_r).simplify())}')


if __name__ == '__main__':
    # calculate_stuff()
    compute_stored_energy_by_quadrature()
