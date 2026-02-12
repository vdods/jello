# This file contains the code for numerically solving for the embedding of a flat disk into a hyperbolic 2-space (Poincare disk model).

import numpy as np
import sympy as sp

from .fem1d import FEFunctionSpace1D
from .j1_from_function import J1_from_function
from .measure import DiscreteMeasure
from .optimize import newtons_method_minimize
from .phi1d import phi, Dphi
from .spatial_manifold import SpatialManifold
from numpy import array, ndarray
from typing import Any, Callable, Optional

def log(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.log(x)
    else:
        return np.log(x)

def det(M: ndarray) -> Any:
    return M[0,0] * M[1,1] - M[0,1] * M[1,0]

def g(x: ndarray) -> ndarray:
    """
    Euclidean metric in polar coordinates, where x = (r, theta).
    """
    r = x[0]
    return array([[1, 0], [0, r**2]])

def g_inv(x: ndarray) -> ndarray:
    """Inverse Euclidean metric in polar coordinates, where x = (r, theta)."""
    r = x[0]
    return array([[1, 0], [0, 1/r**2]])

def h(k: Any, y: ndarray) -> ndarray:
    """
    Metric on the Poincare disk having constant Gaussian curvature k < 0,
    where y = (rho, phi) are polar coordinates, noting that rho in [0, 1).
    """
    rho = y[0]
    return -4 / (k * (1 - rho**2)**2) * array([[1, 0], [0, rho**2]])

def C(k: Any, x: ndarray, y: ndarray, F: ndarray, g_inv: Callable[[ndarray], ndarray], h: Callable[[Any, ndarray], ndarray]) -> ndarray:
    """
    Cauchy strain tensor for a body embedded in the Poincare disk having constant Gaussian curvature k < 0.
    """
    assert x.shape == (2,)
    assert y.shape == (2,)
    assert F.shape == (2, 2)
    retval = np.ndarray((2, 2), dtype=F.dtype)
    retval[...] = g_inv(x)
    retval @= F.T
    retval @= h(k, y)
    retval @= F
    return retval

def W(k: Any, alpha: Any, x: ndarray, y: ndarray, F: ndarray) -> Any:
    """
    Stored energy density for a body embedded in the Poincare disk having constant Gaussian curvature k < 0.
    """
    c = C(k, x, y, F, g_inv, h)
    return alpha * (c.trace() - 2 - log(det(c)))

def stored_energy_integrand_fn() -> Callable[[Any, Any, Any, Any, Any], Any]:
    k = sp.var('k', real=True, negative=True)
    # Can use alpha = 1 here because the stored energy functional is proportional to alpha.
    alpha = 1
    r = sp.var('r', real=True, positive=True)
    theta = sp.var('theta', real=True)
    x = array([r, theta])
    f = sp.var('f', real=True, positive=True)
    f_prime = sp.var('f_prime', real=True)
    phi = array([f, theta])
    Dphi = array([[f_prime, 0], [0, 1]])
    # W is the stored energy density function for the body embedded in the Poincare disk.
    integrand = 2*sp.pi*W(k, alpha, x, phi, Dphi)*r
    print(f'W(k, {alpha}, r, {f}, {f_prime}) = {sp.latex(integrand)}')
    print(f'W(k, {alpha}, r, {f}, {f_prime}) = {W(k, alpha, x, phi, Dphi)}')

    # NOTE the 2*pi*r factor is included here because the integrand comes from an integral in polar coordinates over the body.
    return sp.lambdify([k, r, f, f_prime], integrand)

def solve_disk_body_embedding():
    """
    S is the Poincare disk having constant Gaussian curvature k < 0.
    The body is a flat disk that is embedded in S.
    This will compute the minimal embedding of the body into S.
    """

    # alpha = sp.var('alpha', real=True, positive=True)
    # k = sp.var('k', real=True, negative=True)
    # # R = sp.var('R', real=True, positive=True)

    # r = sp.var('r', real=True, positive=True)
    # theta = sp.var('theta', real=True)
    # x = array([r, theta])
    # f = sp.var('f', real=True, positive=True)
    # f_prime = sp.var('f_prime', real=True)
    # phi = array([f, theta])
    # Dphi = array([[f_prime, 0], [0, 1]])
    # # print(f'W(r, {f}, {f_prime}) = {sp.latex(W(k, alpha, x, phi, Dphi))}')
    # # W is the stored energy density function for the body embedded in the Poincare disk.
    # print(f'W(r, {f}, {f_prime}) = {W(k, alpha, x, phi, Dphi)}')

    # Curvature of the hyperbolic 2-space being embedded into.  Must be negative.
    # k_v = array([-1.0])

    # These are on the order of magnitude as the real values for the problem, but they produce numerically ill-conditioned calculations.
    # k_0 = -1.0e-23
    # k_1 = 1.0e-30
    k_0 = -1.0e-3
    k_1 = 1.0e-6
    k_v = np.linspace(k_0 - k_1, k_0 + k_1, 21, endpoint=True)

    stored_energy_v = np.ndarray((len(k_v),), dtype=np.float64)
    for i, k in enumerate(k_v):
        # Body disk radius.  Different radii should expect to get different embeddings.
        R = 1.0
        # Function space for the embedding.
        F = FEFunctionSpace1D(
            domain_corner_v=array([[0.0], [R]]),
            vertex_count_v=array([6]),
        )
        measure = DiscreteMeasure.gauss_legendre_5_on_interval(F.domain_corner_v, F.vertex_count_v)
        stored_energy_integrand = stored_energy_integrand_fn()

        # Create the initial guess for the embedding.
        initial_guess_fn = lambda X: np.tanh(0.5*(-k)**0.5*R)/R * X[0]
        f_initial = J1_from_function(F.domain_corner_v, F.vertex_count_v, (), initial_guess_fn)
        # assert np.allclose(phi(f_initial, F.domain_corner_v, F.domain_corner_v[0,:]), initial_guess_fn(F.domain_corner_v[0,:]))
        # assert np.allclose(phi(f_initial, F.domain_corner_v, F.domain_corner_v[1,:]), initial_guess_fn(F.domain_corner_v[1,:]))
        print(f'f_initial = {f_initial}')

        # TODO: DOFs with initial point fixed using Dirichlet boundary conditions.
        assert f_initial[0,0] == 0.0, f'f_initial[0,0] = {f_initial[0,0]}'

        def stored_energy(f: ndarray) -> Any:
            if not isinstance(f, ndarray):
                f = array(f)
            f_reshaped = f.reshape(F.scalar_function_shape)
            return measure.integrate(lambda X: stored_energy_integrand(k, X[0], phi(f_reshaped, F.domain_corner_v, X), Dphi(f_reshaped, F.domain_corner_v, X)))

        print(f'stored_energy(f_initial) = {stored_energy(f_initial)}')

        # "metric" is actually an inner product and doesn't depend on the basepoint.
        metric = F.L2_inner_product(measure, dtype=np.float64)
        print(f'metric = {metric}')
        metric_fn = lambda _: metric
        f_optimized = newtons_method_minimize(stored_energy, f_initial.reshape(F.scalar_function_dim), metric_fn=metric_fn, callback=None).reshape(F.scalar_function_shape)
        print(f'f_optimized = {f_optimized}')
        print(f'stored_energy(f_optimized) = {stored_energy(f_optimized)}')
        stored_energy_v[i] = stored_energy(f_optimized)

    print('stored energies:')
    for k, stored_energy in zip(k_v, stored_energy_v):
        print(f'    k = {k}, stored_energy = {stored_energy}')

    # # This is the integrand of the stored energy functional that has been re-expressed in terms of f(r),
    # # where phi(r, theta) = (f(r), theta).
    # stored_energy_integrand_fn = sp.lambdify([k, alpha, r, f, f_prime], 2*sp.pi*W(k, alpha, x, phi, Dphi)*r)

    # def stored_energy_integrand(k: Any, alpha: Any, r: Any, f: Any, f_prime: Any) -> Any:


    # # Start with the simplest mesh, which is a single interval.
    # measure = DiscreteMeasure.gauss_legendre_5_on_interval(array([0.0, R]), mesh_vertex_count_v=array([2]))
    # stored_energy = lambda func: measure.integrate(lambda X: stored_energy_integrand(k, alpha, X[0], phi(func.reshape(F.scalar_function_shape, D), F.domain_corner_v, X)))

    # # print(f'w(k, alpha, r, f, f_prime) = {w(k=-1.0, alpha=1.0, r=0.5, f=0.5, f_prime=1.0)}')

if __name__ == '__main__':
    solve_disk_body_embedding()
