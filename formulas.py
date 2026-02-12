import numpy as np
import sympy as sp
import vorpy as vp
import vorpy.riemannian
import vorpy.symbolic
import vorpy.tensor

from numpy import array, ndarray
from sympy import Integer
from typing import Any, Callable
from vorpy.riemannian import covariant_derivative_of as nabla

# # Calculate the Gaussian curvature of the surface z = -1/r (cylindrical coordinates).
# def compute_gaussian_curvature_in_xy_coordinates():
#     x = sp.var('x')
#     y = sp.var('y')
#     X = array([x, y])

#     r = sp.sqrt(X.dot(X))
#     z = -1/r

#     phi = array([x, y, z])
#     Dphi = vp.symbolic.D(phi, X)
#     g = Dphi.T.dot(Dphi)
#     print(f'g = {g}')

#     g_inv = array(sp.Matrix(g.tolist()).inv().tolist())
#     print(f'g_inv = {g_inv}')

#     Gamma = vp.riemannian.christoffel_symbol(g, g_inv, X)
#     print(f'Gamma = {Gamma}')

#     partial_x = array([Integer(1), Integer(0)])
#     partial_y = array([Integer(0), Integer(1)])

#     nabla_y_partial_y = nabla(partial_y, Gamma, X).dot(partial_y)
#     nabla_x_partial_y = nabla(partial_y, Gamma, X).dot(partial_x)
#     nabla_x_nabla_y_partial_y = nabla(nabla_y_partial_y, Gamma, X).dot(partial_x)
#     nabla_y_nabla_x_partial_y = nabla(nabla_x_partial_y, Gamma, X).dot(partial_y)

#     K_computed = partial_x.dot(g).dot(nabla_x_nabla_y_partial_y - nabla_y_nabla_x_partial_y) / (g[0,0] * g[1,1] - g[0,1] * g[1,0])
#     K_computed = K_computed.simplify()
#     # print(f'K: {sp.latex(K)}')

#     def K(r):
#         return -Integer(2)*r**2 / (Integer(1) + r**4)**2

#     # Formula was derived from K_computed and some hand-computations.
#     K_r = K(r)

#     assert (K_computed - K_r).simplify() == 0

#     R = sp.var('r', real=True, positive=True)
#     K_R = K(R)
#     print(f'Gaussian curvature: K(r) = {K_R} = {sp.latex(K_R)}')

#     DK = K_R.diff(R).simplify()
#     print(f'dK/dr = {DK} = {sp.latex(DK)}')

#     print(f'Critical point(s) of K(r): {sp.solve(DK, R)}')

#     assert DK.subs(R, 3**sp.Rational(-1,4)).simplify() == 0
#     print('Critical point at r = 3^(-1/4) verified')

#     # Compute the above in (r,theta) coordinates also.
#     print('Computing in (r,theta) coordinates...')

# def compute_gaussian_curvature_in_rtheta_coordinates():
#     r = sp.var('r')
#     theta = sp.var('theta')
#     X = array([r, theta])

#     z = -1/r
#     phi = array([r, theta, z])
#     Dphi = vp.symbolic.D(phi, X)
#     g = Dphi.T.dot(Dphi)
#     print(f'g = {g}')

#     g_inv = array(sp.Matrix(g.tolist()).inv().tolist())
#     print(f'g_inv = {g_inv}')

#     Gamma = vp.riemannian.christoffel_symbol(g, g_inv, X)
#     print(f'Gamma = {Gamma}')
#     print(f'Gamma: {sp.latex(Gamma)}')

def compute_formulas_for_surface_of_revolution(z_fn: Callable[[Any], Any], *, compute_in_graph_coordinates: bool = True):
    """
    z_fn should be a function of r.
    """

    r = sp.var('r')
    theta = sp.var('theta')
    polar = array([r, theta])

    z = z_fn(r)

    print('--------------------------------------------------------')
    print(f'Surface of revolution z = {z_fn(r)}')

    if compute_in_graph_coordinates:
        # First, compute the metric and inverse metric in graph coordinates (y).
        y_1 = sp.var('y^1')
        y_2 = sp.var('y^2')
        y = array([y_1, y_2])

        print(f'In graph coordinates {y}:')
        z_y = z.subs(r, sp.sqrt(y.dot(y)))

        emb_graph = array([y[0], y[1], z_y])
        Demb_graph = vp.symbolic.D(emb_graph, y)
        h_y = Demb_graph.T.dot(Demb_graph)
        print(f'h({y[0], y[1]}) = {h_y}')
        print(f'As LaTeX -- h({y[0], y[1]}): {sp.latex(h_y)}')

        h_y_inv = array(sp.Matrix(h_y.tolist()).inv().tolist())
        print(f'h^{{-1}}({y[0], y[1]}) = {h_y_inv}')
        print(f'As LaTeX -- h^{{-1}}({y[0], y[1]}): {sp.latex(h_y_inv)}')

    # Then, compute the metric, inverse metric, Christoffel symbols, and Gaussian curvature in polar coordinates.
    print(f'In polar coordinates {polar}:')
    emb_polar = array([r, theta, z])
    Demb_polar = vp.symbolic.D(emb_polar, polar)
    euclidean_cylindrical_coords = array([
        [1,    0, 0],
        [0, r**2, 0],
        [0,    0, 1],
    ])
    h_polar = Demb_polar.T.dot(euclidean_cylindrical_coords).dot(Demb_polar)
    print(f'h({r}) = {h_polar}')

    h_inv = array(sp.Matrix(h_polar.tolist()).inv().tolist())
    print(f'h^{{-1}}({r}) = {h_inv}')
    print()

    Gamma_polar = vp.riemannian.christoffel_symbol(h_polar, h_inv, polar)
    print(f'Gamma({r}) = {Gamma_polar}')
    print(f'As LaTeX -- Gamma({r}): {sp.latex(Gamma_polar)}')
    print()

    partial_r = array([Integer(1), Integer(0)])
    partial_theta = array([Integer(0), Integer(1)])

    nabla_theta_partial_theta = nabla(partial_theta, Gamma_polar, polar).dot(partial_theta)
    nabla_r_partial_theta = nabla(partial_theta, Gamma_polar, polar).dot(partial_r)
    nabla_r_nabla_theta_partial_theta = nabla(nabla_theta_partial_theta, Gamma_polar, polar).dot(partial_r)
    nabla_theta_nabla_r_partial_theta = nabla(nabla_r_partial_theta, Gamma_polar, polar).dot(partial_theta)

    K_computed = partial_r.dot(h_polar).dot(nabla_r_nabla_theta_partial_theta - nabla_theta_nabla_r_partial_theta) / (h_polar[0,0] * h_polar[1,1] - h_polar[0,1] * h_polar[1,0])
    K_computed = K_computed.simplify()
    print(f'K(r): {K_computed}')
    print(f'As LaTeX -- K(r): {sp.latex(K_computed)}')

def compute_formulas_for_graph(z_fn: Callable[[ndarray], Any]):
    """
    z_fn should be a function of r.
    """

    y_1 = sp.var('y^1')
    y_2 = sp.var('y^2')
    y = array([y_1, y_2])
    z = z_fn(y)

    print('--------------------------------------------------------')
    print(f'Graph z = {z_fn(y)}')

    emb = array([y[0], y[1], z])
    Demb = vp.symbolic.D(emb, y)
    h_y = Demb.T.dot(Demb)
    print(f'h({y[0], y[1]}) = {h_y}')
    print(f'As LaTeX -- h({y[0], y[1]}): {sp.latex(h_y)}')

    h_y_inv = array(sp.Matrix(h_y.tolist()).inv().tolist())
    print(f'h^{{-1}}({y[0], y[1]}) = {h_y_inv}')
    print(f'As LaTeX -- h^{{-1}}({y[0], y[1]}): {sp.latex(h_y_inv)}')

# def compute_gaussian_curvature_schwarzschild_slice():
#     # Calculate the Gaussian curvature of the r,theta slice of the Schwarzschild spacetime.

#     r_s = sp.var('r_s')
#     r = sp.var('r')
#     theta = sp.var('theta')
#     X = array([r, theta])

#     # Take the metric from the (-,+,+,+) convention for the Schwarzschild metric.
#     g = array([[r/(r - r_s), 0], [0, r**2]])
#     print(f'g = {g}')

#     g_inv = array(sp.Matrix(g.tolist()).inv().tolist())
#     print(f'g_inv = {g_inv}')

#     Gamma = vp.riemannian.christoffel_symbol(g, g_inv, X)
#     print(f'Gamma = {Gamma}')

#     partial_r = array([Integer(1), Integer(0)])
#     partial_theta = array([Integer(0), Integer(1)])

#     nabla_theta_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_theta)
#     nabla_r_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_r)
#     nabla_r_nabla_theta_partial_theta = nabla(nabla_theta_partial_theta, Gamma, X).dot(partial_r)
#     nabla_theta_nabla_r_partial_theta = nabla(nabla_r_partial_theta, Gamma, X).dot(partial_theta)

#     K_computed = partial_r.dot(g).dot(nabla_r_nabla_theta_partial_theta - nabla_theta_nabla_r_partial_theta) / (g[0,0] * g[1,1] - g[0,1] * g[1,0])
#     K_computed = K_computed.simplify()
#     print(f'K: {sp.latex(K_computed)}')

# def compute_gaussian_curvature_polar_metric():
#     # Calculate the Gaussian curvature of a diagonal metric in polar coordinates of a particular form.

#     r = sp.var('r')
#     theta = sp.var('theta')
#     X = array([r, theta])
#     f = sp.Function('f')

#     g = array([[f(r), 0], [0, r**2]])
#     print(f'g = {g}')

#     g_inv = array(sp.Matrix(g.tolist()).inv().tolist())
#     print(f'g_inv = {g_inv}')

#     Gamma = vp.riemannian.christoffel_symbol(g, g_inv, X)
#     print(f'Gamma = {Gamma}')

#     partial_r = array([Integer(1), Integer(0)])
#     partial_theta = array([Integer(0), Integer(1)])

#     nabla_theta_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_theta)
#     nabla_r_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_r)
#     nabla_r_nabla_theta_partial_theta = nabla(nabla_theta_partial_theta, Gamma, X).dot(partial_r)
#     nabla_theta_nabla_r_partial_theta = nabla(nabla_r_partial_theta, Gamma, X).dot(partial_theta)

#     K_computed = partial_r.dot(g).dot(nabla_r_nabla_theta_partial_theta - nabla_theta_nabla_r_partial_theta) / (g[0,0] * g[1,1] - g[0,1] * g[1,0])
#     K_computed = K_computed.simplify()
#     print(f'K: {sp.latex(K_computed)}')

def compute_gaussian_curvature_diagonal_polar_metric():
    # Calculate the Gaussian curvature of a diagonal metric in polar coordinates of a particular form.

    r = sp.var('r')
    theta = sp.var('theta')
    X = array([r, theta])
    K_H = sp.var('K_H')
    u = -4/K_H*(1 - r**2)**-2
    # u = sp.Function('u')
    # v = sp.Function('v')

    # g = array([[u(r), 0], [0, u(r)*r**2]])
    g = array([[u, 0], [0, u*r**2]])
    print(f'g = {g}')

    g_inv = array(sp.Matrix(g.tolist()).inv().tolist())
    print(f'g_inv = {g_inv}')

    Gamma = vp.riemannian.christoffel_symbol(g, g_inv, X)
    print(f'Gamma = {Gamma}')

    partial_r = array([Integer(1), Integer(0)])
    partial_theta = array([Integer(0), Integer(1)])

    nabla_theta_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_theta)
    nabla_r_partial_theta = nabla(partial_theta, Gamma, X).dot(partial_r)
    nabla_r_nabla_theta_partial_theta = nabla(nabla_theta_partial_theta, Gamma, X).dot(partial_r)
    nabla_theta_nabla_r_partial_theta = nabla(nabla_r_partial_theta, Gamma, X).dot(partial_theta)

    K_computed = partial_r.dot(g).dot(nabla_r_nabla_theta_partial_theta - nabla_theta_nabla_r_partial_theta) / (g[0,0] * g[1,1] - g[0,1] * g[1,0])
    K_computed = K_computed.simplify()
    print(f'K: {sp.latex(K_computed)}')

if __name__ == '__main__':
    # compute_gaussian_curvature_in_xy_coordinates()
    # compute_gaussian_curvature_in_rtheta_coordinates()
    # compute_gaussian_curvature_schwarzschild_slice()
    # compute_gaussian_curvature_polar_metric()
    # compute_gaussian_curvature_diagonal_polar_metric()

    compute_formulas_for_surface_of_revolution(lambda r: sp.Function('z')(r), compute_in_graph_coordinates=False)
    compute_formulas_for_surface_of_revolution(lambda r: -r**-1)
    compute_formulas_for_surface_of_revolution(lambda r: r**2 / 2)
    R = sp.var('R')
    compute_formulas_for_surface_of_revolution(lambda r: -sp.sqrt(R**2 - r**2), compute_in_graph_coordinates=False)
    compute_formulas_for_surface_of_revolution(lambda r: -sp.exp(-r**2 / 2), compute_in_graph_coordinates=False)
    r_s = sp.var('r_s')
    compute_formulas_for_surface_of_revolution(lambda r: 2*sp.sqrt(r_s * (r - r_s)), compute_in_graph_coordinates=False)

    compute_formulas_for_graph(lambda y: y[1]**2 / 2)
