import num_dual as nd
import numpy as np
import scipy as sc
import time

from .body import HyperelasticBody
from .fem2d import FEFunctionSpace2D
from .measure import DiscreteMeasure
from .material import HyperelasticMaterial
from .optimize import newtons_method_minimize
from .plot import HyperelasticBodyPlotter
from .spatial_manifold import SpatialManifold
from numpy import array, ndarray
from typing import Any, Callable, Optional

def analyze_hessian(
    L_name: str,
    L: Callable[[ndarray], Any],
    cfg: ndarray,
    *,
    metric_fn: Callable[[ndarray], ndarray],
    zero_eigenvalue_threshold: float = 1.0e-8,
    positive_eigenvalue_analyze_count: Optional[int] = 10,
):
    print(f'analyzing {L_name} at cfg with shape {cfg.shape}')

    L_cfg, DL, D2L = nd.hessian(L, cfg)
    DL = array(DL)
    D2L = array(D2L)

    # Verify that D2L is actually symmetric.
    assert np.allclose(D2L, D2L.T), f'D^2 {L_name} is not symmetric; max(abs(D^2 {L_name} - D^2 {L_name}.T)) = {np.max(np.abs(D2L - D2L.T))}'

    metric = metric_fn(cfg)
    # Note that the i-th eigenvector is in the i-th column of D2L_eigenvectors.
    D2L_eigenvalues, D2L_eigenvectors = sc.linalg.eigh(D2L, metric)

    # Sanity check that it produces metric-orthonormal eigenvectors.
    max_abs_error = np.max(np.abs(D2L_eigenvectors.T.dot(metric).dot(D2L_eigenvectors) - np.eye(D2L_eigenvectors.shape[1])))
    print(f'max_abs_error (sanity check for metric-orthonormal eigenvectors) = {max_abs_error}')
    assert max_abs_error < 1.0e-12, f'max_abs_error (sanity check for metric-orthonormal eigenvectors) = {max_abs_error}'

    # # Sanity check that the eigenvectors are indeed eigenvectors.
    # max_abs_error = np.max(np.abs(D2L.dot(D2L_eigenvectors) - metric.dot(D2L_eigenvectors).dot(np.diag(D2L_eigenvalues))))
    # print(f'max_abs_error (sanity check that the eigenvectors are indeed eigenvectors) = {max_abs_error}')
    # assert max_abs_error < 1.0e-7, f'max_abs_error (sanity check that the eigenvectors are indeed eigenvectors) = {max_abs_error}'

    # # Sanity check that the eigenvectors are indeed eigenvectors.
    # max_abs_error = np.max(np.abs(D2L_eigenvectors.T.dot(D2L).dot(D2L_eigenvectors) - np.diag(D2L_eigenvalues)))
    # print(f'max_abs_error (sanity check that the eigenvectors are indeed eigenvectors) = {max_abs_error}')
    # assert max_abs_error < 1.0e-7, f'max_abs_error (sanity check that the eigenvectors are indeed eigenvectors) = {max_abs_error}'

    grad_L = np.linalg.solve(metric, DL)
    norm_grad_L = DL.dot(grad_L)**0.5
    print(f'{L_name} = {L_cfg:e}')
    print(f'norm(grad_{L_name}) = {norm_grad_L:e}')

    # Note that the columns of D2L_eigenvectors are the eigenvectors.
    print(f'D^2 {L_name} eigenvalues = {D2L_eigenvalues.tolist()}')

    negative_eigenvalue_count = np.sum(D2L_eigenvalues < -zero_eigenvalue_threshold)
    print(f'negative_eigenvalue_count = {negative_eigenvalue_count} (where negative is defined to be less than {-zero_eigenvalue_threshold})')
    near_zero_eigenvalue_count = np.sum(np.abs(D2L_eigenvalues) <= zero_eigenvalue_threshold)
    print(f'near_zero_eigenvalue_count = {near_zero_eigenvalue_count} (where near-zero is defined to have abs value less than {zero_eigenvalue_threshold})')

    # Apply the inverse metric to each eigenvector that has negative eigenvalue in order
    # to derive geometric meaning from them.
    print(f'Analyzing eigenvectors of D^2 {L_name}')
    neg_eigenvalue_field = None
    positive_eigenvalue_analyzed_count = 0
    for i in range(len(D2L_eigenvalues)):
        eigenvalue = D2L_eigenvalues[i]
        eigenvector = D2L_eigenvectors[:,i]
        if eigenvalue < -zero_eigenvalue_threshold:
            sign_indicator = 'NEG'
            if neg_eigenvalue_field is None:
                neg_eigenvalue_field = eigenvector
        elif eigenvalue <= zero_eigenvalue_threshold:
            sign_indicator = '~0'
        else:
            sign_indicator = 'POS'
            positive_eigenvalue_analyzed_count += 1
            if positive_eigenvalue_analyzed_count > positive_eigenvalue_analyze_count:
                break
        print(f'eigenvec {i: 4d}: {eigenvalue:13e} ({sign_indicator:3}), along which... ', end='')

        # Compute a bunch of derivatives along this eigenvector.
        v = eigenvector
        L_along_v, DL_along_v, D2L_along_v, D3L_along_v = nd.third_derivative(lambda t: L(cfg + t*v), 0.0)
        print(f'{L_name}\'  = {DL_along_v:13e}, {L_name}\'\' = {D2L_along_v:13e}, {L_name}\'\'\' = {D3L_along_v:13e}, (near-zero-eigenvalue threshold is {zero_eigenvalue_threshold:13e})')

    return L_cfg, DL, D2L, grad_L, neg_eigenvalue_field

def solve_static_problem(
    *,
    M: HyperelasticMaterial,
    S: SpatialManifold,
    mass_specific_potential: Callable[[ndarray], Any],
    body_center_r_initial: float = 1.5,
    body_angle_initial: float = np.pi / 4.0,
    # The levels progress as vertex_count_v = (2,2), (3,3), (5,5), (9,9), etc.
    mesh_refinement_level_count: int = 3,
    vtu_filename_base: Optional[str] = None,
    show_plot: bool = False,
):
    F = FEFunctionSpace2D(
        domain_corner_v=array([[-0.5, -0.5], [0.5, 0.5]]),
        vertex_count_v=array([2, 2]),
    )
    measure = DiscreteMeasure.gauss_legendre_5x5_on_rectangle(F.domain_corner_v, F.vertex_count_v)

    theta = body_angle_initial
    R = array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T = array([body_center_r_initial, 0.0])
    b = HyperelasticBody.from_function(lambda X: R.dot(X) + T, F, measure, M, S, mass_specific_potential)

    print(f'STARTING BODY CONFIGURATION: b.cfg = {b.cfg.tolist()}')

    if vtu_filename_base is not None:
        mesh_refinement_index = 0
        print('Analyzing Hessian of initial body configuration...')
        initial_L, initial_DL, initial_D2L, grad_initial_L, neg_eigenvalue_field = analyze_hessian('L', lambda cfg: b.with_cfg(cfg).lagrangian(), b.cfg, metric_fn=lambda cfg: F.vector_field_metric(cfg, measure, S))
        b.write_vtu(filename='initial_body_configuration.vtu', body_mesh_vertex_count_v=array([17, 17]))
        print(f'wrote initial body configuration to vtu file {vtu_filename_base}')

    for mesh_refinement_index in range(1, mesh_refinement_level_count + 1):
        print(f'mesh_refinement_index: {mesh_refinement_index}, measure: {measure}')
        print(f'b.cfg.shape: {b.cfg.shape}')

        f = lambda cfg: b.with_cfg(cfg).lagrangian()

        if show_plot:
            hbp = HyperelasticBodyPlotter(b)
        else:
            hbp = None

        if True:
            if show_plot:
                hbp.show_nonblocking()

            intermediate_body_configuration = None

            plot_update_counter = 0

            def minimize_callback(X, f_X, Df_X, *, grad_f_X: Optional[ndarray] = None):
                nonlocal b
                nonlocal hbp
                nonlocal intermediate_body_configuration
                nonlocal plot_update_counter

                start = time.time()
                intermediate_body_configuration = X
                b.cfg = X
                if show_plot:
                    hbp.b.cfg = X

                # Take the average of the spatial point at the finite element mesh points.
                # Note that "center of gravity" here is a coordinate-dependent quantity, just meant
                # to see about where the body is in the solution logging.
                print(f'mean spatial point ("center of gravity"): {np.mean(X.reshape(b.function_space.vector_function_shape)[:,:,:,0,0].reshape(2,-1), axis=1)}')
                if grad_f_X is not None:
                    # Take the average of the gradient descent direction at the finite element mesh points.
                    print(f'mean gradient descent direction: {-np.mean(grad_f_X.reshape(b.function_space.vector_function_shape)[:,:,:,0,0].reshape(2,-1), axis=1)}')

                if True:
                    # Compute the variation of the stored energy along \partial_r / |\partial_r|.
                    def normalized_partial_r_function(X: ndarray) -> ndarray:
                        space_point = b.with_cfg(intermediate_body_configuration).phi(X)
                        metric = b.spatial_manifold.spatial_metric(space_point)
                        partial_r = space_point / np.linalg.norm(space_point)
                        norm_partial_r = np.sqrt(partial_r.dot(metric).dot(partial_r))
                        return partial_r / norm_partial_r
                    
                    normalized_partial_r = b.from_function(normalized_partial_r_function, F, measure, M, S, b.mass_specific_potential).cfg.flatten()

                    def stored_energy_variation(t: Any) -> Any:
                        return b.with_cfg(intermediate_body_configuration + t * normalized_partial_r).stored_energy()
                    
                    stored_energy_variation_along_normalized_partial_r = nd.first_derivative(stored_energy_variation, 0.0)[1]
                    print(f'stored_energy_variation_along_normalized_partial_r = {stored_energy_variation_along_normalized_partial_r}')

                if show_plot:
                    # Only update the plot every 10 iterations, since plot update is slow.
                    if plot_update_counter % 10 == 0:
                        L, DL, D2L, grad_L, neg_eigenvalue_field = analyze_hessian('L', lambda cfg: b.with_cfg(cfg).lagrangian(), intermediate_body_configuration, metric_fn=lambda cfg: F.vector_field_metric(cfg, measure, S))

                        # TODO: Figure out how to call this function asynchronously so it doesn't block the optimization.
                        start = time.time()
                        hbp.update(neg_eigenvalue_field=neg_eigenvalue_field)
                        print(f'plot update took {time.time() - start} seconds')
                    plot_update_counter += 1

            try:
                optimized_body_configuration = newtons_method_minimize(
                    f,
                    b.cfg,
                    metric_fn=lambda cfg: F.vector_field_metric(cfg, measure, S),
                    callback=minimize_callback,
                    maxiter=2000,
                    step_size_factor=1.0,
                    stop_below_norm_grad_f=1.0e-16,
                )
            except KeyboardInterrupt:
                print('KeyboardInterrupt -- using intermediate body configuration')
                optimized_body_configuration = intermediate_body_configuration

            b.cfg = optimized_body_configuration
            print('OPTIMIZATION ENDED')

            print('Analyzing Hessian of optimized body configuration...')
            optimized_L, optimized_DL, optimized_D2L, grad_optimized_L, neg_eigenvalue_field = analyze_hessian('L', lambda cfg: b.with_cfg(cfg).lagrangian(), optimized_body_configuration, metric_fn=lambda cfg: F.vector_field_metric(cfg, measure, S))

            if show_plot:
                hbp.b.cfg = optimized_body_configuration
                hbp.update(neg_eigenvalue_field=neg_eigenvalue_field)

        if vtu_filename_base is not None:
            b.with_cfg(optimized_body_configuration).write_vtu(filename=f'{vtu_filename_base}.{mesh_refinement_index:02d}.vtu', body_mesh_vertex_count_v=array([17, 17]))
            print(f'wrote optimized body configuration to vtu file {vtu_filename_base}')

        # Refine the mesh by subdividing it, if there is a refinement level left.
        if mesh_refinement_index >= mesh_refinement_level_count:
            break

        print('----------------------------------------------------------')
        print(f'REFINING MESH (level {mesh_refinement_index} -> {mesh_refinement_index + 1})...')
        print(f'before refinement: b.function_space.vertex_count_v = {b.function_space.vertex_count_v}')
        print(f'before refinement: b.function_space.vector_function_dim = {b.function_space.vector_function_dim}')
        b_lagrangian_before_subdivision = b.lagrangian()
        print(f'b_lagrangian_before_subdivision: {b_lagrangian_before_subdivision}')
        
        # Subdivide the mesh by splitting each element into 4.
        b_subdivided = b.subdivided(refine_measure=True)
        # # Increase vertex count along each axis by 1, instead of splitting each element into 4.
        # b_subdivided = b.subdivided_linear(refine_measure=True)

        print(f'after refinement: b_subdivided.function_space.vertex_count_v = {b_subdivided.function_space.vertex_count_v}')
        print(f'after refinement: b_subdivided.function_space.vector_function_dim = {b_subdivided.function_space.vector_function_dim}')

        # Check that the subdivision is correct by comparing the spatial points at the integration points.
        b_spatial_points = np.apply_along_axis(b.phi, 2, measure.point_t)
        b_subdivided_spatial_points = np.apply_along_axis(b_subdivided.phi, 2, measure.point_t)
        max_abs_error = np.max(np.abs(b_spatial_points - b_subdivided_spatial_points))
        print(f'refined mesh spatial point delta: max_abs_error = {max_abs_error}')
        assert np.allclose(b_spatial_points, b_subdivided_spatial_points), f'b_spatial_points = {b_spatial_points.tolist()}, b_subdivided_spatial_points = {b_subdivided_spatial_points.tolist()}'

        # Also check that the Lagrangian is the same.  Have to use the DiscreteMeasure from the subdivided space.
        b.measure = b_subdivided.measure
        b_lagrangian = b.lagrangian()
        b_subdivided_lagrangian = b_subdivided.lagrangian()
        print(f'b_lagrangian: {b_lagrangian}, b_subdivided_lagrangian: {b_subdivided_lagrangian}')
        max_abs_error = np.max(np.abs(b_lagrangian - b_subdivided_lagrangian))
        print(f'refined mesh Lagrangian delta: max_abs_error = {max_abs_error}')
        assert np.allclose(b_lagrangian, b_subdivided_lagrangian), f'b_lagrangian = {b_lagrangian}, b_subdivided_lagrangian = {b_subdivided_lagrangian}'
       
        b = b_subdivided
        F = b.function_space
        measure = b.measure
        print(f'new mesh has vertex_count_v = {F.vertex_count_v}, F.vector_function_dim = {F.vector_function_dim}')

        if show_plot:
            hbp.plotter.close()

    print('solve_static_problem terminated')

    if show_plot:
        hbp.show_blocking()

