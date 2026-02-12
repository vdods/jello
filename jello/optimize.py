import numpy as np
import num_dual as nd
import scipy as sc
import scipy.integrate
import time

from numpy import array, ndarray
from typing import Any, Callable, Optional, Tuple

def restricted_newtons_method_minimize(
    f: Callable[[ndarray], Any],
    X_initial: ndarray,
    *,
    metric_fn: Callable[[ndarray], ndarray],
    callback: Optional[Callable[[ndarray, Any, ndarray], None]] = None,
    maxiter: int = 100,
    step_size_factor: float = 1.0,
    stop_below_norm_grad_f: float = 1.0e-10,
    zero_eigenvalue_threshold: float = 1.0e-10,
) -> Any:
    """
    Perform a modified form of Newton's method minimization in which only the
    positive-definite directions of the Hessian are used to compute the search direction.
    This is not expected to work for every possible function, but if it can get to a convex
    region of f, then it may converge to a local minimum.  A line search is used along
    each step to very precisely find the local minimum along the search direction without
    escaping the local basin of attraction.
    """

    assert stop_below_norm_grad_f > 0.0

    X = np.copy(X_initial)
    V = np.zeros_like(X)

    def f_along_V(t: Any) -> Any:
        nonlocal f, X, V
        return f(X + t * V)

    for iter in range(maxiter):
        start = time.time()
        f_X, Df_X, D2f_X = nd.hessian(f, X)
        end = time.time()
        print(f'evaluation of hessian of f took {end - start} seconds')
        if not isinstance(Df_X, ndarray):
            Df_X = array(Df_X)
        if not isinstance(D2f_X, ndarray):
            D2f_X = array(D2f_X)
        if callback is not None:
            callback(X, f_X, Df_X)

        metric = metric_fn(X)
        grad_f = np.linalg.solve(metric, Df_X)
        norm_grad_f = (grad_f @ metric @ grad_f)**0.5
        max_abs_grad_f = np.max(np.abs(grad_f))
        print(f'Newton -- iter = {iter}, f_X = {f_X:e}, norm(grad_f) = {norm_grad_f:e}, max_abs_grad_f = {max_abs_grad_f:e}')
        if norm_grad_f <= stop_below_norm_grad_f:
            print('STOPPING newtons_method_minimize because norm(grad_f) <= stop_below_norm_grad_f')
            break

        # Use Newton along the positive-definite subspace.
        D2f_X_eigenvalues, D2f_X_eigenvectors = sc.linalg.eigh(D2f_X, metric)
        if False:
            # Sanity checks
            assert np.allclose(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors, np.diag(D2f_X_eigenvalues), atol=1.0e-6), f'Sanity check for diagonalization of D2f_X; max(abs(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors - np.diag(D2f_X_eigenvalues))) = {np.max(np.abs(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors - np.diag(D2f_X_eigenvalues)))}'
            assert np.allclose(D2f_X_eigenvectors.T @ metric @ D2f_X_eigenvectors, np.eye(D2f_X_eigenvectors.shape[1])), 'Sanity check for metric-orthonormal eigenvectors failed'
            assert np.allclose(D2f_X @ D2f_X_eigenvectors, metric @ D2f_X_eigenvectors @ np.diag(D2f_X_eigenvalues)), 'Sanity check for definition of generalized eigendecomposition problem failed'
        print(f'D2f_X_eigenvalues = {D2f_X_eigenvalues.tolist()}')

        # print('Zeroing-out non-positive eigenvalues and inverting positive eigenvalues')
        positive_eigenvalue_count = 0
        nonpositive_eigenvalue_count = 0
        for i in range(len(D2f_X_eigenvalues)):
            if D2f_X_eigenvalues[i] < zero_eigenvalue_threshold:
                nonpositive_eigenvalue_count += 1
                D2f_X_eigenvalues[i] = 0.0
            else:
                positive_eigenvalue_count += 1
                D2f_X_eigenvalues[i] = 1.0 / D2f_X_eigenvalues[i]
        print(f'positive_eigenvalue_count = {positive_eigenvalue_count}, nonpositive_eigenvalue_count = {nonpositive_eigenvalue_count}')
        V[:] = D2f_X_eigenvectors @ np.diag(D2f_X_eigenvalues) @ D2f_X_eigenvectors.T @ Df_X
        V[:] *= -step_size_factor

        Df_X_along_V = Df_X.dot(V)
        assert Df_X_along_V <= 0.0, f'Df_X_along_V = {Df_X_along_V} > 0.0'

        norm_V = V.dot(metric).dot(V)**0.5
        print(f'Newton -- line search based on hessian computation -- norm(V) = {norm_V:e}')
        success_status, t, f_along_V_t, Df_along_V_t, D2f_along_V_t, D3f_along_V_t = third_order_line_search_local_minimize(f_along_V, 0.0, max_step_size=0.025, maxiter=80)
        print(f'Newton -- iter = {iter}, line search returned t = {t} -- updating X')
        X[:] += t * V
        if t == 0.0:
            print(f'Newton -- iter = {iter}, step size was 0 -- stopping')
            break

    print(f'Newton -- iter = {iter} -- terminating and returning final X = {X.tolist()}')
    return X

def refine_along_eigenvectors(
    f: Callable[[ndarray], Any],
    X_initial: ndarray,
    *,
    metric_fn: Callable[[ndarray], ndarray],
    callback: Optional[Callable[[ndarray, Any, ndarray], None]] = None,
    maxiter: int = 10,
    step_size_factor: float = 1.0,
    stop_below_norm_grad_f: float = 1.0e-10,
    zero_eigenvalue_threshold: float = 1.0e-12,
    Df_tolerance: float = 1.0e-14,
) -> ndarray:
    assert stop_below_norm_grad_f > 0.0

    X = np.copy(X_initial)
    V = np.zeros_like(X)

    def f_along_V(t: Any) -> Any:
        nonlocal f, X, V
        return f(X + t * V)

    for iter in range(maxiter):
        start = time.time()
        f_X, Df_X, D2f_X = nd.hessian(f, X)
        end = time.time()
        print(f'evaluation of hessian of f took {end - start} seconds')
        if not isinstance(Df_X, ndarray):
            Df_X = array(Df_X)
        if not isinstance(D2f_X, ndarray):
            D2f_X = array(D2f_X)
        if callback is not None:
            callback(X, f_X, Df_X)

        metric = metric_fn(X)
        grad_f = np.linalg.solve(metric, Df_X)
        norm_grad_f = (grad_f @ metric @ grad_f)**0.5
        max_abs_grad_f = np.max(np.abs(grad_f))
        print(f'refine_along_eigenvectors -- iter = {iter}, f_X = {f_X:e}, norm(grad_f) = {norm_grad_f:e}, max_abs_grad_f = {max_abs_grad_f:e}')
        if norm_grad_f <= stop_below_norm_grad_f:
            print('STOPPING refine_along_eigenvectors because norm(grad_f) <= stop_below_norm_grad_f')
            break

        D2f_X_eigenvalues, D2f_X_eigenvectors = sc.linalg.eigh(D2f_X, metric)
        print(f'refine_along_eigenvectors -- iter = {iter}, D2f_X_eigenvalues = {D2f_X_eigenvalues.tolist()}')
        # Refine along the eigenvectors of the Hessian.
        # TODO: Maybe order them differently.
        for j in range(len(D2f_X_eigenvalues)):
            i = len(D2f_X_eigenvalues) - 1 - j
            # assert D2f_X_eigenvalues[i] >= -zero_eigenvalue_threshold, f'encountered negative eigenvalue D2f_X_eigenvalues[{i}] = {D2f_X_eigenvalues[i]}, less than -zero_eigenvalue_threshold = {-zero_eigenvalue_threshold}'
            if D2f_X_eigenvalues[i] <= zero_eigenvalue_threshold:
                if D2f_X_eigenvalues[i] < -zero_eigenvalue_threshold:
                    print(f'refine_along_eigenvectors -- iter = {iter}, encountered negative eigenvalue D2f_X_eigenvalues[{i}] = {D2f_X_eigenvalues[i]}, less than -zero_eigenvalue_threshold = {-zero_eigenvalue_threshold}')
                continue
            V[:] = D2f_X_eigenvectors[:, i]
            # norm_V = V.dot(metric).dot(V)**0.5
            print(f'refine_along_eigenvectors -- iter = {iter}, refining along eigenvector {i}')
            success_status, t, f_along_V_t, Df_along_V_t, D2f_along_V_t, D3f_along_V_t = third_order_line_search_local_minimize(f_along_V, 0.0, Df_tolerance=Df_tolerance, max_step_size=step_size_factor*0.025, maxiter=80)
            print(f'refine_along_eigenvectors -- iter = {iter}, line search returned t = {t} -- updating X')
            X[:] += t * V

    print(f'refine_along_eigenvectors -- iter = {iter} -- terminating and returning final X = {X.tolist()}')
    return X

def newtons_method_minimize(
    f: Callable[[ndarray], Any],
    X_initial: ndarray,
    *,
    metric_fn: Callable[[ndarray], ndarray],
    callback: Optional[Callable[[ndarray, Any, ndarray], None]] = None,
    maxiter: int = 100,
    step_size_factor: float = 1.0,
    gradient_descent_step_size_factor: float = 1.0,
    stop_below_norm_grad_f: float = 1.0e-10,
) -> Any:
    """
    Perform a hybrid Newton/"gradient"-descent method minimization.
    """

    assert stop_below_norm_grad_f > 0.0

    X = np.copy(X_initial)
    V = np.zeros_like(X)
    V_pos = np.zeros_like(X)
    V_neg = np.zeros_like(X)
    preallocated_X_min = np.zeros_like(X)
    preallocated_X_trial = np.zeros_like(X)
    iter = 0
    method_index = 0
    # Bit of a hacky way to detect if both passes took 0 step.
    zero_step_count = 0

    def f_along_V(t: Any) -> Any:
        nonlocal f, X, V
        return f(X + t * V)

    for iter in range(maxiter):
        start = time.time()
        print(f'type(f) = {type(f)}, type(X) = {type(X)}, X.shape = {X.shape}')
        f_X, Df_X, D2f_X = nd.hessian(f, X)
        end = time.time()
        print(f'evaluation of hessian of f took {end - start} seconds')
        if not isinstance(Df_X, ndarray):
            Df_X = array(Df_X)
        if not isinstance(D2f_X, ndarray):
            D2f_X = array(D2f_X)
        if callback is not None:
            callback(X, f_X, Df_X)

        max_abs_Df = np.max(np.abs(Df_X))
        metric = metric_fn(X)
        grad_f = np.linalg.solve(metric, Df_X)
        norm_grad_f = grad_f.dot(metric).dot(grad_f)**0.5
        print(f'Newton -- iter = {iter}, f_X = {f_X:e}, norm(grad_f) = {norm_grad_f:e}, max_abs_Df = {max_abs_Df:e}')
        if norm_grad_f <= stop_below_norm_grad_f:
            print('STOPPING newtons_method_minimize because norm(grad_f) <= stop_below_norm_grad_f')
            break
        normalized_grad_f = grad_f / norm_grad_f

        use_hybrid_method = True

        if method_index % 2 == 1:
            D2f_X_eigenvalues, D2f_X_eigenvectors = sc.linalg.eigh(D2f_X, metric)
            assert np.allclose(D2f_X_eigenvectors.T.dot(metric).dot(D2f_X_eigenvectors), np.eye(D2f_X_eigenvectors.shape[1])), 'Sanity check for metric-orthonormal eigenvectors failed'
            print(f'D2f_X_eigenvalues = {D2f_X_eigenvalues.tolist()}')

            # Analyze normalized_grad_f along the eigenvectors of D2f_X.
            normalized_grad_f_dot_metric_dot_eigenvectors = normalized_grad_f.dot(metric).dot(D2f_X_eigenvectors)
            for i, (eigenvalue, normalized_grad_f_dot_metric_dot_eigenvector) in enumerate(zip(D2f_X_eigenvalues, normalized_grad_f_dot_metric_dot_eigenvectors)):
                print(f'eigenvalue {i:3d} = {eigenvalue:+e}, normalized_grad_f.dot(metric).dot(i-th eigenvector) = {normalized_grad_f_dot_metric_dot_eigenvector:+e}')

            if D2f_X_eigenvalues[0] < 0.0:
                V[:] = D2f_X_eigenvectors[:, 0]
                norm_V = V.dot(metric).dot(V)**0.5
                print(f'Newton -- iter = {iter}, the first eigenvalue is negative -- doing gradient descent pass -- norm(V) = {norm_V:e}')
                success_status, t, f_along_V_t, Df_along_V_t, D2f_along_V_t, D3f_along_V_t = third_order_line_search_local_minimize(f_along_V, 0.0, max_step_size=gradient_descent_step_size_factor*0.01, maxiter=500)
                # if np.abs(t) <= 1.0e-18:
                #     print(f'iter = {iter}, t = {t:+e} is too small -- stopping')
                #     break
                # else:
                #     print(f'iter = {iter}, gradient descent line search returned t = {t:+e} -- updating X')
                print(f'Newton -- iter = {iter}, gradient descent line search returned t = {t:+e} -- updating X')
                X[:] += t * V
                if t == 0.0:
                    zero_step_count += 1
                    if zero_step_count >= 2:
                        print(f'Newton -- iter = {iter}, both passes took 0 step -- stopping')
                        break
                else:
                    zero_step_count = 0

            method_index += 1
            continue

        if method_index % 2 == 0:
            # H = D2f_X

            if not use_hybrid_method:
                # Naive Newton's method that includes non-positive-definite directions.
                V[:] = np.linalg.solve(D2f_X, Df_X)
            else:
                # Hybrid Newton/gradient-descent method.  Use Newton along the positive-definite subspace,
                # and gradient descent along its complement.
                zero_eigenvalue_threshold = 1.0e-10
                D2f_X_eigenvalues, D2f_X_eigenvectors = sc.linalg.eigh(D2f_X, metric)
                # assert np.allclose(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors, np.diag(D2f_X_eigenvalues), atol=1.0e-6), f'Sanity check for diagonalization of D2f_X; max(abs(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors - np.diag(D2f_X_eigenvalues))) = {np.max(np.abs(D2f_X_eigenvectors.T @ D2f_X @ D2f_X_eigenvectors - np.diag(D2f_X_eigenvalues)))}'
                # assert np.allclose(D2f_X_eigenvectors.T @ metric @ D2f_X_eigenvectors, np.eye(D2f_X_eigenvectors.shape[1])), 'Sanity check for metric-orthonormal eigenvectors failed'
                # assert np.allclose(D2f_X @ D2f_X_eigenvectors, metric @ D2f_X_eigenvectors @ np.diag(D2f_X_eigenvalues)), 'Sanity check for definition of generalized eigendecomposition problem failed'
                print(f'D2f_X_eigenvalues = {D2f_X_eigenvalues.tolist()}')

                if True:
                    print('Replacing non-positive eigenvalues with 0.0; inverting positive eigenvalues')
                    positive_eigenvalue_count = 0
                    nonpositive_eigenvalue_count = 0
                    for i in range(len(D2f_X_eigenvalues)):
                        if D2f_X_eigenvalues[i] < zero_eigenvalue_threshold:
                            nonpositive_eigenvalue_count += 1
                            # D2f_X_eigenvalues[i] = 1.0
                            D2f_X_eigenvalues[i] = 0.0
                        else:
                            positive_eigenvalue_count += 1
                            D2f_X_eigenvalues[i] = 1.0 / D2f_X_eigenvalues[i]
                        # if D2f_X_eigenvalues[i] < 0.0:
                        #     D2f_X_eigenvalues[i] *= -1.0
                    print(f'positive_eigenvalue_count = {positive_eigenvalue_count}')
                    print(f'nonpositive_eigenvalue_count = {nonpositive_eigenvalue_count}')
                    V[:] = D2f_X_eigenvectors @ np.diag(D2f_X_eigenvalues) @ D2f_X_eigenvectors.T @ Df_X
                else:
                    D2f_X_eigenvalues = array(D2f_X_eigenvalues)
                    D2f_X_eigenvectors = array(D2f_X_eigenvectors)
                    negative_eigenvalue_count = np.sum(D2f_X_eigenvalues <= 1.0e-14)
                    negative_eigenvalues = D2f_X_eigenvalues[:negative_eigenvalue_count]
                    negative_eigenvectors = D2f_X_eigenvectors[:, :negative_eigenvalue_count]
                    positive_eigenvalues = D2f_X_eigenvalues[negative_eigenvalue_count:]
                    assert np.all(positive_eigenvalues > zero_eigenvalue_threshold), f'positive_eigenvalues = {positive_eigenvalues}'
                    positive_eigenvectors = D2f_X_eigenvectors[:, negative_eigenvalue_count:]
                    print(f'negative_eigenvalue_count = {negative_eigenvalues.shape[0]}')
                    print(f'negative_eigenvalues = {negative_eigenvalues}')
                    print(f'positive_eigenvalue_count = {positive_eigenvalues.shape[0]}')
                    print(f'positive_eigenvalues = {positive_eigenvalues}')
                    print(f'positive_eigenvectors.shape = {positive_eigenvectors.shape}')
                    print(f'positive_eigenvectors.T.shape = {positive_eigenvectors.T.shape}')
                    # Newton's method along the positive-definite subspace.
                    V_pos[:] = positive_eigenvectors.dot(np.diag(1.0 / positive_eigenvalues)).dot(positive_eigenvectors.T).dot(Df_X)
                    V_pos[:] *= -step_size_factor
                    # Gradient descent along the negative-definite subspace.
                    V_neg[:] = negative_eigenvectors.dot(negative_eigenvectors.T).dot(Df_X)
                    # gradient_descent_step_size_factor = 1.0e-2
                    V_neg[:] *= -1.0

        V[:] *= -step_size_factor
        method_index += 1

        Df_X_along_V = Df_X.dot(V)
        if use_hybrid_method:
            assert Df_X_along_V <= 0.0, f'Df_X_along_V = {Df_X_along_V} > 0.0'
        else:
            if Df_X_along_V > 0.0:
                print(f'Newton -- iter = {iter}, f_X = {f_X}, Df_X_along_V = {Df_X_along_V} > 0.0 -- flipping sign of V')
                V[:] = -V

        norm_V = V.dot(metric).dot(V)**0.5
        print(f'Newton -- line search based on hessian computation -- norm(V) = {norm_V:e}')
        success_status, t, f_along_V_t, Df_along_V_t, D2f_along_V_t, D3f_along_V_t = third_order_line_search_local_minimize(f_along_V, 0.0, max_step_size=0.025, maxiter=80)
        print(f'Newton -- iter = {iter}, line search returned t = {t} -- updating X')
        X[:] += t * V
        if t == 0.0:
            zero_step_count += 1
            if zero_step_count >= 2:
                print(f'Newton -- iter = {iter}, both passes took 0 step -- stopping')
                break
        else:
            zero_step_count = 0

    return X

def third_order_line_search_local_minimize(
    f: Callable[[Any], Any],
    t_initial: Any,
    *,
    Df_tolerance: float = 1.0e-14,
    max_step_size: float,
    maxiter: int = 1000,
) -> Tuple[bool, Any, Any, Any, Any, Any]:
    """
    Returns (success_status, t_min, f(t_min), f'(t_min), f''(t_min), f'''(t_min)).
    """

    assert Df_tolerance > 0.0, f'Df_tolerance = {Df_tolerance}'
    assert max_step_size > 0.0, f'max_step_size = {max_step_size}'
    assert maxiter > 0, f'maxiter = {maxiter}'

    # Somewhat arbitrary.
    zero_tolerance = 1.0e-14
    h_zero_tolerance = 1.0e-16
    # Somewhat arbitrary.
    default_step_size = max_step_size * 0.1

    success_status = False

    t = t_initial
    for iter in range(maxiter):
        f_t, Df_t, D2f_t, D3f_t = nd.third_derivative(f, t)
        print(f'3OLS - iter: {iter:3d}, t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
        
        a = D3f_t
        b = D2f_t
        c = Df_t
        # p(h) := c + h * b + 1/2 * h^2 * a
        # 3rd order Taylor expansion of f' at t is f'(t + h) = p(h) + O(h^3).

        def pick_h_to_the_left() -> Tuple[float, str]:
            h, h_label = -max_step_size, '-h_M'
            abs_a = np.abs(a)
            if abs_a > zero_tolerance:
                h_N = -b / a
                if h < h_N < -h_zero_tolerance:
                    h, h_label = h_N, 'h_N'
            delta = b**2 - 2*a*c
            if delta >= 0.0:
                sqrt_delta = delta**0.5
                h_minus = (-b - sqrt_delta) / abs_a
                if h < h_minus < -h_zero_tolerance:
                    h, h_label = h_minus, 'h_minus'
                h_plus = (-b + sqrt_delta) / abs_a
                if h < h_plus < -h_zero_tolerance:
                    h, h_label = h_plus, 'h_plus'
            assert h < 0.0
            # print(f'pick_h_to_the_left: h = {h:+22.17e}, h_label = {h_label}')
            return h, h_label
        
        def pick_h_to_the_right() -> Tuple[float, str]:
            h, h_label = max_step_size, 'h_M'
            abs_a = np.abs(a)
            if abs_a > zero_tolerance:
                h_N = -b / a
                if h_zero_tolerance < h_N < h:
                    h, h_label = h_N, 'h_N'
            delta = b**2 - 2*a*c
            if delta >= 0.0:
                sqrt_delta = delta**0.5
                h_minus = (-b - sqrt_delta) / abs_a
                if h_zero_tolerance < h_minus < h:
                    h, h_label = h_minus, 'h_minus'
                h_plus = (-b + sqrt_delta) / abs_a
                if h_zero_tolerance < h_plus < h:
                    h, h_label = h_plus, 'h_plus'
            assert h > 0.0
            # print(f'pick_h_to_the_right: h = {h:+22.17e}, h_label = {h_label}')
            return h, h_label

        if c > zero_tolerance:
            # local min of f is to the left of t.
            h, h_label = pick_h_to_the_left()
        elif c < -zero_tolerance:
            # local min of f is to the right of t.
            h, h_label = pick_h_to_the_right()
        else:
            # t is a critical point of f.
            if b > zero_tolerance:
                # t is a local min of f.
                print(f'third_order_line_search_local_minimize: SUCCESS -- converged because Df_t = {Df_t:+22.17e} <= Df_tolerance = {Df_tolerance:+22.17e}; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
                success_status = True
                break
            elif b < -zero_tolerance:
                # t is a local max of f.
                # print(f'third_order_line_search_local_minimize: FAILURE -- current t is a local max -- terminating with failure; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
                # TEMP HACK -- unseat local max if possible.
                # TODO: Try not doing this, because this could potentially escape the local basin of attraction.
                if (a < -zero_tolerance):
                    h, h_label = pick_h_to_the_right()
                    print(f'third_order_line_search_local_minimize: current t is a local max -- resolving (to the right) via 3rd derivative; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}, h = {h:+22.17e}, h_label = {h_label}')
                elif (a > zero_tolerance):
                    h, h_label = pick_h_to_the_left()
                    print(f'third_order_line_search_local_minimize: current t is a local max -- resolving (to the left) via 3rd derivative; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}, h = {h:+22.17e}, h_label = {h_label}')
                else:
                    print(f'third_order_line_search_local_minimize: FAILURE -- current t is a local max -- terminating with failure; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
                    break
            else:
                # t is a degenerate critical point of f.
                if a > zero_tolerance:
                    # Strictly speaking, f is increasing on both sides of t, and we must terminate with failure.
                    # But realistically, the desired local min is to the left of t.
                    h, h_label = pick_h_to_the_left()
                    print(f'third_order_line_search_local_minimize: current t is a degenerate critical point -- resolving (to the left) via 3rd derivative; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}, h = {h:+22.17e}, h_label = {h_label}')
                elif a < -zero_tolerance:
                    # Strictly speaking, f is decreasing on both sides of t, and we must terminate with failure.
                    # But realistically, the desired local min is to the right of t.
                    h, h_label = pick_h_to_the_right()
                    print(f'third_order_line_search_local_minimize: current t is a degenerate critical point -- resolving (to the right) via 3rd derivative; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}, h = {h:+22.17e}, h_label = {h_label}')
                else:
                    # The 3-jet of f is zero, and so we can't make any meaningful step.
                    print(f'third_order_line_search_local_minimize: FAILURE -- current t has 3-jet equal to 0 -- terminating with failure; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
                    break

        t_trial = t + h
        # print(f'iter: {iter:3d}, t = {t:+22.17e}, t_trial = {t_trial:+22.17e}, abs(t_trial - t) = {np.abs(t_trial - t):+22.17e}, h = {h:+22.17e}, h_label = {h_label}')
        if t_trial == t:
            print(f'third_order_line_search_local_minimize: FAILURE -- t + h == t -- terminating with failure; t = {t:+22.17e}, f_t = {f_t:+22.17e}, Df_t = {Df_t:+22.17e}, D2f_t = {D2f_t:+22.17e}, D3f_t = {D3f_t:+22.17e}')
            break
        f_t_trial, Df_t_trial, D2f_t_trial = nd.second_derivative(f, t_trial)
        # print(f'iter: {iter}, t_trial = {t_trial}, f_t_trial = {f_t_trial}, Df_t_trial = {Df_t_trial}, D2f_t_trial = {D2f_t_trial}')

        if (Df_t < -Df_tolerance and Df_tolerance < Df_t_trial) or (Df_t_trial < -Df_tolerance and Df_tolerance < Df_t):
            # By intermediate value theorem, there should be at least one local min of f in (t, t_trial).
            # Assume for now that max_step_size is sufficiently small such that there is only one local min.
            # Update t based on Df_t and Df_t_trial.
            # Compute the zero of the line going through the point (t, Df_t) and (t_trial, Df_t_trial).
            # 0 = y(x) = Df_t + (Df_t_trial - Df_t) / h * x
            # x = -Df_t * h / (Df_t_trial - Df_t)
            print(f'third_order_line_search_local_minimize: bracketed a local min of f; t = {t:+22.17e}, t_trial = {t_trial:+22.17e}, h = {h:+22.17e}, h_label = {h_label}, Df_t = {Df_t:+22.17e}, Df_t_trial = {Df_t_trial:+22.17e}')

            t_root, result = sc.optimize.bisect(lambda s: nd.first_derivative(f, s)[1], min(t, t_trial), max(t, t_trial), full_output=True)
            print(f'third_order_line_search_local_minimize: bisection method returned t_root = {t_root:+22.17e}, result = {result}')
            f_root, Df_root = nd.first_derivative(f, t_root)
            if np.abs(Df_root) < Df_tolerance:
                print(f'third_order_line_search_local_minimize: SUCCESS -- converged because Df_root = {Df_root:+22.17e} <= Df_tolerance = {Df_tolerance:+22.17e}; t = {t:+22.17e}, t_trial = {t_trial:+22.17e}, h = {h:+22.17e}, h_label = {h_label}, Df_t = {Df_t:+22.17e}, Df_t_trial = {Df_t_trial:+22.17e}')
                success_status = True
            else:
                print(f'third_order_line_search_local_minimize: FAILURE -- bisection method returned Df_root = {Df_root:+22.17e} > Df_tolerance = {Df_tolerance:+22.17e}; t = {t:+22.17e}, t_trial = {t_trial:+22.17e}, h = {h:+22.17e}, h_label = {h_label}, Df_t = {Df_t:+22.17e}, Df_t_trial = {Df_t_trial:+22.17e}')

            t = t_root
            break            
        else:
            t = t_trial

    if iter >= maxiter:
        # Ensure all values are up to date with respect to the final point before returning.
        f_t, Df_t, D2f_t, D3f_t = nd.third_derivative(f, t)

    return success_status, t, f_t, Df_t, D2f_t, D3f_t

def accelerated_gradient_descent(
    f: Callable[[ndarray], Any],
    X_initial: ndarray,
    *,
    metric_fn: Callable[[ndarray], ndarray],
    callback: Optional[Callable[[ndarray, Any, ndarray], None]] = None,
    maxiter: int = 100,
    damping_factor: float = 100.0,
    force_factor: float = 1.0,
) -> Any:
    assert len(X_initial.shape) == 1, f'X_initial.shape = {X_initial.shape}'
    n = X_initial.shape[0]

    def vector_field(t: Any, XV: ndarray) -> ndarray:
        assert XV.shape == (2*n,), f'XV.shape = {XV.shape}'
        X = XV[:n]
        V = XV[n:]
        metric = metric_fn(X)
        kinetic_energy_X = V.dot(metric).dot(V) / 2
        f_X, Df_X = nd.gradient(f, X)
        total_energy_X = f_X + kinetic_energy_X
        print(f'kinetic_energy_X = {kinetic_energy_X:+22.17e}, lagrangian_X = {f_X:+22.17e}, total_energy_X = {total_energy_X:+22.17e}')
        grad_f_X = np.linalg.solve(metric, Df_X)
        retval = np.zeros((2*n,), dtype=X.dtype)
        retval[:n] = V
        retval[n:] = -damping_factor * V - force_factor * grad_f_X
        return retval

    X = np.copy(X_initial)
    Df_X = np.zeros_like(X_initial)
    # XV = np.zeros((2*X.shape[0],), dtype=X.dtype)
    XV_initial = np.zeros((2*n,), dtype=X.dtype)
    XV_initial[:n] = X_initial

    if callback is not None:
        f_X, Df_X[:] = nd.gradient(f, X)
        callback(X, f_X, Df_X)

    rk4 = sc.integrate.RK45(vector_field, 0.0, XV_initial, 1.0)
    for iter in range(maxiter):
        # TODO: Maybe do a 3rd order line search along the RK4 step.
        rk4.step()
        if callback is not None:
            X[:] = rk4.y[:n]
            f_X, Df_X[:] = nd.gradient(f, X)
            callback(X, f_X, Df_X)
        if rk4.status == 'finished' or rk4.status == 'failed':
            print(f'accelerated_gradient_descent: RK4 solver status = {rk4.status} -- terminating.')
            break

    if iter >= maxiter:
        print(f'accelerated_gradient_descent: maximum number of iterations reached -- terminating.')

    return rk4.y[:X.shape[0]]
