import num_dual as nd
import numpy as np

from numpy import array, ndarray
from typing import Any, Callable, Optional

def get_first_derivative(d1: nd.Dual64) -> float:
    return d1.first_derivative

def get_second_derivative(d2: nd.HyperDual64) -> float:
    return d2.second_derivative

get_first_derivative_vectorized = np.vectorize(get_first_derivative)
get_second_derivative_vectorized = np.vectorize(get_second_derivative)

def integrate_flow_curve(V: Callable[[ndarray], ndarray], X_initial: ndarray, *, step_size: float, maxiter: int, retval_jet_order: Optional[int] = None) -> ndarray:
    """
    Integrates the vector field V starting from X_initial for maxiter steps, with step size step_size,
    producing a discretized flow curve.  It uses the 3rd order Taylor polynomial of the flow curve at
    each step (which is fully derivable from V and its derivatives), and therefore has error O(step_size^4).

    If retval_jet_order is None, then the return value has shape (maxiter,) + X_initial.shape and is
    indexed as [i,...], where i is the step index and ... indexes the components of the flow curve
    (... has the same shape as X_initial).

    If retval_jet_order is 0, 1, 2, or 3, then the return value has shape
    (maxiter, retval_jet_order + 1) + X_initial.shape and is indexed as [i,j,...], where i is the step
    index, j is the jet index (0 for 0th derivative, 1 for 1st derivative, etc), and ... indexes
    the components of the flow curve (... has the same shape as X_initial).
    """

    assert retval_jet_order is None or 0 <= retval_jet_order <= 3, f'retval_jet_order = {retval_jet_order} must be None or 0, 1, 2, or 3'

    if retval_jet_order is None:
        retval_shape = (maxiter,) + X_initial.shape
    else:
        retval_shape = (maxiter, retval_jet_order + 1) + X_initial.shape

    iter = 0
    X = X_initial.copy()
    X_next = np.zeros_like(X_initial)
    V_X = np.zeros_like(X_initial)
    DV_X_dot_V_X = np.zeros_like(X_initial)
    DV_X_dot_DV_X_dot_V_X = np.zeros_like(X_initial)
    D2V_X_double_dot_V_X_tensor_squared = np.zeros_like(X_initial)
    retval = np.ndarray(retval_shape, dtype=X_initial.dtype)
    while iter < maxiter:
        V_X[...] = V(X)
        DV_X_dot_V_X[...] = get_first_derivative_vectorized(V(X + nd.Dual64(0.0, 1.0) * V_X))
        DV_X_dot_DV_X_dot_V_X[...] = get_first_derivative_vectorized(V(X + nd.Dual64(0.0, 1.0) * DV_X_dot_V_X))
        D2V_X_double_dot_V_X_tensor_squared[...] = get_second_derivative_vectorized(V(X + nd.HyperDual64(0.0, 1.0, 1.0, 0.0) * V_X))

        X_next[...] = D2V_X_double_dot_V_X_tensor_squared
        X_next[...] += DV_X_dot_DV_X_dot_V_X
        X_next[...] *= step_size / 3
        X_next[...] += DV_X_dot_V_X
        X_next[...] *= step_size / 2
        X_next[...] += V_X
        X_next[...] *= step_size
        X_next[...] += X

        # X_next[...] = X
        # X_next[...] += step_size * V_X
        # X_next[...] += step_size**2 / 2.0 * DV_X_dot_V_X

        if retval_jet_order is None:
            retval[iter,...] = X
        else:
            if retval_jet_order >= 0:
                retval[iter,0,...] = X
            if retval_jet_order >= 1:
                retval[iter,1,...] = V_X
            if retval_jet_order >= 2:
                retval[iter,2,...] = DV_X_dot_V_X
            if retval_jet_order >= 3:
                retval[iter,3,...] = D2V_X_double_dot_V_X_tensor_squared
                retval[iter,3,...] += DV_X_dot_DV_X_dot_V_X

        iter += 1
        X[...] = X_next

    return retval

def test():
    import time

    # Hamiltonian for Kepler problem with -1/r gravitational potential.
    def H(qp: ndarray) -> ndarray:
        # Position
        q = qp[0,:]
        # Momentum
        p = qp[1,:]
        # Kinetic energy
        K = p.dot(p) / 2
        # Potential energy
        U = -1 / q.dot(q)**0.5
        # Hamiltonian
        return K + U

    # Symplectic gradient of H is X_H := \omega^-1 \cdot dH, where \omega is the symplectic form.
    def X_H(qp: ndarray) -> ndarray:
        # Position
        q = qp[0,:]
        # Momentum
        p = qp[1,:]
        return array([
            # q' = dH_dp
            p,
            # p' = -dH_dq
            -q.dot(q)**-1.5 * q,
        ])
    
    # qp_initial = np.array([[1.0, 0.0], [0.0, 0.5]])
    qp_initial = np.array([np.linspace(1.0, 10.0, 100, endpoint=True), np.linspace(0.0, 8.0, 100, endpoint=True)**0.5])
    H_initial = H(qp_initial)
    print(f'H_initial = {H_initial}')
    retval_jet_order = 3
    for i, (maxiter, step_size) in enumerate([(1000, 0.1), (10000, 0.01), (100000, 0.001)]):
        start = time.time()
        flow_curve_t = integrate_flow_curve(X_H, qp_initial, step_size=step_size, maxiter=maxiter, retval_jet_order=retval_jet_order)
        print(f'maxiter = {maxiter}, step_size = {step_size}, time taken to integrate flow curve: {time.time() - start} seconds')
        # Evaluate H along the flow curve.
        H_flow_curve_t = np.apply_along_axis(lambda qp_flat: H(qp_flat.reshape(2, -1)), 1, flow_curve_t[:,0,...].reshape(maxiter, -1))
        print(f'maxiter = {maxiter}, step_size = {step_size}, max(abs(H_flow_curve_t - H_initial)) = {np.max(np.abs(H_flow_curve_t - H_initial))}')
        # Check the error 

if __name__ == '__main__':
    test()