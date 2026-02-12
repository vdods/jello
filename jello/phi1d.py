import numpy as np
import num_dual as nd

from .j1 import J1, DJ1
from numpy import array, ndarray
from typing import Any, Optional, Tuple

def region_index_and_corners_for_point(vertex_count_v: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """
    Computes the region index and corners for the given point X.  This is used to perform interpolation
    of J1 using the minimal data required.

    Returns a tuple (region_index_v, region_corner_v), where region_index_v is a 1-element array of integers
    such that if j1_data has shape (vertex_count_v[0], 2) (defining an scalar function over the domain)),
    then the data needed to interpolate the function at X is given by j1_data[region_index_v[0]:region_index_v[0]+2, :],
    and region_corner_v is the corners of the region that X is in.
    """
    assert vertex_count_v.shape == (1,)
    assert vertex_count_v.dtype == int
    assert np.all(vertex_count_v >= 2)
    assert corner_v.shape == (2, 1)
    assert np.all(corner_v[0,:] < corner_v[1,:])
    assert X.shape == (1,)

    region_count_v = vertex_count_v - 1
    assert np.all(region_count_v > 0)
    region_size = (corner_v[1,:] - corner_v[0,:]) / region_count_v
    # Figure out which region X is in.
    X_real = ndarray((1,), dtype=np.float64)
    if isinstance(X[0], nd.Dual64) or isinstance(X[0], nd.HyperDual64):
        X_real[0] = X[0].value
    else:
        X_real[0] = X[0]
    assert np.all(corner_v[0,:] <= X_real)
    assert np.all(X_real <= corner_v[1,:])
    region_index_v = np.floor((X_real - corner_v[0,:]) / region_size).astype(int)
    assert np.all(region_index_v >= 0)
    for i in range(1):
        if region_index_v[i] == region_count_v[i]:
            region_index_v[i] -= 1
    assert np.all(region_index_v < region_count_v)
    assert np.all(region_index_v >= 0)

    region_corner_0 = corner_v[0,:] + region_index_v * region_size
    region_corner_1 = region_corner_0 + region_size
    region_corner_v = array([region_corner_0, region_corner_1])
    assert np.all(region_corner_v[0,:] <= X_real)
    # This check does require the epsilon tolerance.
    assert np.all(X_real <= region_corner_v[1,:] + array([1.0e-14, 1.0e-14]))#, f'vertex_count_v = {vertex_count_v}, corner_v = {corner_v}, region_index_v = {region_index_v}, X_real = {X_real}, region_corner_v = {region_corner_v}, region_corner_v[1,:]-X_real = {region_corner_v[1,:]-X_real}'

    return region_index_v, region_corner_v

def phi_and_Dphi(
    j1_data: ndarray,
    corner_v: ndarray,
    X: ndarray,
    *,
    compute_phi: bool = True,
    compute_Dphi: bool = True
) -> Tuple[Optional[ndarray], Optional[ndarray]]:
    """
    phi(X) is a function defined over a 1D, interval domain that is the linear combination of finite elements
    on that domain.  corner_v defines the lower and upper corners of the domain.  The region is subdivided into
    a sequence of intervals (the subdivision given by j1_data.shape[1], which defines the number of vertices
    along the interval), and at each vertex, there is data specifying the J1 data for that vertex.

    In particular, j1_data[k,i,:] gives the kth component of the J1 data for vertex i, which is a vector

        [phi(V), phi_{x0}(V)],

    where x0 is the coordinate along the interval, and V is the vertex at index i.

    This function (phi_and_Dphi) computes both phi(X) and Dphi(X) at the same time, which is more efficient than
    computing them separately.  The compute_phi and compute_Dphi flags allow you to compute only one of the two,
    which is useful if you only need one of them.  There are convenience functions `phi` and `Dphi` that call
    this function with the appropriate flags, if only one of them is needed.
    """

    assert len(j1_data.shape) >= 2, f'j1_data.shape = {j1_data.shape}'
    assert j1_data.shape[-1] == 2, f'j1_data.shape = {j1_data.shape}'
    assert corner_v.shape == (2, 1), f'corner_v.shape = {corner_v.shape}'
    assert np.all(corner_v[0,:] < corner_v[1,:])
    assert X.shape == (1,)

    output_shape = j1_data.shape[0:-2]

    vertex_count_v = array([j1_data.shape[-2]])
    region_index_v, region_corner_v = region_index_and_corners_for_point(vertex_count_v, corner_v, X)
    region_data = j1_data[..., region_index_v[0]:region_index_v[0]+2, :]
    assert len(region_data.shape) >= 2
    assert region_data.shape[-2] == 2
    assert region_data.shape[-1] == 2

    if compute_phi:
        j1 = J1(X[0], region_corner_v[:,0])
        assert j1.shape == (2, 2), f'j1.shape = {j1.shape}'
        phi_X = region_data.reshape(-1, 4).dot(j1.flatten()).reshape(output_shape)
        # Turn it into a scalar if necessary.
        if phi_X.shape == ():
            phi_X = phi_X[()]
    else:
        phi_X = None
    if compute_Dphi:
        dj1 = DJ1(X[0], region_corner_v[:,0])
        assert dj1.shape == (2, 2), f'dj1.shape = {dj1.shape}'
        Dphi_X = region_data.reshape(-1, 4).dot(dj1.flatten()).reshape(output_shape)
        # Turn it into a scalar if necessary.
        if Dphi_X.shape == ():
            Dphi_X = Dphi_X[()]
    else:
        Dphi_X = None

    return phi_X, Dphi_X

def phi(j1_data: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """Convenience function that calls phi_and_Dphi with compute_phi=True and compute_Dphi=False."""
    return phi_and_Dphi(j1_data, corner_v, X, compute_phi=True, compute_Dphi=False)[0]

def Dphi(j1_data: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """Convenience function that calls phi_and_Dphi with compute_phi=False and compute_Dphi=True."""
    return phi_and_Dphi(j1_data, corner_v, X, compute_phi=False, compute_Dphi=True)[1]

def test():
    for corner_v in [array([[0.0], [1.0]]), array([[0.5], [2.0]])]:
        for region_count_v in [array([1]), array([2]), array([3]), array([4])]:
            # Generate some random j1_data.
            j1_data = np.random.randn(region_count_v[0]+1, 2)
            for x0_real in np.linspace(corner_v[0,0], corner_v[1,0], 23, endpoint=True):
                X = array([x0_real])
                phi_X, Dphi_X = phi_and_Dphi(j1_data, corner_v, X)
                expected_Dphi_X = nd.first_derivative(lambda x0: phi(j1_data, corner_v, array([x0])), x0_real)[1]
                assert np.allclose(Dphi_X, expected_Dphi_X)

    print('jello.phi1d.test passed')

if __name__ == '__main__':
    test()
