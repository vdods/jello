import numpy as np
import num_dual as nd

from .j1_otimes_j1 import J1_otimes_J1, D_J1_otimes_J1
from numpy import array, ndarray
from typing import Any, Optional, Tuple

def region_index_and_corners_for_point(vertex_count_v: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """
    Computes the region index and corners for the given point X.  This is used to perform interpolation
    of J1 \\otimes J1 using the minimal data required.

    Returns a tuple (region_index_v, region_corner_v), where region_index_v is a 2-element array of integers
    such that if j1_otimes_j1_data has shape (N, vertex_count_v[0], vertex_count_v[1], 2, 2) (defining
    an N-vector-valued function over the domain)), then the data needed to interpolate the function at X
    is given by j1_otimes_j1_data[:, region_index_v[0]:region_index_v[0]+2, region_index_v[1]:region_index_v[1]+2, :, :],
    and region_corner_v is the corners of the region that X is in.
    """
    assert vertex_count_v.shape == (2,)
    assert vertex_count_v.dtype == int
    assert np.all(vertex_count_v >= 2)
    assert corner_v.shape == (2, 2)
    assert np.all(corner_v[0,:] < corner_v[1,:])
    assert X.shape == (2,)

    region_count_v = vertex_count_v - 1
    assert np.all(region_count_v > 0)
    region_size = (corner_v[1,:] - corner_v[0,:]) / region_count_v
    # Figure out which region X is in.
    X_real = ndarray((2,), dtype=np.float64)
    if isinstance(X[0], nd.Dual64) or isinstance(X[0], nd.HyperDual64):
        X_real[0] = X[0].value
    else:
        X_real[0] = X[0]
    if isinstance(X[1], nd.Dual64) or isinstance(X[1], nd.HyperDual64):
        X_real[1] = X[1].value
    else:
        X_real[1] = X[1]
    assert np.all(corner_v[0,:] <= X_real)
    assert np.all(X_real <= corner_v[1,:])
    region_index_v = np.floor((X_real - corner_v[0,:]) / region_size).astype(int)
    assert np.all(region_index_v >= 0)
    for i in range(2):
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
    j1_otimes_j1_data: ndarray,
    corner_v: ndarray,
    X: ndarray,
    *,
    compute_phi: bool = True,
    compute_Dphi: bool = True
) -> Tuple[Optional[ndarray], Optional[ndarray]]:
    """
    phi(X) is a function defined over a 2D, rectangular domain that is the linear combination of finite elements
    on that domain.  corner_v defines the lower and upper corners of the domain.  The region is subdivided into
    a grid of rectangular cells (the subdivision given by j1_otimes_j1_data.shape[1] and j1_otimes_j1_data.shape[2],
    which define the number of vertices in the x0 and x1 directions, respectively), and at each vertex, there is
    data specifying the J1 \\otimes J1 data for that vertex.

    In particular, j1_otimes_j1_data[k,i,j,:,:] gives the kth component of the J1 \\otimes J1 data for the vertex
    at the corner (i,j), which is a 2x2 matrix, which in particular specifies:

        [[phi(V),      phi_{x0}(V)],
         [phi_{x1}(V), phi_{x0x1}(V)]]
        
    where X = [x0, x1], and V is the vertex at the corner (i,j).

    This function (phi_and_Dphi) computes both phi(X) and Dphi(X) at the same time, which is more efficient than
    computing them separately.  The compute_phi and compute_Dphi flags allow you to compute only one of the two,
    which is useful if you only need one of them.  There are convenience functions `phi` and `Dphi` that call
    this function with the appropriate flags, if only one of them is needed.
    """

    assert len(j1_otimes_j1_data.shape) == 5
    assert j1_otimes_j1_data.shape[0] == 2
    assert j1_otimes_j1_data.shape[3] == 2
    assert j1_otimes_j1_data.shape[4] == 2
    assert corner_v.shape == (2, 2)
    assert np.all(corner_v[0,:] < corner_v[1,:])
    assert X.shape == (2,)

    vertex_count_v = array([j1_otimes_j1_data.shape[1], j1_otimes_j1_data.shape[2]])
    region_index_v, region_corner_v = region_index_and_corners_for_point(vertex_count_v, corner_v, X)
    region_data = j1_otimes_j1_data[:, region_index_v[0]:region_index_v[0]+2, region_index_v[1]:region_index_v[1]+2, :, :]
    assert region_data.shape == (2, 2, 2, 2, 2), f'region_data.shape = {region_data.shape}'

    if compute_phi:
        j1_otimes_j1 = J1_otimes_J1(region_corner_v, X)
        assert j1_otimes_j1.shape == (2, 2, 2, 2)
        phi_X = region_data.reshape(2, -1).dot(j1_otimes_j1.flatten())
    else:
        phi_X = None
    if compute_Dphi:
        d_j1_otimes_j1 = D_J1_otimes_J1(region_corner_v, X)
        assert d_j1_otimes_j1.shape == (2, 2, 2, 2, 2)
        # Bit of a hacky way to get the correct dtype, but hey.
        dtype = type(region_data[0,0,0,0,0]*d_j1_otimes_j1[0,0,0,0,0])
        Dphi_X = ndarray((2, 2), dtype=dtype)
        # TODO: Make this into a single einsum if possible.
        for i in range(2):
            for j in range(2):
                Dphi_X[i,j] = region_data[i,:,:,:,:].flatten().dot(d_j1_otimes_j1[j,:,:,:,:].flatten())
    else:
        Dphi_X = None

    return phi_X, Dphi_X

def phi(j1_otimes_j1_data: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """Convenience function that calls phi_and_Dphi with compute_phi=True and compute_Dphi=False."""
    return phi_and_Dphi(j1_otimes_j1_data, corner_v, X, compute_phi=True, compute_Dphi=False)[0]

def Dphi(j1_otimes_j1_data: ndarray, corner_v: ndarray, X: ndarray) -> ndarray:
    """Convenience function that calls phi_and_Dphi with compute_phi=False and compute_Dphi=True."""
    return phi_and_Dphi(j1_otimes_j1_data, corner_v, X, compute_phi=False, compute_Dphi=True)[1]

def test():
    corner_v = array([[0.0, 0.0], [1.0, 1.0]])
    for region_count_v in [array([2, 2]), array([3, 3]), array([3, 4]), array([4, 4])]:
        # Generate some random j1_otimes_j1_data.
        j1_otimes_j1_data = np.random.randn(2, region_count_v[0]+1, region_count_v[1]+1, 2, 2)
        for x0_real in np.linspace(corner_v[0,0], corner_v[1,0], 23, endpoint=True):
            for x1_real in np.linspace(corner_v[0,1], corner_v[1,1], 23, endpoint=True):
                X = array([x0_real, x1_real])

                phi_X, Dphi_X = phi_and_Dphi(j1_otimes_j1_data, corner_v, X)
                # print(f'phi(X) = {phi_X}')
                # print(f'Dphi(X) = {Dphi_X}')

                expected_Dphi_X = ndarray((2, 2), dtype=np.float64)
                for i in range(2):
                    expected_Dphi_X[i,0] = nd.first_derivative(lambda x0: phi(j1_otimes_j1_data, corner_v, array([x0, x1_real]))[i], x0_real)[1]
                    expected_Dphi_X[i,1] = nd.first_derivative(lambda x1: phi(j1_otimes_j1_data, corner_v, array([x0_real, x1]))[i], x1_real)[1]
                # print(f'expected_Dphi_X = {expected_Dphi_X}')

                assert np.allclose(Dphi_X, expected_Dphi_X)

    print('jello.phi2d.test passed')

if __name__ == '__main__':
    test()
