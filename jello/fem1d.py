import numpy as np
import num_dual as nd

from .j1 import J1
from .phi1d import phi, region_index_and_corners_for_point
from numpy import array, ndarray
from typing import Any, Callable

# TODO: Factor this with FEFunctionSpace1D.
class FEFunctionSpace1D:
    """
    Represents the function space of J1 functions defined over a mesh made of regular intervals.
    """

    def __init__(self, domain_corner_v: ndarray, vertex_count_v: ndarray):
        assert domain_corner_v.shape == (2, 1)
        assert vertex_count_v.shape == (1,)
        assert vertex_count_v.dtype == int
        assert np.all(vertex_count_v >= 2)

        domain_size = domain_corner_v[1,:] - domain_corner_v[0,:]
        assert np.all(domain_size > 0.0)

        self.domain_corner_v = domain_corner_v
        self.vertex_count_v = vertex_count_v
        self.scalar_function_shape = tuple(array([vertex_count_v[0], 2]))
        self.scalar_function_dim = np.prod(self.scalar_function_shape)
   
    def L2_inner_product_eval(self, f1: ndarray, f2: ndarray, measure: 'DiscreteMeasure') -> Any:
        """
        Evaluate the L2 inner product for the given scalar functions f1 and f2.
        """
        assert f1.shape == (self.scalar_function_dim,)
        assert f2.shape == (self.scalar_function_dim,)
        retval = 0.0
        f1_reshaped = f1.reshape(self.scalar_function_shape)
        f2_reshaped = f2.reshape(self.scalar_function_shape)
        for rho_X, X in zip(measure.weight_v, measure.point_v):
            f1_X = phi(f1_reshaped, self.domain_corner_v, X)
            f2_X = phi(f2_reshaped, self.domain_corner_v, X)
            retval += rho_X * f1_X * f2_X
        return retval

    def L2_inner_product(self, measure: 'DiscreteMeasure', dtype: Any) -> ndarray:
        """
        Return the matrix for the L2 inner product.
        """
        retval = np.zeros(self.scalar_function_shape + self.scalar_function_shape, dtype=dtype)
        for rho_X, X in zip(measure.weight_v, measure.point_v):
            region_index_v, region_corner_v = region_index_and_corners_for_point(self.vertex_count_v, self.domain_corner_v, X)
            j1_X = J1(X[0], region_corner_v[:,0])
            assert j1_X.shape == (2, 2)
            retval[region_index_v[0]:region_index_v[0]+2, :, region_index_v[0]:region_index_v[0]+2, :] += np.einsum('ij,kl->ijkl', rho_X * j1_X, j1_X)
        return retval.reshape(self.scalar_function_dim, self.scalar_function_dim)
    
    def subdivided(self) -> 'FEFunctionSpace1D':
        """
        Return the subdivided function space, which is a grid with twice the resolution.
        """
        return FEFunctionSpace1D(
            domain_corner_v=self.domain_corner_v,
            vertex_count_v=(self.vertex_count_v - array([1])) * 2 + array([1]),
        )
    
    def subdivided_linear(self) -> 'FEFunctionSpace1D':
        """
        Increase vertex count along each axis by 1, instead of splitting each element into 2.
        """
        return FEFunctionSpace1D(self.domain_corner_v, (self.vertex_count_v + array([1])))
    
    @staticmethod
    def test_L2_inner_product():
        from .measure import DiscreteMeasure

        F = FEFunctionSpace1D(
            domain_corner_v=array([[0.5], [2.0]]),
            vertex_count_v=array([3]),
        )
        measure = DiscreteMeasure.trapezoid_rule_on_interval(F.domain_corner_v, F.vertex_count_v * 8)

        L2_inner_product = F.L2_inner_product(measure, dtype=np.float64)
        assert np.allclose(L2_inner_product, L2_inner_product.T)

        for i in range(100):
            f1 = np.random.randn(F.scalar_function_dim)
            f2 = np.random.randn(F.scalar_function_dim)
            L2_inner_product_f1_f2 = F.L2_inner_product_eval(f1, f2, measure)
            expected_L2_inner_product_f1_f2 = measure.integrate(lambda X: phi(f1.reshape(F.scalar_function_shape), F.domain_corner_v, X) * phi(f2.reshape(F.scalar_function_shape), F.domain_corner_v, X))
            assert np.allclose(L2_inner_product_f1_f2, expected_L2_inner_product_f1_f2), f'L2_inner_product_f1_f2 = {L2_inner_product_f1_f2}, expected_L2_inner_product_f1_f2 = {expected_L2_inner_product_f1_f2}'

            computed_L2_inner_product_f1_f2 = f1 @ L2_inner_product @ f2
            assert np.allclose(computed_L2_inner_product_f1_f2, expected_L2_inner_product_f1_f2), f'computed_L2_inner_product_f1_f2 = {computed_L2_inner_product_f1_f2}, expected_L2_inner_product_f1_f2 = {expected_L2_inner_product_f1_f2}'

        print('jello.fem1d.FEFunctionSpace1D.test_L2_inner_product passed')

def test():
    FEFunctionSpace1D.test_L2_inner_product()

if __name__ == '__main__':
    test()
