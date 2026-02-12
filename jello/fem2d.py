import numpy as np
import num_dual as nd

from .j1_otimes_j1 import J1_otimes_J1
from .phi2d import phi, region_index_and_corners_for_point
from numpy import array, ndarray
from typing import Any, Callable

class FEFunctionSpace2D:
    """
    Represents the function space of J1 \\otimes J1 functions defined over a regular, rectangular mesh.
    """

    def __init__(self, domain_corner_v: ndarray, vertex_count_v: ndarray):
        assert domain_corner_v.shape == (2, 2)
        assert vertex_count_v.shape == (2,)
        assert vertex_count_v.dtype == int
        assert np.all(vertex_count_v >= 2)

        domain_size = domain_corner_v[1,:] - domain_corner_v[0,:]
        assert np.all(domain_size > 0.0)

        self.domain_corner_v = domain_corner_v
        self.vertex_count_v = vertex_count_v
        self.scalar_function_shape = array([vertex_count_v[0], vertex_count_v[1], 2, 2])
        self.scalar_function_dim = np.prod(self.scalar_function_shape)
        # The vector index is first, then the scalar function shape.
        self.vector_function_shape = array([2] + self.scalar_function_shape.tolist())
        self.vector_function_dim = np.prod(self.vector_function_shape)
        # A metric is expressed with respect to the tensor frame:
        # 0: dx \otimes dx
        # 1: (dx \otimes dy + dy \otimes dx) / 2
        # 2: dy \otimes dy,
        # noting that the dual coframe is:
        # 0: \partial_x \otimes \partial_x
        # 1: \partial_x \otimes \partial_y + \partial_y \otimes \partial_x
        # 2: \partial_y \otimes \partial_y.
        # The symmetrized metric index is first, then the scalar function shape.
        self.sym_metric_function_shape = array([3] + self.scalar_function_shape.tolist())
        self.sym_metric_function_dim = np.prod(self.sym_metric_function_shape)
   
    def vector_field_metric_eval(self, base_cfg: ndarray, v1: ndarray, v2: ndarray, measure: 'DiscreteMeasure', spatial_manifold: 'SpatialManifold') -> Any:
        """
        Evaluate the metric at the given vector-valued function base_cfg, and the given tangent vectors v1 and v2
        (each of which represent a vector field along the function base_cfg).
        """
        assert base_cfg.shape == (self.vector_function_dim,)
        assert v1.shape == (self.vector_function_dim,)
        assert v2.shape == (self.vector_function_dim,)
        retval = 0.0
        base_cfg_reshaped = base_cfg.reshape(self.vector_function_shape)
        v1_reshaped = v1.reshape(self.vector_function_shape)
        v2_reshaped = v2.reshape(self.vector_function_shape)
        for rho_X, X in zip(measure.weight_v, measure.point_v):
            Y = phi(base_cfg_reshaped, self.domain_corner_v, X)
            h_Y = spatial_manifold.spatial_metric(Y)
            V1 = phi(v1_reshaped, self.domain_corner_v, X)
            V2 = phi(v2_reshaped, self.domain_corner_v, X)
            retval += rho_X * V1.dot(h_Y).dot(V2)
        return retval

    def vector_field_metric(self, base_cfg: ndarray, measure: 'DiscreteMeasure', spatial_manifold: 'SpatialManifold') -> ndarray:
        """
        Let F denote this (scalar) function space.  Let Y denote coordinates in spatial_manifold.
        Let V denote Y \\otimes F, which is the space of vector-valued functions on this function
        space's domain.  Then this function computes the metric on the tangent space of V at the
        given vector-valued function whose coordinates (in V) are given by base_cfg.
        """
        assert base_cfg.shape == (self.vector_function_dim,)
        # Ideally this could take advantage of the symmetry of the metric in order to reduce memory and computation,
        # but we don't have that for now.
        retval = np.zeros(tuple(self.vector_function_shape.tolist()) + tuple(self.vector_function_shape.tolist()), dtype=base_cfg.dtype)
        # TODO: Make this more efficient.
        base_cfg_reshaped = base_cfg.reshape(self.vector_function_shape)
        # X is the integration point, and rho_X is the integration weight.
        for rho_X, X in zip(measure.weight_v, measure.point_v):
            Y = phi(base_cfg_reshaped, self.domain_corner_v, X)
            h_Y = spatial_manifold.spatial_metric(Y)
            region_index_v, region_corner_v = region_index_and_corners_for_point(self.vertex_count_v, self.domain_corner_v, X)
            j1_otimes_j1_X = J1_otimes_J1(region_corner_v, X)

            retval[:, region_index_v[0]:region_index_v[0]+2, region_index_v[1]:region_index_v[1]+2, :, :, :, region_index_v[0]:region_index_v[0]+2, region_index_v[1]:region_index_v[1]+2, :, :] += np.einsum('uv,ijkl,pqrs->uijklvpqrs', rho_X * h_Y, j1_otimes_j1_X, j1_otimes_j1_X)

        return retval.reshape(self.vector_function_dim, self.vector_function_dim)
    
    def vector_field_gradient(self, f: Callable[[ndarray], Any], base_cfg: ndarray, measure: 'DiscreteMeasure', spatial_manifold: 'SpatialManifold') -> ndarray:
        """
        Compute the gradient of the given vector-valued function at the given base configuration.
        """
        assert base_cfg.shape == (self.vector_function_dim,)
        Df = array(nd.gradient(f, base_cfg)[1])
        metric = self.vector_field_metric(base_cfg, measure, spatial_manifold)
        return np.linalg.solve(metric, Df)
    
    def subdivided(self) -> 'FEFunctionSpace2D':
        """
        Return the subdivided function space, which is a grid with twice the resolution.
        """
        return FEFunctionSpace2D(
            domain_corner_v=self.domain_corner_v,
            vertex_count_v=(self.vertex_count_v - array([1, 1])) * 2 + array([1, 1]),
        )
    
    def subdivided_linear(self) -> 'FEFunctionSpace2D':
        # EXPERIMENTAL -- increase vertex count along each axis by 1, instead of splitting each element into 4.
        return FEFunctionSpace2D(self.domain_corner_v, (self.vertex_count_v + array([1, 1])))
    
    @staticmethod
    def test_metric():
        from .measure import DiscreteMeasure
        from .spatial_manifold import SpatialManifold

        F = FEFunctionSpace2D(
            domain_corner_v=array([[0.5, 2.0], [1.0, 3.0]]),
            vertex_count_v=array([3, 2]),
        )
        measure = DiscreteMeasure.trapezoid_rule_on_rectangle(F.domain_corner_v, F.vertex_count_v * 8)

        # Just a made-up, nonlinear metric.
        def spatial_metric(X: ndarray) -> ndarray:
            a = 1.25 + np.sin(X[0]**2)
            b = np.exp(-X.dot(X))
            c = 1.25 + np.cos(X[1]**2)
            return np.array([[a, b], [b, c]])
        
        S = SpatialManifold(
            surface_name='test-surface',
            spatial_embedding_z=lambda X: 0.0,
            spatial_metric=spatial_metric,
        )
        mass_specific_potential = lambda X: 0.0

        # print('FEFunctionSpace2D.test_metric starting -------------------')
        for i in range(10):
            # print(f'i = {i}')
            # Generate some random vector-valued functions to test with.
            base_cfg = np.random.randn(F.vector_function_dim)
            metric = F.vector_field_metric(base_cfg, measure, S)
            for j in range(10):
                # print(f'j = {j}')
                v1 = np.random.randn(F.vector_function_dim)
                v2 = np.random.randn(F.vector_function_dim)
                X = array([np.random.uniform(F.domain_corner_v[0,0], F.domain_corner_v[1,0]), np.random.uniform(F.domain_corner_v[0,1], F.domain_corner_v[1,1])])
                v1_metric_v2_from_eval = F.vector_field_metric_eval(base_cfg, v1, v2, measure, S)
                v1_metric_v2_from_metric = v1.dot(metric).dot(v2)
                assert np.allclose(v1_metric_v2_from_eval, v1_metric_v2_from_metric)

        print('jello.fem2d.FEFunctionSpace2D.test_metric passed')

def test():
    FEFunctionSpace2D.test_metric()

if __name__ == '__main__':
    test()
