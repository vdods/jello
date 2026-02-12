import numpy as np
# import num_dual as nd
from numpy import array, ndarray
from typing import Any, Callable, Optional

class DiscreteMeasure:
    """
    Represents a discrete measure on a domain, meant to be a finite-dimensional approximation of a continuous measure.
    The domain is not assumed to have any particular shape, but rather the integration points and weights implicitly
    define the domain.  Integration of a function `f` by this measure is defined as
    `sum(weight[i] * f(point[i]) for i in range(integration_point_count))`.
    """

    def __init__(self, *, weight_t: ndarray, point_t: ndarray):
        """
        Construct a discrete measure from the given integration points and weights.
        Args:
            integration_point_weight_v: An array of shape `S` containing the weights for each integration point.
            integration_point_v: An array of shape `S + (N,)` containing the integration points, where `N` is the dimension of the domain.
        """
        assert len(weight_t.shape) >= 1, f'weight_t.shape = {weight_t.shape}'
        assert len(point_t.shape) == len(weight_t.shape) + 1, f'point_t.shape = {point_t.shape}, weight_t.shape = {weight_t.shape}'
        assert point_t.shape[0:-1] == weight_t.shape, f'point_t.shape = {point_t.shape}, weight_t.shape = {weight_t.shape}'
        self.weight_t = weight_t
        self.point_t = point_t

    def __repr__(self) -> str:
        return f'DiscreteMeasure(self.weight_t.shape: {self.weight_t.shape}, self.point_t.shape: {self.point_t.shape})'

    # TODO: Factor this with the other trapezoid rule, and make it dimension-agnostic.
    @staticmethod
    def trapezoid_rule_on_interval(domain_boundary_v: ndarray, integration_axis_point_count_v: ndarray) -> 'DiscreteMeasure':
        """
        Create a discrete measure on an interval, defined as the trapezoidal rule.
        Args:
            domain_boundary_v: The boundary points of the interval, as an array of shape `(2,1)`, where the first index is 0 or 1, indicating the lower or upper boundary of the interval.
            integration_axis_point_count_v: The number of integration points along each axis.  Each value is recommended to be `2^n + 1` for some nonnegative integer `n`, or failing that, is recommended to be an odd number.
        Returns:
            A DiscreteMeasure object.
        """
        assert domain_boundary_v.shape == (2, 1)
        assert integration_axis_point_count_v.shape == (1,)
        assert integration_axis_point_count_v.dtype == int
        assert np.all(integration_axis_point_count_v >= 2)

        domain_size = domain_boundary_v[1,:] - domain_boundary_v[0,:]
        assert np.all(domain_size > 0.0)
        domain_length = np.prod(domain_size)

        # These weights give the trapezoid rule along each axis.
        weight_t = np.ndarray((integration_axis_point_count_v[0],), dtype=np.float64)
        weight_t[:] = domain_size[0]
        weight_t[0] /= 2
        weight_t[-1] /= 2
        weight_t /= (integration_axis_point_count_v[0] - 1)

        # Ensure that integration of the constant function 1 over the domain gives the area of the domain.
        assert np.allclose(np.sum(weight_t), domain_length), f'np.sum(weight_t) = {np.sum(weight_t)}, domain_length = {domain_length}'
        
        point_t = np.linspace(domain_boundary_v[0,0], domain_boundary_v[1,0], integration_axis_point_count_v[0], endpoint=True).reshape(-1, 1)

        return DiscreteMeasure(weight_t=weight_t, point_t=point_t)

    @staticmethod
    def trapezoid_rule_on_rectangle(domain_boundary_v: ndarray, integration_axis_point_count_v: ndarray) -> 'DiscreteMeasure':
        """
        Create a discrete measure on a rectangle, defined as the product of a trapezoidal rule along each axis.
        Args:
            domain_boundary_v: The boundary points of the domain, as an array of shape `(2, N)`, where `N` is the dimension of the domain, and the first index is 0 or 1, indicating the lower or upper boundary of the axis.
            integration_axis_point_count_v: The number of integration points along each axis.  Each value is recommended to be `2^n + 1` for some nonnegative integer `n`, or failing that, is recommended to be an odd number.
        Returns:
            A DiscreteMeasure object.
        """
        assert domain_boundary_v.shape == (2, 2)
        assert integration_axis_point_count_v.shape == (2,)
        assert integration_axis_point_count_v.dtype == int
        assert np.all(integration_axis_point_count_v >= 2)

        domain_size = domain_boundary_v[1,:] - domain_boundary_v[0,:]
        assert np.all(domain_size > 0.0)
        domain_area = np.prod(domain_size)

        # These weights give the trapezoid rule along each axis.
        weight_0 = np.ndarray((integration_axis_point_count_v[0],), dtype=np.float64)
        weight_0[:] = domain_size[0]
        weight_0[0] /= 2
        weight_0[-1] /= 2
        weight_0 /= (integration_axis_point_count_v[0] - 1)

        weight_1 = np.ndarray((integration_axis_point_count_v[1],), dtype=np.float64)
        weight_1[:] = domain_size[1]
        weight_1[0] /= 2
        weight_1[-1] /= 2
        weight_1 /= (integration_axis_point_count_v[1] - 1)

        weight_t = np.outer(weight_0, weight_1)
        # Ensure that integration of the constant function 1 over the domain gives the area of the domain.
        assert np.allclose(np.sum(weight_t), domain_area), f'np.sum(weight_t) = {np.sum(weight_t)}, domain_area = {domain_area}'
        
        point_0 = np.linspace(domain_boundary_v[0,0], domain_boundary_v[1,0], integration_axis_point_count_v[0], endpoint=True)
        point_1 = np.linspace(domain_boundary_v[0,1], domain_boundary_v[1,1], integration_axis_point_count_v[1], endpoint=True)
        point_0, point_1 = np.meshgrid(point_0, point_1, indexing='ij')
        point_t = np.stack([point_0, point_1], axis=2)

        return DiscreteMeasure(weight_t=weight_t, point_t=point_t)

    @staticmethod
    def gauss_legendre_5_on_interval(domain_boundary_v: ndarray, mesh_vertex_count_v: ndarray) -> 'DiscreteMeasure':
        assert domain_boundary_v.shape == (2, 1)
        assert mesh_vertex_count_v.shape == (1,)
        assert mesh_vertex_count_v.dtype == int
        assert np.all(mesh_vertex_count_v >= 2)

        mesh_interval_count_v = mesh_vertex_count_v - 1
        assert np.all(mesh_interval_count_v > 0)

        domain_size = domain_boundary_v[1,:] - domain_boundary_v[0,:]
        assert np.all(domain_size > 0.0)
        domain_length = np.prod(domain_size)

        interval_size = domain_size / mesh_interval_count_v
        assert np.all(interval_size > 0.0)

        # This is the 5-point G-L rule for for the interval [-1, 1].

        a_minus = 5.0 - 2.0*np.sqrt(10.0/7.0)
        a_plus = 5.0 + 2.0*np.sqrt(10.0/7.0)
        b_minus = np.sqrt(a_minus) / 3.0
        b_plus = np.sqrt(a_plus) / 3.0
        univariate_GL_point_v = array([-b_plus, -b_minus, 0.0, b_minus, b_plus])
        # Shift to [0, 1]
        univariate_GL_point_v *= 0.5
        univariate_GL_point_v += 0.5

        c_minus = (322.0 - 13.0*np.sqrt(70)) / 900.0
        c_plus = (322.0 + 13.0*np.sqrt(70)) / 900.0
        univariate_GL_weight_v = array([c_minus, c_plus, 128.0 / 225.0, c_plus, c_minus])
        # Shift to [0, 1]
        univariate_GL_weight_v *= 0.5
        assert np.allclose(np.sum(univariate_GL_weight_v), 1.0), f'np.sum(univariate_GL_weight_v) = {np.sum(univariate_GL_weight_v)}'

        point_t = np.ndarray((5*mesh_interval_count_v[0],1), dtype=np.float64)
        weight_t = np.ndarray((5*mesh_interval_count_v[0],), dtype=np.float64)
        for i in range(mesh_interval_count_v[0]):
            interval_start = domain_boundary_v[0,0] + i * interval_size[0]
            point_t[i*5:(i+1)*5,0] = univariate_GL_point_v * interval_size[0] + interval_start
            weight_t[i*5:(i+1)*5] = univariate_GL_weight_v * interval_size[0]

        # Ensure that integration of the constant function 1 over the domain gives the length of the domain.
        assert np.allclose(np.sum(weight_t), domain_length), f'np.sum(weight_t) = {np.sum(weight_t)}, domain_length = {domain_length}'
 
        return DiscreteMeasure(weight_t=weight_t, point_t=point_t)

    @staticmethod
    def gauss_legendre_5x5_on_rectangle(domain_boundary_v: ndarray, mesh_vertex_count_v: ndarray) -> 'DiscreteMeasure':
        assert domain_boundary_v.shape == (2, 2)
        assert mesh_vertex_count_v.shape == (2,)
        assert mesh_vertex_count_v.dtype == int
        assert np.all(mesh_vertex_count_v >= 2)

        mesh_rectangle_count_v = mesh_vertex_count_v - 1
        assert np.all(mesh_rectangle_count_v > 0)

        domain_size = domain_boundary_v[1,:] - domain_boundary_v[0,:]
        assert np.all(domain_size > 0.0)
        domain_area = np.prod(domain_size)

        rectangle_size = domain_size / mesh_rectangle_count_v
        assert np.all(rectangle_size > 0.0)

        # This is the 5-point G-L rule for for the interval [-1, 1].

        a_minus = 5.0 - 2.0*np.sqrt(10.0/7.0)
        a_plus = 5.0 + 2.0*np.sqrt(10.0/7.0)
        b_minus = np.sqrt(a_minus) / 3.0
        b_plus = np.sqrt(a_plus) / 3.0
        univariate_GL_point_v = array([-b_plus, -b_minus, 0.0, b_minus, b_plus])
        # Shift to [0, 1]
        univariate_GL_point_v *= 0.5
        univariate_GL_point_v += 0.5

        c_minus = (322.0 - 13.0*np.sqrt(70)) / 900.0
        c_plus = (322.0 + 13.0*np.sqrt(70)) / 900.0
        univariate_GL_weight_v = array([c_minus, c_plus, 128.0 / 225.0, c_plus, c_minus])
        # Shift to [0, 1]
        univariate_GL_weight_v *= 0.5
        assert np.allclose(np.sum(univariate_GL_weight_v), 1.0), f'np.sum(univariate_GL_weight_v) = {np.sum(univariate_GL_weight_v)}'

        point_0 = np.ndarray((5*mesh_rectangle_count_v[0],), dtype=np.float64)
        weight_0 = np.ndarray((5*mesh_rectangle_count_v[0],), dtype=np.float64)
        for i in range(mesh_rectangle_count_v[0]):
            interval_start = domain_boundary_v[0,0] + i * rectangle_size[0]
            point_0[i*5:(i+1)*5] = univariate_GL_point_v * rectangle_size[0] + interval_start
            weight_0[i*5:(i+1)*5] = univariate_GL_weight_v * rectangle_size[0]

        assert np.allclose(np.sum(weight_0), domain_size[0]), f'np.sum(weight_0) = {np.sum(weight_0)}, domain_size[0] = {domain_size[0]}'

        point_1 = np.ndarray((5*mesh_rectangle_count_v[1],), dtype=np.float64)
        weight_1 = np.ndarray((5*mesh_rectangle_count_v[1],), dtype=np.float64)
        for i in range(mesh_rectangle_count_v[1]):
            interval_start = domain_boundary_v[0,1] + i * rectangle_size[1]
            point_1[i*5:(i+1)*5] = univariate_GL_point_v * rectangle_size[1] + interval_start
            weight_1[i*5:(i+1)*5] = univariate_GL_weight_v * rectangle_size[1]

        assert np.allclose(np.sum(weight_1), domain_size[1]), f'np.sum(weight_1) = {np.sum(weight_1)}, domain_size[1] = {domain_size[1]}'

        weight_t = np.outer(weight_0, weight_1)
        # Ensure that integration of the constant function 1 over the domain gives the area of the domain.
        assert np.allclose(np.sum(weight_t), domain_area), f'np.sum(weight_t) = {np.sum(weight_t)}, domain_area = {domain_area}'
        
        point_0, point_1 = np.meshgrid(point_0, point_1, indexing='ij')
        point_t = np.stack([point_0, point_1], axis=2)

        return DiscreteMeasure(weight_t=weight_t, point_t=point_t)

    @property
    def integration_point_count(self) -> int:
        return np.prod(self.weight_t.shape, dtype=int)
    
    @property
    def weight_v(self) -> ndarray:
        return self.weight_t.flatten()
    
    @property
    def point_v(self) -> ndarray:
        return self.point_t.reshape(-1, self.point_t.shape[-1])

    def integrate(self, f: Callable[[ndarray], Any], *, temp: Optional[ndarray] = None) -> Any:
        """
        Integrate the given function over the domain using the discrete measure.
        Args:
            f: The function to integrate.
            temp: An optional pre-allocated array to avoid memory allocation.
        Returns:
            The integral of the function over the domain.
        """
        if temp is None:
            # TODO: Figure out how to vectorize, and ideally parallelize this call.
            temp = np.apply_along_axis(f, 1, self.point_v).flatten()
            assert temp.shape == (self.integration_point_count,), f'expected temp.shape = {temp.shape} to be equal to (self.integration_point_count,) = {(self.integration_point_count,)}'
        else:
            assert temp.shape == (self.integration_point_count,), f'expected temp.shape = {temp.shape} to be equal to (self.integration_point_count,) = {(self.integration_point_count,)}'
            # TODO: Figure out how to vectorize, and ideally parallelize this call.
            temp[:] = np.apply_along_axis(f, 1, self.point_v)

        temp *= self.weight_v
        return np.sum(temp)
    
    @staticmethod
    def test_discrete_measure_on_interval(measure: 'DiscreteMeasure', *, domain_boundary_v: ndarray, expected_exact_integration_polynomial_degree: int):
        assert expected_exact_integration_polynomial_degree >= 0
        assert expected_exact_integration_polynomial_degree <= 4, 'haven\'t yet implemented testing of exact integration of polynomials of degree 5 or higher'
        assert domain_boundary_v.shape == (2, 1)
        assert measure.weight_t.shape == (measure.integration_point_count,)
        assert measure.point_t.shape == (measure.integration_point_count, 1)

        domain_length = domain_boundary_v[1,0] - domain_boundary_v[0,0]

        # Ensure that the constant function 1 integrates to the length of the domain.
        assert np.allclose(measure.integrate(lambda x: 1.0), domain_length)

        # Ensure that the expected degrees of polynomials integrate exactly.
        for c_0 in [1.0, 2.0, 3.0]:
            # Degree 0 polynomial.
            p0 = lambda x: c_0
            p0_antiderivative = lambda x: c_0 * x
            assert np.allclose(measure.integrate(p0), p0_antiderivative(domain_boundary_v[1,0]) - p0_antiderivative(domain_boundary_v[0,0]))
            if expected_exact_integration_polynomial_degree < 1:
                continue

            for c_1 in [1.0, 2.0, 3.0]:
                # Degree 1 polynomial.
                p1 = lambda x: c_0 + c_1*x
                p1_antiderivative = lambda x: c_0*x + c_1*x**2/2
                assert np.allclose(measure.integrate(p1), p1_antiderivative(domain_boundary_v[1,0]) - p1_antiderivative(domain_boundary_v[0,0]))
                if expected_exact_integration_polynomial_degree < 2:
                    continue

                for c_2 in [1.0, 2.0, 3.0]:
                    # Degree 2 polynomial.
                    p2 = lambda x: c_0 + c_1*x + c_2*x**2
                    p2_antiderivative = lambda x: c_0*x + c_1*x**2/2 + c_2*x**3/3
                    assert np.allclose(measure.integrate(p2), p2_antiderivative(domain_boundary_v[1,0]) - p2_antiderivative(domain_boundary_v[0,0]))
                    if expected_exact_integration_polynomial_degree < 3:
                        continue

                    for c_3 in [1.0, 2.0, 3.0]:
                        # Degree 3 polynomial.
                        p3 = lambda x: c_0 + c_1*x + c_2*x**2 + c_3*x**3
                        p3_antiderivative = lambda x: c_0*x + c_1*x**2/2 + c_2*x**3/3 + c_3*x**4/4
                        assert np.allclose(measure.integrate(p3), p3_antiderivative(domain_boundary_v[1,0]) - p3_antiderivative(domain_boundary_v[0,0]))
                        if expected_exact_integration_polynomial_degree < 4:
                            continue

                        for c_4 in [1.0, 2.0, 3.0]:
                            # Degree 4 polynomial.
                            p4 = lambda x: c_0 + c_1*x + c_2*x**2 + c_3*x**3 + c_4*x**4
                            p4_antiderivative = lambda x: c_0*x + c_1*x**2/2 + c_2*x**3/3 + c_3*x**4/4 + c_4*x**5/5
                            assert np.allclose(measure.integrate(p4), p4_antiderivative(domain_boundary_v[1,0]) - p4_antiderivative(domain_boundary_v[0,0]))

    @staticmethod
    def test_trapezoid_rule_on_interval():
        for domain_boundary_v in [array([[0.0], [1.0]]), array([[1.0], [2.3]])]:
            for integration_axis_point_count in [array([2]), array([3]), array([10])]:
                measure = DiscreteMeasure.trapezoid_rule_on_interval(domain_boundary_v, integration_axis_point_count)
                DiscreteMeasure.test_discrete_measure_on_interval(measure, domain_boundary_v=domain_boundary_v, expected_exact_integration_polynomial_degree=1)

        print('jello.measure.DiscreteMeasure.test_trapezoid_rule_on_interval passed')

    @staticmethod
    def test_gauss_legendre_5_on_interval():
        for domain_boundary_v in [array([[0.0], [1.0]]), array([[1.0], [2.3]])]:
            for mesh_vertex_count in [array([2]), array([3]), array([4])]:
                # Note that the 5-point Gauss-Legendre rule is actually exact for polynomials of degree 2*5-1 = 9,
                # but we only test up to degree 4 here to avoid the overhead of testing higher-degree polynomials.
                measure = DiscreteMeasure.gauss_legendre_5_on_interval(domain_boundary_v, mesh_vertex_count)
                DiscreteMeasure.test_discrete_measure_on_interval(measure, domain_boundary_v=domain_boundary_v, expected_exact_integration_polynomial_degree=4)

        print('jello.measure.DiscreteMeasure.test_gauss_legendre_5_on_interval passed')

def test():
    DiscreteMeasure.test_trapezoid_rule_on_interval()
    DiscreteMeasure.test_gauss_legendre_5_on_interval()

    print('jello.measure.test passed')

