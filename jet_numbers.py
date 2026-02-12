# This is an improvement on the "hyper-dual numbers" for automatic differentiation.
# It can compute total 0th, 1st, and 2nd order derivatives of a function (which can be
# univariate or multivariate, and can be scalar-valued or vector-valued), and does so
# without any redundant computation.  Contrast this with the hyper-dual number approach
# in which to compute one component of the (k+1)-th order derivative, all values of the 0th
# through kth derivatives must be computed -- furthermore, the hyper-dual numbers store
# redundant values (e.g. two copies of the 1st order derivative in order to compute a
# 2nd order derivative).

import num_dual as nd
import numpy as np
import sympy as sp
import time
import vorpy as vp

import vorpy.symbolic
import vorpy.tensor

from numpy import array, ndarray
from sympy import Integer, Rational
from typing import Any, Tuple

def subs(expr: Any, *subs_v) -> Any:
    if isinstance(expr, ndarray):
        return array([subs(e, *subs_v) for e in expr])
    else:
        return expr.subs(*subs_v)

simplify = np.vectorize(sp.simplify)

def outer_product(a: Any, b: Any) -> Any:
    """
    Computes the outer product of two values.
    If a and b are scalars, returns a * b, which is a scalar.
    If a and b are vectors, returns np.outer(a, b), which is an ndarray with shape np.shape(a) + np.shape(b).
    """
    shape_a = np.shape(a)
    shape_b = np.shape(b)
    if len(shape_a) == len(shape_b) == 0:
        return a * b
    elif len(shape_a) == len(shape_b) == 1:
        return np.outer(a, b)
    else:
        raise ValueError(f'outer_product is only defined for scalar or vector arguments, but a has shape {shape_a} and b has shape {shape_b}')

def symmetric_outer_product(a: Any, b: Any) -> Any:
    """
    Computes the symmetric outer product of two values.
    If a and b are scalars, returns a * b, which is a scalar.
    If a and b are vectors, returns (np.outer(a, b) + np.outer(b, a)) / 2, which is an ndarray with shape np.shape(a) + np.shape(b).
    """
    shape_a = np.shape(a)
    shape_b = np.shape(b)
    if len(shape_a) == len(shape_b) == 0:
        return a * b
    elif len(shape_a) == len(shape_b) == 1:
        return (np.outer(a, b) + np.outer(b, a)) / 2
    else:
        raise ValueError(f'outer_product is only defined for scalar or vector arguments, but a has shape {shape_a} and b has shape {shape_b}')

# TODO: Rename f, j, h to d0, d1, d2.
# TODO: Could probably just make the scalar form of this, which would form an algebra,
# and then simply use that as the dtype in ndarrays.
class J2:
    """
    Represents the 2-jet of a function X -> \\mathbb{R} at a point, or equivalently the 2nd order
    Taylor polynomial of the function (at the point) whose multiplication operation involves
    truncating polynomial terms of order 3 and higher (this is equivalent to the dual numbers'
    `epsilon` squaring to zero).
    """

    def __init__(self, flat: ndarray, shape_X: tuple):
        assert len(shape_X) <= 1, f'shape_X = {shape_X} must be `()` (indicating a scalar) or `(dim_X,)` (indicating a vector of dimension dim_X)'
        dim_X = np.prod(shape_X, dtype=np.int32)
        assert len(flat) == 1 + dim_X + dim_X**2, f'`flat` array must have length {1 + dim_X + dim_X**2} (which is {1} + {dim_X} + {dim_X}^2) but it has length {len(flat)}'
        self.flat = flat
        # self.dim_X = dim_X
        self.shape_X = shape_X

    @staticmethod
    def from_values(f: Any, j: Any, h: Any) -> 'J2':
        """
        Directly specifies the values of the 0th, 1st, and 2nd order derivatives of the function at the given point.
        f \\in \\mathbb{R}
        j \\in X^*
        h \\in X^* \\otimes X^*
        X could be the real numbers, in which case f, j, h \\in \\mathbb{R}.
        (TODO: symmetrize h.)
        """
        shape_f = np.shape(f)
        shape_j = np.shape(j)
        shape_h = np.shape(h)
        if len(shape_f) == len(shape_j) == len(shape_h):
            # Univariate function case.
            assert shape_f == shape_j == shape_h, 'Univariate function case, expected all arguments to have the same shape'
            shape_X = ()
        else:
            # Multivariate function case.
            assert len(shape_j) == len(shape_f) + 1, f'Jacobian must have one more index than the function (function shape: {shape_f}, Jacobian shape: {shape_j})'
            assert len(shape_h) == len(shape_f) + 2, f'Hessian must have two more indexes than the function (function shape: {shape_f}, Hessian shape: {shape_h})'
            dim_X = shape_j[-1]
            assert shape_h[-2:] == (dim_X, dim_X), f'Hessian was expected to have shape {shape_f + (dim_X, dim_X)} but it had shape {shape_h}'
            assert shape_j[:-1] == shape_f, f'Jacobian was expected to have shape {shape_f + (dim_X,)} but it had shape {shape_j}'
            assert shape_h[:-2] == shape_f, f'Hessian was expected to have shape {shape_f + (dim_X, dim_X)} but it had shape {shape_h}'
            shape_X = (shape_j[-1],)

        j2 = J2.uninitialized(shape_X, type(f))
        j2.f[...] = f
        j2.j[...] = j
        j2.h[...] = h
        return j2
    
    @staticmethod
    def from_scalar_var(x: Any) -> Any:
        """
        A scalar variable essentially represents the 2-jet of the identity function \\mathbb{R} -> \\mathbb{R},
        evaluated at x \\in \\mathbb{R}.  In particular, it is J2.from_values(x, 1, 0).  Note that shape_X
        is necessarily `()` in this case.
        """
        assert len(np.shape(x)) == 0, f'x ({x}) must be a scalar'
        return J2.from_values(x, 1, 0)

    @staticmethod
    def from_const(c: Any, shape_X: tuple, dtype: Any) -> Any:
        """
        A constant essentially represents the 2-jet of a constant function at a point.
        In particular, it is J2.from_values(c, 0, 0).
        """
        assert len(np.shape(c)) == 0, f'c ({c}) must be a scalar'
        retval = J2.zero(shape_X, dtype)
        retval.f[...] = c
        return retval

    # @staticmethod
    # def from_var(x: Any) -> Any:
    #     """
    #     A variable essentially represents the 2-jet of the identity function X -> X, where x \\in X,
    #     at the given point.  In particular, it is J2.from_values(x, I, 0), where I denotes the
    #     identity matrix on X.
    #     """
    #     shape_X = np.shape(x)
    #     if len(shape_X) == 0:
    #         # Scalar case, return J2 directly.
    #         return J2.from_values(x, 1, 0)
    #     elif len(shape_X) == 1:
    #         # Vector case, return an ndarray of J2 objects.
    #         retval = np.ndarray(shape_X, dtype=J2)
    #         for i in range(shape_X[0]):
    #             retval[i] = J2.zero(shape_X, x.dtype)
    #             retval[i].f[:] = x[i]
    #             retval[i].j[i] = 1
    #         return retval
    #     else:
    #         raise ValueError(f'x has shape {shape_X}, which is not a scalar or vector')

    # @staticmethod
    # def from_const(c: Any, *, shape_X: tuple) -> Any:
    #     """
    #     A constant essentially represents the 2-jet of a constant function at the given point.
    #     In particular, it is J2.from_components(c, 0, 0), where the first 0 is an ndarray with
    #     shape np.shape(c) + shape_X, and the second 0 is an ndarray with shape np.shape(c) + shape_X + shape_X.
    #     """
    #     shape_c = np.shape(c)
    #     if len(shape_c) == 0:
    #         # Scalar case, return J2 directly.
    #         return J2.from_values(c, 0, 0)
    #     elif len(shape_c) == 1:
    #         # Vector case, return an ndarray of J2 objects.
    #         retval = np.ndarray(shape_c, dtype=J2)
    #         for i in range(shape_c[0]):
    #             retval[i] = J2.zero(shape_X, c.dtype)
    #             retval[i].f[:] = c[i]
    #         return retval
    #     else:
    #         raise ValueError(f'c has shape {shape_c}, which is not a scalar or vector')

    #     # shape_Y = np.shape(c)
    #     # j2 = J2.uninitialized(shape_X, shape_Y, c.dtype)
    #     # j2.f[...] = c
    #     # j2.j[...] = 0
    #     # j2.h[...] = 0
    #     # return j2
    
    @staticmethod
    def zero(shape_X: tuple, dtype: Any) -> 'J2':
        j2 = J2.uninitialized(shape_X, dtype)
        j2.flat[:] = 0
        return j2

    @staticmethod
    def uninitialized(shape_X: tuple, dtype: Any) -> 'J2':
        assert len(shape_X) <= 1, f'shape_X = {shape_X} must be `()` (indicating a scalar) or `(dim_X,)` (indicating a vector of dimension dim_X)'
        dim_X = np.prod(shape_X, dtype=np.int32)
        flat = np.ndarray((1 + dim_X + dim_X**2,), dtype=dtype)
        return J2(flat, shape_X)

    @property
    def f(self) -> Any:
        """Returns a view into the 0th order component of the 2-jet."""
        # () is the shape of a scalar
        return self.flat[:1].reshape(self.shape_f)
    
    @property
    def j(self) -> Any:
        """Returns a view into the 1st order component of the 2-jet."""
        return self.flat[1:1 + self.dim_X].reshape(self.shape_j)
    
    @property
    def h(self) -> Any:
        """Returns a view into the 2nd order component of the 2-jet."""
        return self.flat[1 + self.dim_X:].reshape(self.shape_h)

    @property
    def f_flattened(self) -> ndarray:
        """Returns a flattened view into the 0th order component of the 2-jet."""
        return self.flat[:1]
    
    @property
    def j_flattened(self) -> ndarray:
        """Returns a flattened view into the 1st order component of the 2-jet."""
        return self.flat[1:1 + self.dim_X]
    
    @property
    def h_flattened(self) -> ndarray:
        """Returns a flattened view into the 2nd order component of the 2-jet."""
        return self.flat[1 + self.dim_X:]

    @property
    def shape_f(self) -> tuple:
        """Returns the shape of the 0th order component of the 2-jet (meaning the value of the 0th order derivative of the function it represents at a particular point)."""
        return ()
    
    @property
    def shape_j(self) -> tuple:
        """Returns the shape of the 1st order component of the 2-jet (meaning the value of the 1st order derivative of the function it represents at a particular point)."""
        return self.shape_X
    
    @property
    def shape_h(self) -> tuple:
        # TODO: Symmetrize the Hessian.
        """Returns the shape of the 2nd order component of the 2-jet (meaning the value of the 2nd order derivative of the function it represents at a particular point)."""
        return self.shape_X + self.shape_X

    @property
    def dim_X(self) -> int:
        """Returns the dimension of the space X, which is the domain of the 2-jet (meaning the domain of the function it represents)."""
        return np.prod(self.shape_X, dtype=np.int32)

    @property
    def is_univariate(self) -> bool:
        """Returns True if the 2-jet is univariate (meaning the function it represents is univariate), False otherwise."""
        return len(self.shape_X) == 0
    
    @property
    def dtype(self) -> Any:
        """Returns the dtype of self.flat."""
        return self.flat.dtype
    
    def copied(self) -> 'J2':
        return J2(self.flat.copy(), self.shape_X)

    def __repr__(self) -> str:
        return f'J2(f={self.f}, j={self.j}, h={self.h}, dim_X={self.dim_X}, shape_X={self.shape_X})'

    # TODO
    # def __abs__(self, other: 'J2') -> 'J2':
    #     pass

    def compose(self, other: 'J2') -> 'J2':
        assert self.shape_X == (), 'J2.compose is only defined for scalar-valued J2 objects (for now)'
        retval = other.copied()
        retval.f[...] = self.f
        retval.j[...] *= self.j
        retval.h[...] *= self.j
        retval.h[...] += self.h * symmetric_outer_product(other.j, other.j)
        return retval

    def __iadd__(self, other: 'J2') -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X
            self.flat[...] += other.flat
        else:
            # other must be a constant of the correct shape.
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            self.f[...] += other
        return self

    def __add__(self, other: 'J2') -> 'J2':
        retval = self.copied()
        retval += other
        return retval

    def __isub__(self, other: 'J2') -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X
            self.flat[...] -= other.flat
        else:
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            self.f[...] -= other
        return self
    
    def __sub__(self, other: 'J2') -> 'J2':
        retval = self.copied()
        retval -= other
        return retval
    
    def __neg__(self) -> 'J2':
        retval = self.copied()
        retval.flat[...] = -retval.flat
        return retval
    
    def __pos__(self) -> 'J2':
        return self

    def __imul__(self, other: 'J2') -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X, f'shape_X of self ({self.shape_X}) and other ({other.shape_X}) must match'
            self.h[...] = self.f * other.h + 2 * symmetric_outer_product(self.j, other.j) + other.f * self.h
            self.j[...] = self.f * other.j + other.f * self.j
            self.f[...] = self.f * other.f
        else:
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            self.flat[...] *= other
        return self

    def __mul__(self, other: 'J2') -> 'J2':
        retval = self.copied()
        retval *= other
        return retval

    def __rmul__(self, other: Any) -> 'J2':
        # The algebra is commutative.
        return self.__mul__(other)
    
    def __itruediv__(self, other: Any) -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X
            self *= other.inverted()
        else:
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            self.flat[...] /= other
        return self

    def __truediv__(self, other: Any) -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X
            return self * other.inverted()
        else:
            # Assume other is a constant scalar.
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            retval = self.copied()
            retval.flat[...] /= other
            return retval
        
    def __rtruediv__(self, other: Any) -> 'J2':
        if isinstance(other, J2):
            assert self.shape_X == other.shape_X
            return other.__truediv__(self)
        else:
            # Assume other is a constant scalar.
            assert np.shape(other) == self.shape_f, f'if other is not J2, then np.shape(other) (which is {np.shape(other)}) must be {self.shape_f}'
            return self.inverted() * other
        
    # TODO: __rtruediv__ ? Not sure if this would work -- https://stackoverflow.com/questions/37310077/rtruediv-method-does-not-work-as-i-expect
    
    def __pow__(self, x: Any) -> 'J2':
        if isinstance(x, int):
            f = self.f[()]
            # TODO: Maybe special-case x = 0 and x = 1.
            j2_pow_n = J2.from_values(f**x, x * f**(x - 1), x * (x - 1) * f**(x - 2))
            return j2_pow_n.compose(self)
        else:
            return (self.log() * x).exp()
        # if isinstance(exponent, int):
        #     if exponent == 0:
        #         return J2.from_const(1, self.shape_X, self.dtype)
        #     elif exponent == 1:
        #         # TODO: Does this need to return a copy?
        #         # return self.copied()
        #         return self
        #     else:
        #         retval = J2.uninitialized(self.shape_X, self.dtype)
        #         retval.f[...] = self.f**exponent
        #         retval.j[...] = exponent * self.f**(exponent - 1) * self.j
        #         retval.h[...] = exponent * self.f**(exponent - 2) * \
        #                         (self.f * self.h + (exponent - 1) * \
        #                         symmetric_outer_product(self.j, self.j))
        #         return retval
        # else:
        #     assert False, 'todo: implement'
        #     # return (self.log() * exponent).exp()
        
    # TODO __call__, could evaluate the Taylor polynomial along a given vector.
        
    def invert(self):
        """Inverts this 2-jet number in place with no memory allocation."""
        self.f[...] = 1 / self.f
        # The line of code after this comment is equivalent to the following:
        # self.j[...] *= -self.f**2
        # self.h[...] *= -self.f**2
        self.flat[1:] *= -self.f**2
        self.h[...] += (2 / self.f) * symmetric_outer_product(self.j, self.j)

    def inverted(self) -> 'J2':
        retval = self.copied()
        retval.invert()
        return retval

    def exp(self) -> 'J2':
        # if isinstance(self.f[()], sp.Expr):
        #     exp_f = sp.exp(self.f[()])
        # else:
        #     exp_f = np.exp(self.f[()])
        exp_f = exp(self.f[()])
        j2_exp = J2.from_values(exp_f, exp_f, exp_f)
        return j2_exp.compose(self)
        # retval = self.copied()
        # retval.f[...] = 1
        # retval.h[...] += symmetric_outer_product(retval.j, retval.j)
        # retval.flat[...] *= exp_f
        # return retval
    
    def log(self) -> 'J2':
        # if isinstance(self.f[()], sp.Expr):
        #     log_f = sp.log(self.f[()])
        # else:
        #     log_f = np.log(self.f[()])
        log_f = log(self.f[()])
        j2_log = J2.from_values(log_f, 1 / self.f, -1 / self.f**2)
        return j2_log.compose(self)
        # return J2(
        #     log_f,
        #     self.j_flattened / self.f_flattened,
        #     (-self.j_flattened*self.j_flattened / self.f_flattened**2 + self.h_flattened / self.f_flattened)
        # )

    def cos(self) -> 'J2':
        # if isinstance(self.f[()], sp.Expr):
        #     cos_f = sp.cos(self.f[()])
        #     sin_f = sp.sin(self.f[()])
        # else:
        #     cos_f = np.cos(self.f[()])
        #     sin_f = np.sin(self.f[()])
        cos_f = cos(self.f[()])
        sin_f = sin(self.f[()])
        j2_cos = J2.from_values(cos_f, -sin_f, -cos_f)
        return j2_cos.compose(self)
    
    def sin(self) -> 'J2':
        # if isinstance(self.f[()], sp.Expr):
        #     sin_f = sp.sin(self.f[()])
        #     cos_f = sp.cos(self.f[()])
        # else:
        #     sin_f = np.sin(self.f[()])
        #     cos_f = np.cos(self.f[()])
        sin_f = sin(self.f[()])
        cos_f = cos(self.f[()])
        j2_sin = J2.from_values(sin_f, cos_f, -sin_f)
        return j2_sin.compose(self)

    def sqrt(self) -> 'J2':
        sqrt_f = sqrt(self.f[()])
        j2_sqrt = J2.from_values(sqrt_f, 1 / (sqrt_f * 2), -1 / (sqrt_f**3 * 4))
        return j2_sqrt.compose(self)

    def simplify(self):
        self.flat = simplify(self.flat)

    def simplified(self) -> 'J2':
        retval = self.copied()
        retval.simplify()
        return retval

def exp(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.exp(x)
    else:
        return np.exp(x)

def log(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.log(x)
    else:
        return np.log(x)
    
def cos(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.cos(x)
    else:
        return np.cos(x)
    
def sin(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.sin(x)
    else:
        return np.sin(x)

def sqrt(x: Any) -> Any:
    if isinstance(x, sp.Expr):
        return sp.sqrt(x)
    else:
        return np.sqrt(x)

def extend(f: Any, X: Any) -> Any:
    """Compute 2-jet extension of a symbolic expression f in the variable(s) X."""
    assert len(np.shape(X)) == 0 or len(np.shape(X)) == 1, f'X must be a scalar or a 1-dimensional array (for now), but it has shape {np.shape(X)}'
    j = vp.symbolic.D(f, X)
    h = vp.symbolic.D(j, X)
    return J2.from_values(f, j, h)

def test_stuff():
    a = J2.from_values(sp.var('f'), vp.symbolic.tensor('j', (2,)), vp.symbolic.tensor('h', (2, 2)))
    A = J2.from_values(sp.var('F'), vp.symbolic.tensor('J', (2,)), vp.symbolic.tensor('H', (2, 2)))
    print(f'a*A = {a*A}')

def test_inversion():
    j = J2.from_values(sp.var('a'), vp.symbolic.tensor('j', (2,)), vp.symbolic.tensor('h', (2, 2)))
    j_inv = j.inverted()
    j_inv_times_j = (j_inv * j).simplified()
    assert j_inv_times_j.f == 1, f'j_inv_times_j.f = {j_inv_times_j.f}'
    assert np.all(j_inv_times_j.j == 0), f'j_inv_times_j.j = {j_inv_times_j.j}'
    assert np.all(j_inv_times_j.h == 0), f'j_inv_times_j.h = {j_inv_times_j.h}'
    print('inversion is correct')

def test_pow():
    j = J2.from_values(sp.var('a'), vp.symbolic.tensor('j', (2,)), vp.symbolic.tensor('h', (2, 2)))
    expected_j_to_the_n = J2.from_const(1, j.shape_X, j.dtype)
    for n in range(0, 9):
        j_to_the_n = j**n
        error = simplify(j_to_the_n.flat - expected_j_to_the_n.flat)
        assert np.all(error == 0), f'n: {n}; j_to_the_n = {j_to_the_n} but expected_j_to_the_n = {expected_j_to_the_n} ; the error is {error}'
        print(f'j**{n} is correct for n = {n}')
        expected_j_to_the_n *= j

    j_inv = j.inverted()
    expected_j_to_the_n = j_inv.copied()
    for negative_n in range(1, 9):
        n = -negative_n
        j_to_the_n = j**n
        error = simplify(j_to_the_n.flat - expected_j_to_the_n.flat)
        assert np.all(error == 0), f'n: {n}; j_to_the_n = {j_to_the_n} but expected_j_to_the_n = {expected_j_to_the_n} ; the error is {error}'
        print(f'j_inv**{n} is correct for n = {n}')
        expected_j_to_the_n *= j_inv

    print('pow is correct')

# def test_other_stuff():
#     j = J2.from_values(sp.var('a'), sp.var('b'), sp.var('c'))
#     print(f'j**2 = {(j*j).simplified()}')
#     print(f'j**3 = {(j*j*j).simplified()}')
#     print(f'j**4 = {(j*j*j*j).simplified()}')
#     print(f'j**5 = {(j*j*j*j*j).simplified()}')
#     print(f'j**6 = {(j*j*j*j*j*j).simplified()}')
#     print(f'j**7 = {(j*j*j*j*j*j*j).simplified()}')
#     print(f'j**8 = {(j*j*j*j*j*j*j*j).simplified()}')
#     print(f'j**9 = {(j*j*j*j*j*j*j*j*j).simplified()}')
#     print(f'j**10 = {(j*j*j*j*j*j*j*j*j*j).simplified()}')

#     j_inv = j.inverted()
#     print(f'j_inv = {j_inv}')
#     print(f'j_inv**2 = {(j_inv*j_inv).simplified()}')
#     print(f'j_inv**3 = {(j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**4 = {(j_inv*j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**5 = {(j_inv*j_inv*j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**6 = {(j_inv*j_inv*j_inv*j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**7 = {(j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**8 = {(j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv).simplified()}')
#     print(f'j_inv**9 = {(j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv*j_inv).simplified()}')

def test_commutative_diagram():
    X = vp.symbolic.tensor('X', (2,))
    f = sp.Function('f')(*X)
    g = sp.Function('g')(*X)

    f_extended = extend(f, X)
    g_extended = extend(g, X)

    f_plus_g_extended = extend(f + g, X)
    error = simplify(f_plus_g_extended.flat - (f_extended + g_extended).flat)
    assert np.all(error == 0), f'f_plus_g_extended = {f_plus_g_extended} but f_extended + g_extended = {f_extended + g_extended} ; the error is {error}'
    print('__add__ commutes with extend')

    f_minus_g_extended = extend(f - g, X)
    error = simplify(f_minus_g_extended.flat - (f_extended - g_extended).flat)
    assert np.all(error == 0), f'f_minus_g_extended = {f_minus_g_extended} but f_extended - g_extended = {f_extended - g_extended} ; the error is {error}'
    print('__sub__ commutes with extend')

    f_times_g_extended = extend(f * g, X)
    error = simplify(f_times_g_extended.flat - (f_extended * g_extended).flat)
    assert np.all(error == 0), f'f_times_g_extended = {f_times_g_extended} but f_extended * g_extended = {f_extended * g_extended} ; the error is {error}'
    print('__mul__ commutes with extend')

    f_inverted_extended = extend(1 / f, X)
    error = simplify(f_inverted_extended.flat - (f_extended.inverted()).flat)
    assert np.all(error == 0), f'f_inverted_extended = {f_inverted_extended} but f_extended.inverted() = {f_extended.inverted()} ; the error is {error}'
    print('inversion commutes with extend')

    f_over_g_extended = extend(f / g, X)
    error = simplify(f_over_g_extended.flat - (f_extended / g_extended).flat)
    assert np.all(error == 0), f'f_over_g_extended = {f_over_g_extended} but f_extended / g_extended = {f_extended / g_extended} ; the error is {error}'
    print('__truediv__ commutes with extend')

    exp_f_extended = extend(exp(f), X)
    error = simplify(exp_f_extended.flat - (f_extended.exp()).flat)
    assert np.all(error == 0), f'exp_f_extended = {exp_f_extended} but f_extended.exp() = {f_extended.exp()} ; the error is {error}'
    print('exp commutes with extend')

    log_f_extended = extend(log(f), X)
    error = simplify(log_f_extended.flat - (f_extended.log()).flat)
    assert np.all(error == 0), f'log_f_extended = {log_f_extended} but f_extended.log() = {f_extended.log()} ; the error is {error}'
    print('log commutes with extend')

    cos_f_extended = extend(cos(f), X)
    error = simplify(cos_f_extended.flat - (f_extended.cos()).flat)
    assert np.all(error == 0), f'cos_f_extended = {cos_f_extended} but f_extended.cos() = {f_extended.cos()} ; the error is {error}'
    print('cos commutes with extend')

    sin_f_extended = extend(sin(f), X)
    error = simplify(sin_f_extended.flat - (f_extended.sin()).flat)
    assert np.all(error == 0), f'sin_f_extended = {sin_f_extended} but f_extended.sin() = {f_extended.sin()} ; the error is {error}'
    print('sin commutes with extend')

    for n in range(-7, 8):
        f_to_the_n_extended = extend(f**n, X)
        error = simplify(f_to_the_n_extended.flat - (f_extended**n).flat)
        assert np.all(error == 0), f'n: {n}; f_to_the_n_extended = {f_to_the_n_extended} but f_extended**n = {f_extended**n} ; the error is {error}'
        print(f'__pow__ commutes with extend for exponent {n}')

    a = sp.Rational(-3,7)
    b = sp.Rational(8,3)
    for i in range(33):
        x = a + i * (b - a) / 32
        f_to_the_x_extended = extend(f**x, X)
        error = simplify(f_to_the_x_extended.flat - (f_extended**x).flat)
        assert np.all(error == 0), f'x: {x}; f_to_the_x_extended = {f_to_the_x_extended} but f_extended**x = {f_extended**x} ; the error is {error}'
        print(f'__pow__ commutes with extend for exponent {x}')

def simple_hyperelasticity():
    """
    This is really just to test out J2 and compare it with num_dual.
    """
    alpha = 1000.0

    def tr(M: ndarray) -> Any:
        assert M.shape == (2, 2)
        return M[0,0] + M[1,1]

    def det(M: ndarray) -> Any:
        assert M.shape == (2, 2)
        return M[0,0] * M[1,1] - M[0,1] * M[1,0]

    def g(X: ndarray) -> ndarray:
        """Metric for paraboloid z = r^2 / 2."""
        retval = np.outer(X, X)
        retval[0,0] += 1
        retval[1,1] += 1
        # retval = np.eye(2, dtype=X.dtype)
        # retval += np.outer(X, X)
        return retval
    
    def g_inv(X: ndarray) -> ndarray:
        """Inverse metric for paraboloid z = r^2 / 2."""
        # TODO: Explicitly code this up
        return np.linalg.inv(g(X))
    
    def h(X: ndarray) -> ndarray:
        """Metric for funnel z = -r^{-1}."""
        # retval = np.eye(2, dtype=X.dtype)
        # r_squared = X.dot(X)
        # retval += np.outer(X, X) / r_squared**3
        retval = np.outer(X, X)
        retval /= X.dot(X)**3
        retval[0,0] += 1
        retval[1,1] += 1
        return retval
    
    def h_inv(X: ndarray) -> ndarray:
        """Inverse metric for funnel z = -r^{-1}."""
        # TODO: Explicitly code this up
        return np.linalg.inv(h(X))
    
    def C(X: ndarray, Y: ndarray, F: ndarray) -> ndarray:
        """Cauchy strain tensor."""
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        retval = np.ndarray((2, 2), dtype=F.dtype)
        retval[...] = g_inv(X)
        # print(f'retval = {retval}, retval.dtype = {retval.dtype}')
        retval @= F.T
        retval @= h(Y).astype(F.dtype)
        retval @= F
        return retval

    def W_density(X: ndarray, Y: ndarray, F: ndarray) -> ndarray:
        """Stored energy function for the hyperelastic body."""
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        c = C(X, Y, F)
        return alpha * (tr(c) - 2 - log(det(c)))
    
    def G_density(Y: ndarray) -> Any:
        """Gravitational potential on space."""
        r = np.linalg.norm(Y)
        return -1 / r

    def L_density(X: ndarray, Y: ndarray, F: ndarray) -> Any:
        """Lagrange density for the hyperelastic body."""
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        return W_density(X, Y, F) + G_density(Y)

    def compute_Dphi(phi: ndarray) -> ndarray:
        assert len(phi.shape) == 3
        assert phi.shape[0] == 2
        assert phi.shape[1] == phi.shape[2]
        n = phi.shape[1]

        Dphi = np.zeros((2,) + phi.shape, dtype=phi.dtype)
        n_arange = np.arange(1.0, n)
        # Polynomial derivative is shift and multiply by the exponent.
        # TODO: assign and then mul-assign to speed this up?
        Dphi[:, 0, :-1, :] = n_arange[None, :, None] * phi[:, 1:, :]
        Dphi[:, 1, :, :-1] = n_arange[None, None, :] * phi[:, :, 1:]
        return Dphi

    def evaluate_phi_and_Dphi(phi: ndarray, Dphi: ndarray, X: ndarray) -> Tuple[ndarray, ndarray]:
        """Evaluate the spatial point and deformation gradient corresponding to the given body point."""
        assert len(phi.shape) == 3, f'phi.shape = {phi.shape}'
        assert phi.shape[0] == 2
        assert phi.shape[1] == phi.shape[2]
        n = phi.shape[1]
        assert len(Dphi.shape) == 4
        assert Dphi.shape[0] == 2
        assert Dphi.shape[1] == 2
        assert Dphi.shape[2] == Dphi.shape[3] == n

        # Polynomial evaluation at X.
        n_arange = np.arange(n)
        X0_powers = X[0]**n_arange
        X1_powers = X[1]**n_arange
        phi_X = (phi @ X0_powers) @ X1_powers
        Dphi_X = (Dphi @ X0_powers) @ X1_powers
        # phi_X = np.einsum('ijk,j,k->i', phi, X0_powers, X1_powers)
        # Dphi_X = np.einsum('ijkl,k,l->ij', Dphi, X0_powers, X1_powers)
        return phi_X, Dphi_X

    # def evaluate_phi(phi: ndarray, X: ndarray) -> ndarray:
    #     """Evaluate the spatial point corresponding to the given body point."""
    #     assert len(phi.shape) == 3
    #     assert phi.shape[0] == 2
    #     assert phi.shape[1] == phi.shape[2]
    #     n = phi.shape[1]
    #     assert X.shape == (2,)
    #     X0_powers = X[0]**np.arange(n)
    #     X1_powers = X[1]**np.arange(n)
    #     return np.einsum('ijk,j,k->i', phi, X0_powers, X1_powers)
    
    # def evaluate_Dphi(Dphi: ndarray, X: ndarray) -> ndarray:
    #     """Evaluate the deformation gradient corresponding to the given body point."""
    #     assert len(Dphi.shape) == 4
    #     assert Dphi.shape[0] == 2
    #     assert Dphi.shape[1] == 2
    #     assert Dphi.shape[2] == Dphi.shape[3]
    #     n = Dphi.shape[2]
    #     assert X.shape == (2,)
    #     X0_powers = X[0]**np.arange(n)
    #     X1_powers = X[1]**np.arange(n)

    def evaluate_L_density(X: ndarray, phi: ndarray, Dphi: ndarray) -> Any:
        """Evaluate the Lagrange density corresponding to the given body point."""
        phi_X, Dphi_X = evaluate_phi_and_Dphi(phi, Dphi, X)
        return L_density(X, phi_X, Dphi_X)
    
    domain_radius = 1.0
    radial_row_count = 33
    outer_angular_point_count = 90
    row_point_count_v = np.linspace(1, outer_angular_point_count, radial_row_count, endpoint=True, dtype=int)
    point_count = np.sum(row_point_count_v)
    integration_point_v = np.ndarray((point_count, 2), dtype=np.float64)
    integration_point_weight_v = np.ndarray((point_count,), dtype=np.float64)
    point_index = 0
    for i, row_point_count in enumerate(row_point_count_v):
        r = i / (radial_row_count - 1) * domain_radius
        for j in range(row_point_count):
            theta = j / row_point_count * 2 * np.pi
            integration_point_v[point_index, 0] = r * np.cos(theta)
            integration_point_v[point_index, 1] = r * np.sin(theta)
            # TODO figure out boundary point weights
            integration_point_weight_v[point_index] = 1.0
            point_index += 1
    integration_point_weight_v *= np.pi * domain_radius**2 / np.sum(integration_point_weight_v)

    def L(phi: ndarray) -> Any:
        """Lagrange functional for the hyperelastic body."""
        Dphi = compute_Dphi(phi)
        return np.sum(integration_point_weight_v * np.apply_along_axis(evaluate_L_density, 1, integration_point_v, phi, Dphi))
    
    polynomial_degree = 3

    # Set phi to be the identity map.
    phi = np.zeros((2, polynomial_degree+1, polynomial_degree+1), dtype=np.float64)
    # Constant offset in the x0 direction.
    phi[0, 0, 0] = 1.5
    # Linear terms in the x0 and x1 directions.
    phi[0, 1, 0] = 1.0
    phi[1, 0, 1] = 1.0

    print(f'L(phi) = {L(phi)}')

    # Use J2 to evaluate L, DL, D2L.

    N = np.prod(phi.shape, dtype=np.int32)
    J2N_zero = J2.zero((N,), dtype=np.float64)

    J2_phi = np.ndarray(phi.shape, dtype=J2)
    # This is dumb, but an explicit copy is necessary in order to not have all elements point at J2N_zero.
    for i in range(N):
        J2_phi.flat[i] = J2N_zero.copied()
    for i in range(N):
        J2_phi.flat[i].f[...] = phi.flat[i]
    # J2_phi[0, 0, 0].f[...] = 1.5
    # J2_phi[0, 1, 0].f[...] = 1.0
    # J2_phi[1, 0, 1].f[...] = 1.0
    # Now set up the derivative components.
    # print(f'J2_phi_flattened = {J2_phi_flattened}')
    for i in range(N):
        J2_phi.flat[i].j[i] = 1.0
    # TODO Why is this setting j[0] to 1.0 for each of J2_phi_flattened?
    # J2_phi.flat[0].j[0] = 1.0
    # J2_phi_flattened[1].j[1] = 1.0

    # print(f'J2_phi = {J2_phi}')
    print('----------------------------')
    start = time.time()
    L_J2_phi = L(J2_phi)
    end = time.time()
    print(f'Time taken to evaluate L(J2_phi): {end - start} seconds')
    print(f'L(J2_phi) = {L_J2_phi}')
    # print()

    # Now use num_dual to compare.
    start = time.time()
    L_phi, DL_phi, D2L_phi = nd.hessian(lambda phi_flattened :L(array(phi_flattened).reshape(phi.shape)), phi.flatten())
    DL_phi = array(DL_phi)
    D2L_phi = array(D2L_phi)
    end = time.time()
    print(f'Time taken to evaluate L(phi), DL(phi), D2L(phi): {end - start} seconds')
    # print(f'L_phi = {L_phi}')
    # print(f'DL_phi = {DL_phi}')
    # print(f'D2L_phi = {D2L_phi}')
    # print()

    # Compare the results.
    error = np.abs(L_J2_phi.f - L_phi)
    assert error <= 1.0e-12, f'Abs error in L(phi) is {error}'
    error = np.max(np.abs(DL_phi - L_J2_phi.j))
    assert error <= 1.0e-11, f'Max abs error in DL(phi) is {error}'
    error = np.max(np.abs(D2L_phi - L_J2_phi.h))
    assert error <= 1.0e-10, f'Max abs error in D2L(phi) is {error}'




if __name__ == '__main__':
    # test_stuff()
    # test_inversion()
    # test_other_stuff()
    # test_pow()
    # test_commutative_diagram()

    simple_hyperelasticity()




# def f(X: Any) -> ndarray:
#     return array([
#         3*X*X*X - 2*X*X + 1,
#         X*X - 5*X,
#     ])

# def g(Y: ndarray) -> Any:
#     return Y[0]**2 + exp(Y[1]**2)

# # def TODO implement example function and test D2

# x = sp.var('x')
# X = J2.from_var(x)
# print(f'X = {X}')

# F_D2 = f(X)
# print(f'F = {F_D2}')
# G_circ_F_D2 = g(F_D2)
# print(f'G \circ F (as D2) = {G_circ_F_D2}')

# G_circ_F_symbolic = g(f(x))

# print(f'Symbolic: G \circ F = {G_circ_F_symbolic}')
# DG_circ_F_symbolic = vp.symbolic.D(G_circ_F_symbolic, x)
# print(f'Symbolic: DG_circ_F = {DG_circ_F_symbolic}')
# D2G_circ_F_symbolic = vp.symbolic.D(DG_circ_F_symbolic, x)
# print(f'Symbolic: D2G_circ_F = {D2G_circ_F_symbolic}')

# assert (G_circ_F_symbolic - G_circ_F_D2.f).simplify() == 0, f'G_circ_F_symbolic = {G_circ_F_symbolic} but G_circ_F_D2.f = {G_circ_F_D2.f} ; the difference is {G_circ_F_symbolic - G_circ_F_D2.f}'
# assert (DG_circ_F_symbolic - G_circ_F_D2.j).simplify() == 0, f'DG_circ_F_symbolic = {DG_circ_F_symbolic} but G_circ_F_D2.j = {G_circ_F_D2.j} ; the difference is {DG_circ_F_symbolic - G_circ_F_D2.j}'
# assert (D2G_circ_F_symbolic - G_circ_F_D2.h).simplify() == 0, f'D2G_circ_F_symbolic = {D2G_circ_F_symbolic} but G_circ_F_D2.h = {G_circ_F_D2.h} ; the difference is {D2G_circ_F_symbolic - G_circ_F_D2.h}'

# print(f'exp(X) = {X.exp()}')
# print(f'log(X) = {X.log()}')

# # Now test D2 with multivariate functions.

# y = vp.symbolic.tensor('y', (2,))
# Y = J2.from_var(y)
# print(f'Y = {Y}')

# def q(Y: ndarray) -> Any:
#     return Y.dot(Y)

# Q_D2 = q(Y)
# print(f'Q_D2 = {Q_D2}')

# Q_symbolic = q(y)
# print(f'Symbolic: Q = {Q_symbolic}')
# DQ_symbolic = vp.symbolic.D(Q_symbolic, y)
# print(f'Symbolic: DQ = {DQ_symbolic}')
# D2Q_symbolic = vp.symbolic.D(DQ_symbolic, y)
# print(f'Symbolic: D2Q = {D2Q_symbolic}')

# assert np.all(simplify(Q_symbolic - Q_D2.f) == 0), f'Q_symbolic = {Q_symbolic} but Q_D2.f = {Q_D2.f} ; the difference is {Q_symbolic - Q_D2.f}'
# assert np.all(simplify(DQ_symbolic - Q_D2.j) == 0), f'DQ_symbolic = {DQ_symbolic} but Q_D2.j = {Q_D2.j} ; the difference is {DQ_symbolic - Q_D2.j}'
# assert np.all(simplify(D2Q_symbolic - Q_D2.h) == 0), f'D2Q_symbolic = {D2Q_symbolic} but Q_D2.h = {Q_D2.h} ; the difference is {D2Q_symbolic - Q_D2.h}'
