import numpy as np

from typing import Any

def sym_eval(expr: Any, x_sym: Any, x_value: Any) -> Any:
    return np.vectorize(lambda e: e.subs(x_sym, x_value))(expr)

def sym_D(expr: Any, x_sym: Any) -> Any:
    return np.vectorize(lambda e: e.diff(x_sym))(expr)

def sym_simplify(expr: Any) -> Any:
    return np.vectorize(lambda e: e.simplify())(expr)
