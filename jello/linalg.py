from numpy import ndarray
from typing import Any

#
# Linear algebraic functions specific to 2D.
#

def tr(M: ndarray) -> Any:
    """2D-specific trace function."""
    assert M.shape == (2, 2)
    return M[0,0] + M[1,1]

def det(M: ndarray) -> Any:
    """2D-specific determinant function."""
    assert M.shape == (2, 2)
    return M[0,0]*M[1,1] - M[0,1]*M[1,0]
