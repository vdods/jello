import numpy as np
import num_dual as nd

from .linalg import tr, det
from numpy import ndarray
from typing import Any, Callable, Tuple

class HyperelasticMaterial:
    def __init__(self, *, alpha: Any, uniform_mass_density: Any):
        self.alpha = alpha
        self.uniform_mass_density = uniform_mass_density

    def y1(self, x: Any) -> Any:
        """Stored energy density corresponding to changes in length."""
        retval = self.alpha*(x - 2)
        return retval

    def y2(self, x: Any) -> Any:
        """Stored energy density corresponding to changes in area."""
        retval = -self.alpha*np.log(x)
        return retval
    
    def cauchy_strain_tensor(
        self,
        X: ndarray,
        Y: ndarray,
        F: ndarray,
        # body_metric_inv_fn: Callable[[ndarray], ndarray],
        # TODO: Just use the spatial manifold metric function here, not requiring an actual SpatialManifold object.
        spatial_manifold: 'SpatialManifold',
    ) -> ndarray:
        """
        Cauchy strain tensor.  Encodes the squares of the principal stretches (i.e. squared local changes in length).
        X is the body point, Y is the space point.
        """
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        # g_inv_X = body_metric_inv_fn(X)
        h_Y = spatial_manifold.spatial_metric(Y)
        assert h_Y.shape == (2, 2)
        # return g_inv_X.dot(F.T).dot(h_Y).dot(F)
        return F.T.dot(h_Y).dot(F)
    
    def cauchy_tensor_invariants(
        self,
        X: ndarray,
        Y: ndarray,
        F: ndarray,
        # body_metric_inv_fn: Callable[[ndarray], ndarray],
        # TODO: Just use the spatial manifold metric function here, not requiring an actual SpatialManifold object.
        spatial_manifold: 'SpatialManifold',
    ) -> Tuple[Any, Any]:
        """
        Tensor invariants for the Cauchy strain tensor.  X is the body point, Y is the space point.
        """
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        C = self.cauchy_strain_tensor(X, Y, F, spatial_manifold)
        assert C.shape == (2, 2)
        return (tr(C), det(C))
    
    def stored_energy_density(
        self,
        X: ndarray,
        Y: ndarray,
        F: ndarray,
        # body_metric_inv_fn: Callable[[ndarray], ndarray],
        # TODO: Just use the spatial manifold metric function here, not requiring an actual SpatialManifold object.
        spatial_manifold: 'SpatialManifold',
        *,
        y1_component: bool = True,
        y2_component: bool = True,
    ) -> Any:
        """
        Stored energy density function for a given deformation gradient F.  X is the body point, Y is the space point.
        y1_component and y2_component control which components of the stored energy density are computed.
        """
        assert y1_component or y2_component
        C = self.cauchy_strain_tensor(X, Y, F, spatial_manifold)
        if y1_component:
            retval = self.y1(tr(C))
            if y2_component:
                retval += self.y2(det(C))
            return retval
        else:
            assert y2_component
            return self.y2(det(C))

    def potential_energy_density(
        self,
        X: ndarray,
        Y: ndarray,
        mass_specific_potential: Callable[[ndarray], Any],
    ) -> Any:
        """The potential energy density is the energy per unit reference volume of the material."""
        return self.uniform_mass_density * mass_specific_potential(Y)

    def lagrangian_density(
        self,
        X: ndarray,
        Y: ndarray,
        F: ndarray,
        # body_metric_inv_fn: Callable[[ndarray], ndarray],
        spatial_manifold: 'SpatialManifold',
        mass_specific_potential: Callable[[ndarray], Any],
    ) -> Any:
        """Lagrangian density function for a given body point X and spatial manifold."""
        assert X.shape == (2,)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        # If there were a kinetic energy density, it would have a positive sign, and the stored energy density
        # and potential energy density terms would have a negative sign.
        return self.stored_energy_density(X, Y, F, spatial_manifold) + self.potential_energy_density(X, Y, mass_specific_potential)
