import numpy as np
import num_dual as nd
import scipy as sc
import vtk
import vtk.util.numpy_support

from .fem2d import FEFunctionSpace2D
from .measure import DiscreteMeasure
from .material import HyperelasticMaterial
from .spatial_manifold import SpatialManifold
from .phi2d import phi, Dphi, phi_and_Dphi
from .j1_otimes_j1_from_function import J1_otimes_J1_from_function
from numpy import array, ndarray
from typing import Any, Callable, Tuple, Optional

class HyperelasticBody:
    def __init__(
        self,
        cfg: ndarray,
        function_space: FEFunctionSpace2D,
        measure: 'DiscreteMeasure',
        material: HyperelasticMaterial,
        spatial_manifold: 'SpatialManifold',
        mass_specific_potential: Callable[[ndarray], Any],
    ):
        """
        cfg is the body configuration (i.e. a vector in the finite element function space).
        """
        assert cfg.shape == (function_space.vector_function_dim,)

        self.cfg = cfg
        self.function_space = function_space
        self.measure = measure
        self.material = material
        self.spatial_manifold = spatial_manifold
        self.mass_specific_potential = mass_specific_potential

    @staticmethod
    def from_function(
        function: Callable[[ndarray], ndarray],
        function_space: FEFunctionSpace2D,
        measure: 'DiscreteMeasure',
        material: HyperelasticMaterial,
        spatial_manifold: 'SpatialManifold',
        mass_specific_potential: Callable[[ndarray], Any],
    ) -> 'HyperelasticBody':
        """
        Produce a HyperelasticBody from the given function and function space.
        """
        assert measure.integration_point_count >= function_space.scalar_function_dim, 'the integration measure has {measure.integration_point_count} points, but the scalar function space has {function_space.scalar_function_dim} dimensions'
        cfg = J1_otimes_J1_from_function(function_space.domain_corner_v, function_space.vertex_count_v, array([2]), function).flatten()
        return HyperelasticBody(cfg, function_space, measure, material, spatial_manifold, mass_specific_potential)

    @staticmethod
    def identity_embedding(
        function_space: FEFunctionSpace2D,
        measure: 'DiscreteMeasure',
        material: HyperelasticMaterial,
        spatial_manifold: 'SpatialManifold',
        mass_specific_potential: Callable[[ndarray], Any],
    ) -> 'HyperelasticBody':
        """
        Produce a HyperelasticBody for the identity body configuration.  If either the body or space Riemannian metric is not
        the Euclidean metric, then this is a coordinate-specific quantity.
        """
        return HyperelasticBody.from_function(lambda X: X, function_space, measure, material, spatial_manifold, mass_specific_potential)

    def with_cfg(self, cfg: ndarray) -> 'HyperelasticBody':
        """
        Produce a HyperelasticBody of the same kind (i.e. function_space, material, spatial_manifold)
        but with the given cfg.
        """
        if not isinstance(cfg, ndarray):
            cfg = array(cfg)
        assert cfg.shape == (self.function_space.vector_function_dim,), f'cfg.shape = {cfg.shape}'
        return HyperelasticBody(cfg, self.function_space, self.measure, self.material, self.spatial_manifold, self.mass_specific_potential)

    def subdivided(self, *, refine_measure: Optional[bool] = True) -> 'HyperelasticBody':
        """
        Return a HyperelasticBody using the subdivided function space, which should have the same configuration
        but at twice the resolution.  Note that this does NOT subdivide the DiscreteMeasure.
        """
        F_subdivided = self.function_space.subdivided()
        if refine_measure:
            print(f'measure before refinement: {self.measure}')
            measure = DiscreteMeasure.gauss_legendre_5x5_on_rectangle(F_subdivided.domain_corner_v, F_subdivided.vertex_count_v)
            print(f'refined measure: {measure}')
        else:
            measure = self.measure

        return HyperelasticBody.from_function(lambda X: self.phi(X), F_subdivided, measure, self.material, self.spatial_manifold, self.mass_specific_potential)

    def subdivided_linear(self, *, refine_measure: Optional[bool] = True) -> 'HyperelasticBody':
        # Increase vertex count along each axis by 1, instead of splitting each element into 4.

        F_subdivided = self.function_space.subdivided_linear()
        if refine_measure:
            print(f'measure before refinement: {self.measure}')
            measure = DiscreteMeasure.gauss_legendre_5x5_on_rectangle(F_subdivided.domain_corner_v, F_subdivided.vertex_count_v)
            print(f'refined measure: {measure}')
        else:
            measure = self.measure

        return HyperelasticBody.from_function(lambda X: self.phi(X), F_subdivided, measure, self.material, self.spatial_manifold, self.mass_specific_potential)

    # TODO: This probably belongs in FEFunctionSpace2D.
    def phi_and_Dphi(self, X: ndarray, *, compute_phi: bool = True, compute_Dphi: bool = True) -> Tuple[ndarray, ndarray]:
        """
        Compute the spatial point and/or deformation gradient corresponding to the given body point.  The
        compute_phi and compute_Dphi flags control which quantities are computed.
        """
        assert X.shape == (2,)
        return phi_and_Dphi(self.cfg.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, X, compute_phi=compute_phi, compute_Dphi=compute_Dphi)
    
    def phi(self, X: ndarray) -> ndarray:
        """
        Compute the spatial point corresponding to the given body point.  If you want to compute both this and the
        deformation gradient, then it is more efficient to call phi_and_Dphi instead.
        """
        assert X.shape == (2,)
        return phi(self.cfg.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, X)

    def Dphi(self, X: ndarray) -> ndarray:
        """
        Compute the deformation gradient corresponding to the given body point.  If you want to compute both this and the
        spatial point, then it is more efficient to call phi_and_Dphi instead.
        """
        assert X.shape == (2,)
        return Dphi(self.cfg.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, X)
    
    def cauchy_tensor_invariants(
        self,
        X: ndarray,
    ) -> Tuple[Any, Any]:
        """
        Tensor invariants for the Cauchy strain tensor at the given body point.
        """
        assert X.shape == (2,)
        Y, F = self.phi_and_Dphi(X)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        return self.material.cauchy_tensor_invariants(X, Y, F, self.spatial_manifold)
    
    def stored_energy_density(
        self,
        X: ndarray,
        *,
        y1_component: bool = True,
        y2_component: bool = True,
    ) -> Any:
        """
        Returns the stored energy density for a given body point X.
        y1_component and y2_component control which components of the stored energy density are computed.
        """
        assert X.shape == (2,)
        Y, F = self.phi_and_Dphi(X)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        return self.material.stored_energy_density(X, Y, F, self.spatial_manifold, y1_component=y1_component, y2_component=y2_component)
        
    def potential_energy_density(self, X: ndarray) -> Any:
        """
        Returns the potential energy density for a given body point X.
        The potential energy density is the energy per unit reference volume of the material.
        """
        assert X.shape == (2,)
        Y = self.phi(X)
        return self.material.potential_energy_density(X, Y, self.mass_specific_potential)

    def lagrangian_density(self, X: ndarray) -> Any:
        """Lagrangian density function for a given body point X and spatial manifold."""
        assert X.shape == (2,)
        # X is the body point, Y is the space point, and F is the deformation gradient.
        Y, F = self.phi_and_Dphi(X)
        assert Y.shape == (2,)
        assert F.shape == (2, 2)
        return self.material.lagrangian_density(X, Y, F, self.spatial_manifold, self.mass_specific_potential)

    def stored_energy(self) -> Any:
        """Stored energy functional for a given body configuration."""
        return self.measure.integrate(lambda X: self.stored_energy_density(X))

    def potential_energy(self) -> Any:
        """Potential energy functional for a given body configuration."""
        return self.measure.integrate(lambda X: self.potential_energy_density(X))

    def lagrangian(self) -> Any:
        """Lagrangian (action) functional for a given body configuration."""
        return self.measure.integrate(lambda X: self.lagrangian_density(X))
    
    def write_vtu(self, *, filename: str, body_mesh_vertex_count_v: Optional[ndarray] = array([17, 17])):
        """
        Write this body as a VTU (VTK Unstructured Grid) file with quad cells.
        
        VTU files store unstructured grids with point data (scalars/vectors).
        All fields are attached as "point data" arrays, which allows ParaView
        to switch between them using the "Coloring" dropdown.
        
        Args:
            filename: Output filename
            points: (N, 3) point coordinates
            quads: (M, 4) quad connectivity
            scalar_fields: dict of (N,) arrays
            vector_fields: dict of (N, 3) arrays
        """

        # Sample the body configuration to get the mesh points.
        body_point_x_v = np.linspace(self.function_space.domain_corner_v[0,0], self.function_space.domain_corner_v[1,0], body_mesh_vertex_count_v[0], endpoint=True)
        body_point_y_v = np.linspace(self.function_space.domain_corner_v[0,1], self.function_space.domain_corner_v[1,1], body_mesh_vertex_count_v[1], endpoint=True)
        body_point_x_mesh_v, body_point_y_mesh_v = np.meshgrid(body_point_x_v, body_point_y_v, indexing='ij')
        body_point_x_mesh_v = body_point_x_mesh_v.flatten()
        body_point_y_mesh_v = body_point_y_mesh_v.flatten()
        body_point_z_mesh_v = np.zeros_like(body_point_x_mesh_v)
        body_point_xy_v = np.column_stack([body_point_x_mesh_v, body_point_y_mesh_v])
        body_point_xyz_v = np.column_stack([body_point_xy_v, body_point_z_mesh_v])

        space_point_xy_v = np.apply_along_axis(lambda body_point: self.phi(body_point), 1, body_point_xy_v)
        space_point_z_v = np.apply_along_axis(self.spatial_manifold.spatial_embedding_z, 1, space_point_xy_v)
        space_point_xyz_v = np.column_stack([space_point_xy_v, space_point_z_v])

        quad_v = []
        for i in range(body_mesh_vertex_count_v[0] - 1):
            for j in range(body_mesh_vertex_count_v[1] - 1):
                ij = i * body_mesh_vertex_count_v[1] + j
                iJ = i * body_mesh_vertex_count_v[1] + j + 1
                Ij = (i + 1) * body_mesh_vertex_count_v[1] + j
                IJ = (i + 1) * body_mesh_vertex_count_v[1] + j + 1
                quad_v.append([ij, iJ, IJ, Ij])
        quad_v = np.array(quad_v, dtype=np.int32)

        # Create unstructured grid
        grid = vtk.vtkUnstructuredGrid()
        
        # Set points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(space_point_xyz_v, deep=True))
        grid.SetPoints(vtk_points)
        
        # Set cells (quads)
        for quad in quad_v:
            cell = vtk.vtkQuad()
            cell.GetPointIds().SetId(0, int(quad[0]))
            cell.GetPointIds().SetId(1, int(quad[1]))
            cell.GetPointIds().SetId(2, int(quad[2]))
            cell.GetPointIds().SetId(3, int(quad[3]))
            grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        
        #
        # Compute the various scalar fields.
        #

        stored_energy_density_y1_v = np.apply_along_axis(lambda body_point: self.stored_energy_density(body_point, y1_component=True, y2_component=False), 1, body_point_xy_v)
        stored_energy_density_y2_v = np.apply_along_axis(lambda body_point: self.stored_energy_density(body_point, y1_component=False, y2_component=True), 1, body_point_xy_v)
        stored_energy_density_v = np.apply_along_axis(lambda body_point: self.stored_energy_density(body_point), 1, body_point_xy_v)
        cauchy_tensor_invariant_tr_v = np.apply_along_axis(lambda body_point: self.cauchy_tensor_invariants(body_point)[0], 1, body_point_xy_v)
        cauchy_tensor_invariant_det_v = np.apply_along_axis(lambda body_point: self.cauchy_tensor_invariants(body_point)[1], 1, body_point_xy_v)
        potential_energy_density_v = np.apply_along_axis(lambda body_point: self.potential_energy_density(body_point), 1, body_point_xy_v)
        lagrangian_density_v = np.apply_along_axis(lambda body_point: self.lagrangian_density(body_point), 1, body_point_xy_v)

        scalar_field_v = [
            ('Stored Energy Density (alpha*(tr(C) - 2) term)', stored_energy_density_y1_v),
            ('Stored Energy Density (-alpha*log(det(C)) term)', stored_energy_density_y2_v),
            ('Stored Energy Density', stored_energy_density_v),
            ('tr(C)', cauchy_tensor_invariant_tr_v),
            ('det(C)', cauchy_tensor_invariant_det_v),
            ('Potential Energy Density', potential_energy_density_v),
            ('Lagrangian Density', lagrangian_density_v),
        ]
        for name, values in scalar_field_v:
            vtk_array = vtk.util.numpy_support.numpy_to_vtk(values, deep=True)
            vtk_array.SetName(name)
            grid.GetPointData().AddArray(vtk_array)

        # Set Lagrangian Density as active (ParaView will use this by default)
        grid.GetPointData().SetActiveScalars('Lagrangian Density')

        #
        # Compute the various vector fields.
        # - negative gradient of Lagrangian
        # - negative gradient of potential energy
        # - negative gradient of stored energy
        # - negative gradient of stored energy (alpha*tr(C) term)
        # - negative gradient of stored energy (-alpha*log(det(C)) term)
        # - vector fields corresponding to eigenvectors of hessian of Lagrangian
        #

        def stored_energy_y1(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            body = self.with_cfg(body_cfg)
            stored_energy_y1_density = lambda X: body.stored_energy_density(X, y1_component=True, y2_component=False)
            return self.measure.integrate(stored_energy_y1_density)
        
        def stored_energy_y2(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            body = self.with_cfg(body_cfg)
            stored_energy_y2_density = lambda X: body.stored_energy_density(X, y1_component=False, y2_component=True)
            return self.measure.integrate(stored_energy_y2_density)
        
        def stored_energy(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            body = self.with_cfg(body_cfg)
            stored_energy_density = lambda X: body.stored_energy_density(X)
            return self.measure.integrate(stored_energy_density)
        
        def potential_energy(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            body = self.with_cfg(body_cfg)
            potential_energy_density = lambda X: body.potential_energy_density(X)
            return self.measure.integrate(potential_energy_density)
        
        def lagrangian(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            body = self.with_cfg(body_cfg)
            lagrangian_density = lambda X: body.lagrangian_density(X)
            return self.measure.integrate(lagrangian_density)

        # TODO: Could compute the metric once and re-use.        
        neg_grad_stored_energy_y1 = -self.function_space.vector_field_gradient(stored_energy_y1, self.cfg, self.measure, self.spatial_manifold)
        neg_grad_stored_energy_y2 = -self.function_space.vector_field_gradient(stored_energy_y2, self.cfg, self.measure, self.spatial_manifold)
        neg_grad_stored_energy = -self.function_space.vector_field_gradient(stored_energy, self.cfg, self.measure, self.spatial_manifold)
        neg_grad_potential_energy = -self.function_space.vector_field_gradient(potential_energy, self.cfg, self.measure, self.spatial_manifold)
        neg_grad_lagrangian = -self.function_space.vector_field_gradient(lagrangian, self.cfg, self.measure, self.spatial_manifold)

        # Now sample each vector field at the 2d body points.
        neg_grad_stored_energy_y1_xy_v = np.apply_along_axis(lambda body_point: phi(neg_grad_stored_energy_y1.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)
        neg_grad_stored_energy_y2_xy_v = np.apply_along_axis(lambda body_point: phi(neg_grad_stored_energy_y2.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)
        neg_grad_stored_energy_xy_v = np.apply_along_axis(lambda body_point: phi(neg_grad_stored_energy.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)
        neg_grad_potential_energy_xy_v = np.apply_along_axis(lambda body_point: phi(neg_grad_potential_energy.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)
        neg_grad_lagrangian_xy_v = np.apply_along_axis(lambda body_point: phi(neg_grad_lagrangian.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)

        def Dz(space_point_xy: ndarray, space_vector_xy: ndarray) -> Any:
            return nd.first_derivative(lambda t: self.spatial_manifold.spatial_embedding_z(space_point_xy + t * space_vector_xy), 0.0)[1]

        def vector_field_z(space_vector_xy_v: ndarray) -> ndarray:
            retval = np.zeros_like(space_vector_xy_v[:,0])
            for i, (space_point_xy, space_vector_xy) in enumerate(zip(space_point_xy_v, space_vector_xy_v)):
                retval[i] = Dz(space_point_xy, space_vector_xy)
            return retval
        
        # def verify_tangency_residual_of_vector_field(vector_field_xyz_v: ndarray):
        #     residual = np.max([np.abs(self.spatial_manifold.tangency_residual_of_vector(space_point_xyz, vector_field_xyz_v[i])) for i, space_point_xyz in enumerate(space_point_xyz_v)])
        #     assert residual < 1.0e-10

        neg_grad_stored_energy_y1_z_v = vector_field_z(neg_grad_stored_energy_y1_xy_v)
        neg_grad_stored_energy_y2_z_v = vector_field_z(neg_grad_stored_energy_y2_xy_v)
        neg_grad_stored_energy_z_v = vector_field_z(neg_grad_stored_energy_xy_v)
        neg_grad_potential_energy_z_v = vector_field_z(neg_grad_potential_energy_xy_v)
        neg_grad_lagrangian_z_v = vector_field_z(neg_grad_lagrangian_xy_v)

        neg_grad_stored_energy_y1_xyz_v = np.column_stack([neg_grad_stored_energy_y1_xy_v, neg_grad_stored_energy_y1_z_v])
        neg_grad_stored_energy_y2_xyz_v = np.column_stack([neg_grad_stored_energy_y2_xy_v, neg_grad_stored_energy_y2_z_v])
        neg_grad_stored_energy_xyz_v = np.column_stack([neg_grad_stored_energy_xy_v, neg_grad_stored_energy_z_v])
        neg_grad_potential_energy_xyz_v = np.column_stack([neg_grad_potential_energy_xy_v, neg_grad_potential_energy_z_v])
        neg_grad_lagrangian_xyz_v = np.column_stack([neg_grad_lagrangian_xy_v, neg_grad_lagrangian_z_v])

        # verify_tangency_residual_of_vector_field(neg_grad_stored_energy_y1_xyz_v)
        # verify_tangency_residual_of_vector_field(neg_grad_stored_energy_y2_xyz_v)
        # verify_tangency_residual_of_vector_field(neg_grad_stored_energy_xyz_v)
        # verify_tangency_residual_of_vector_field(neg_grad_potential_energy_xyz_v)
        # verify_tangency_residual_of_vector_field(neg_grad_lagrangian_xyz_v)


        vector_field_v = [
            ('Negative Gradient of Stored Energy alpha*tr(C) term', neg_grad_stored_energy_y1_xyz_v),
            ('Negative Gradient of Stored Energy -alpha*log(det(C)) term', neg_grad_stored_energy_y2_xyz_v),
            ('Negative Gradient of Stored Energy', neg_grad_stored_energy_xyz_v),
            ('Negative Gradient of Potential Energy', neg_grad_potential_energy_xyz_v),
            ('Negative Gradient of Lagrangian', neg_grad_lagrangian_xyz_v),
        ]

        # Compute the metric at this body configuration.
        metric = self.function_space.vector_field_metric(self.cfg, self.measure, self.spatial_manifold)

        D2L = nd.hessian(lambda cfg: self.with_cfg(cfg).lagrangian(), self.cfg)[2]
        D2L_eigenvalues, D2L_eigenvectors = sc.linalg.eigh(D2L, metric)
        for i, eigenvalue in enumerate(D2L_eigenvalues):
            eigenvector = D2L_eigenvectors[:,i]
            # Now sample each vector field at the 2d body points.
            eigenvector_xy_v = np.apply_along_axis(lambda body_point: phi(eigenvector.reshape(self.function_space.vector_function_shape), self.function_space.domain_corner_v, body_point), 1, body_point_xy_v)
            eigenvector_z_v = vector_field_z(eigenvector_xy_v)
            eigenvector_xyz_v = np.column_stack([eigenvector_xy_v, eigenvector_z_v])
            # verify_tangency_residual_of_vector_field(eigenvector_xyz_v)
            vector_field_v.append((f'| Vector field for {i:03d}th eigenvalue {eigenvalue:e} of $D^2 L$ |', eigenvector_xyz_v))

        for name, values in vector_field_v:
            vtk_array = vtk.util.numpy_support.numpy_to_vtk(values, deep=True)
            vtk_array.SetName(name)
            vtk_array.SetNumberOfComponents(3)
            grid.GetPointData().AddArray(vtk_array)

        # Set Negative Gradient of Lagrangian as active (for glyphs)
        grid.GetPointData().SetActiveVectors('Negative Gradient of Lagrangian')
        
        # Write file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
        print(f'Successfully wrote HyperelasticBody to file {filename}')

    @staticmethod
    def test():
        F = FEFunctionSpace2D(
            domain_corner_v=array([[0.0, 0.0], [2.0, 3.0]]),
            vertex_count_v=array([4, 6]),
        )
        mu = DiscreteMeasure.trapezoid_rule_on_rectangle(F.domain_corner_v, F.vertex_count_v * 8)
        M = HyperelasticMaterial(
            alpha=1.0,
            uniform_mass_density=1.0,
        )
        S = SpatialManifold(
            surface_name='TestSurface',
            spatial_metric=lambda Y: np.eye(2, dtype=np.float64),
            spatial_embedding_z=lambda Y: 0.0,
        )
        mass_specific_potential = lambda Y: 0.0
        B = HyperelasticBody.identity_embedding(F, mu, M, S, mass_specific_potential)

        # Sample the body configuration at the mesh points, and verify that body_configuration actually represents the identity map.
        for sample_x in np.linspace(F.domain_corner_v[0,0], F.domain_corner_v[1,0], F.vertex_count_v[0]*10, endpoint=True):
            for sample_y in np.linspace(F.domain_corner_v[0,1], F.domain_corner_v[1,1], F.vertex_count_v[1]*10, endpoint=True):
                body_point_v = array([sample_x, sample_y])
                space_point_v, Dspace_point_v = B.phi_and_Dphi(body_point_v)
                # Verify pointwise identity.
                assert np.allclose(space_point_v, body_point_v)
                # Verify that the derivative is the identity matrix.
                assert np.allclose(Dspace_point_v, array([[1.0, 0.0], [0.0, 1.0]]))

        L, DL, D2L = nd.hessian(lambda cfg: B.with_cfg(cfg).lagrangian(), B.cfg)
        # lagrangian = B.lagrangian()
        # print(f'lagrangian = {lagrangian}')
        assert np.allclose(L, 0.0)
        assert np.allclose(DL, 0.0)
        # TODO: Assert D2L is positive definite, modulo symmetries (in this case, translations and rotations).

        print('jello.body.HyperelasticBody.test passed')

def test():
    HyperelasticBody.test()

if __name__ == '__main__':
    test()
