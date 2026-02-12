import numpy as np
# import num_dual as nd
import pyvista as pv
# import vtk
# import vtk.util.numpy_support

from .body import HyperelasticBody
from .phi2d import phi
from numpy import array, ndarray
from typing import Any, Callable, Optional

def grid_from_body(
    b: HyperelasticBody,
    mesh_vertex_count_v: ndarray,
    *,
    embedding_z: Optional[Callable[[ndarray], Any]] = None
) -> pv.StructuredGrid:
    assert mesh_vertex_count_v.shape == (2,)
    assert mesh_vertex_count_v.dtype == int
    assert np.all(mesh_vertex_count_v >= 2)

    body_x = np.linspace(b.function_space.domain_corner_v[0,0], b.function_space.domain_corner_v[1,0], mesh_vertex_count_v[0], endpoint=True)
    body_y = np.linspace(b.function_space.domain_corner_v[0,1], b.function_space.domain_corner_v[1,1], mesh_vertex_count_v[1], endpoint=True)
    body_mesh_x, body_mesh_y = np.meshgrid(body_x, body_y, indexing='ij')
    body_point_v = np.stack([body_mesh_x, body_mesh_y], axis=2)

    # Sample the body configuration at the mesh points.
    space_point_v = np.apply_along_axis(lambda body_point: b.phi(body_point), 2, body_point_v)
    if embedding_z is not None:
        space_point_z = np.apply_along_axis(embedding_z, 2, space_point_v)
    else:
        space_point_z = np.zeros_like(space_point_v[:,:,0])

    grid = pv.StructuredGrid(space_point_v[:,:,0], space_point_v[:,:,1], space_point_z)
    # body_point_v is stored reshaped as a list of vectors, one for each point.  The mesh structure
    # is not represented explicitly in this ndarray, but it's easier to work with in this form.
    # NOTE: This shows up in grid.active_scalars, but it should be in active_vectors if anything.
    grid.point_data['body_point_v'] = body_point_v.reshape(-1,2)

    print(grid)

    return grid

def update_grid_from_body(grid: pv.StructuredGrid, b: HyperelasticBody, *, embedding_z: Optional[Callable[[ndarray], Any]] = None):
    body_point_v = grid.point_data['body_point_v']
    assert len(body_point_v.shape) == 2
    assert body_point_v.shape[1] == 2

    # Sample the body configuration at the mesh points.
    space_point_v = np.apply_along_axis(lambda body_point: b.phi(body_point), 1, body_point_v)
    grid.points[:,0:2] = space_point_v
    if embedding_z is not None:
        grid.points[:,2] = np.apply_along_axis(embedding_z, 1, space_point_v)

class HyperelasticBodyPlotter:
    def __init__(self, b: HyperelasticBody, *, mesh_vertex_count_v: ndarray = array([16, 16])):
        self.b = b
        self.arrow_scale_factor = 1.0e-2

        grid_stored_energy_density_y1 = grid_from_body(b, mesh_vertex_count_v)
        grid_stored_energy_density_y2 = grid_stored_energy_density_y1.copy()
        grid_stored_energy_density = grid_stored_energy_density_y1.copy()
        grid_cauchy_tensor_invariant_tr = grid_stored_energy_density_y1.copy()
        grid_cauchy_tensor_invariant_det = grid_stored_energy_density_y1.copy()
        grid_lagrangian_density = grid_stored_energy_density_y1.copy()
        grid_potential_energy_density = grid_stored_energy_density_y1.copy()
        grid_neg_eigenvalue_field = grid_stored_energy_density_y1.copy()
        grid_embedded = grid_from_body(b, mesh_vertex_count_v, embedding_z=b.spatial_manifold.spatial_embedding_z)

        grid_stored_energy_density_y1.point_data['Stored Energy Density y1'] = np.zeros(grid_stored_energy_density_y1.points.shape[0], dtype=np.float64)
        grid_stored_energy_density_y2.point_data['Stored Energy Density y2'] = np.zeros(grid_stored_energy_density_y2.points.shape[0], dtype=np.float64)
        grid_stored_energy_density.point_data['Stored Energy Density'] = np.zeros(grid_stored_energy_density.points.shape[0], dtype=np.float64)
        grid_cauchy_tensor_invariant_tr.point_data['Cauchy tensor invariant tr (log 2 scale)'] = np.zeros(grid_cauchy_tensor_invariant_tr.points.shape[0], dtype=np.float64)
        grid_cauchy_tensor_invariant_det.point_data['Cauchy tensor invariant det (log 2 scale)'] = np.zeros(grid_cauchy_tensor_invariant_det.points.shape[0], dtype=np.float64)
        grid_lagrangian_density.point_data['Lagrangian Density'] = np.zeros(grid_lagrangian_density.points.shape[0], dtype=np.float64)
        grid_potential_energy_density.point_data['Potential Energy Density'] = np.zeros(grid_potential_energy_density.points.shape[0], dtype=np.float64)

        # shape is (rows, columns).
        plotter = pv.Plotter(shape=(2, 3))

        plotter.subplot(0, 0)
        plotter.add_mesh(grid_stored_energy_density_y1, scalars='Stored Energy Density y1', show_edges=True, edge_color='black', line_width=2.0)

        self.grid_stored_energy_y1_neg_grad_polydata = pv.PolyData(grid_stored_energy_density_y1.points)
        self.grid_stored_energy_y1_neg_grad_polydata['vectors'] = np.zeros_like(grid_stored_energy_density_y1.points)
        grid_stored_energy_y1_neg_grad_arrow_glyph = self.grid_stored_energy_y1_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_stored_energy_y1_neg_grad_arrow_actor = plotter.add_mesh(grid_stored_energy_y1_neg_grad_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        plotter.subplot(0, 1)
        plotter.add_mesh(grid_stored_energy_density_y2, scalars='Stored Energy Density y2', show_edges=True, edge_color='black', line_width=2.0)

        self.grid_stored_energy_y2_neg_grad_polydata = pv.PolyData(grid_stored_energy_density_y2.points)
        self.grid_stored_energy_y2_neg_grad_polydata['vectors'] = np.zeros_like(grid_stored_energy_density_y2.points)
        grid_stored_energy_y2_neg_grad_arrow_glyph = self.grid_stored_energy_y2_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_stored_energy_y2_neg_grad_arrow_actor = plotter.add_mesh(grid_stored_energy_y2_neg_grad_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        plotter.subplot(0, 2)
        plotter.add_mesh(grid_stored_energy_density, scalars='Stored Energy Density', show_edges=True, edge_color='black', line_width=2.0)

        self.grid_stored_energy_neg_grad_polydata = pv.PolyData(grid_stored_energy_density.points)
        self.grid_stored_energy_neg_grad_polydata['vectors'] = np.zeros_like(grid_stored_energy_density.points)
        grid_stored_energy_neg_grad_arrow_glyph = self.grid_stored_energy_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_stored_energy_neg_grad_arrow_actor = plotter.add_mesh(grid_stored_energy_neg_grad_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        plotter.subplot(1, 0)
        plotter.add_mesh(grid_lagrangian_density, scalars='Lagrangian Density', show_edges=True, edge_color='black', line_width=2.0)

        self.grid_lagrangian_neg_grad_polydata = pv.PolyData(grid_stored_energy_density_y1.points)
        self.grid_lagrangian_neg_grad_polydata['vectors'] = np.zeros_like(grid_stored_energy_density_y1.points)
        grid_lagrangian_neg_grad_arrow_glyph = self.grid_lagrangian_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_lagrangian_neg_grad_arrow_actor = plotter.add_mesh(grid_lagrangian_neg_grad_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        plotter.subplot(1, 1)
        plotter.add_mesh(grid_potential_energy_density, scalars='Potential Energy Density', show_edges=True, edge_color='black', line_width=2.0)

        self.grid_potential_energy_neg_grad_polydata = pv.PolyData(grid_stored_energy_density_y1.points)
        self.grid_potential_energy_neg_grad_polydata['vectors'] = np.zeros_like(grid_stored_energy_density_y1.points)
        grid_potential_energy_neg_grad_arrow_glyph = self.grid_potential_energy_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_potential_energy_neg_grad_arrow_actor = plotter.add_mesh(grid_potential_energy_neg_grad_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        plotter.subplot(1, 2)
        plotter.add_mesh(grid_neg_eigenvalue_field, scalars=None, show_edges=True, edge_color='black', line_width=2.0)

        self.grid_neg_eigenvalue_field_polydata = pv.PolyData(grid_stored_energy_density_y1.points)
        self.grid_neg_eigenvalue_field_polydata['vectors'] = np.zeros_like(grid_stored_energy_density_y1.points)
        grid_neg_eigenvalue_field_arrow_glyph = self.grid_neg_eigenvalue_field_polydata.glyph(orient='vectors', scale='vectors', factor=self.arrow_scale_factor)
        self.grid_neg_eigenvalue_field_arrow_actor = plotter.add_mesh(grid_neg_eigenvalue_field_arrow_glyph, color='red')

        plotter.enable_2d_style()
        plotter.enable_parallel_projection()
        # NOTE: This is expensive.  Other anti-aliasing options are available.
        # plotter.enable_anti_aliasing('ssaa')
        plotter.show_axes()
        plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        plotter.view_xy()

        if False:
            plotter.subplot(2, 0)
            plotter.add_mesh(grid_cauchy_tensor_invariant_tr, scalars='Cauchy tensor invariant tr (log 2 scale)', show_edges=True, edge_color='black', line_width=2.0)
            plotter.enable_2d_style()
            plotter.enable_parallel_projection()
            # NOTE: This is expensive.  Other anti-aliasing options are available.
            # plotter.enable_anti_aliasing('ssaa')
            plotter.show_axes()
            plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
            plotter.view_xy()

            plotter.subplot(2, 1)
            plotter.add_mesh(grid_cauchy_tensor_invariant_det, scalars='Cauchy tensor invariant det (log 2 scale)', show_edges=True, edge_color='black', line_width=2.0)
            plotter.enable_2d_style()
            plotter.enable_parallel_projection()
            # NOTE: This is expensive.  Other anti-aliasing options are available.
            # plotter.enable_anti_aliasing('ssaa')
            plotter.show_axes()
            plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
            plotter.view_xy()

        # # NOTE: Apparently the interaction style is shared between all subplots, and can't be decoupled.
        # # So this solution is a hack.

        # plotter.subplot(1, 2)
        # plotter.add_mesh(grid_embedded, scalars=None, show_edges=True, edge_color='black', line_width=2.0)
        # # plotter.disable_2d_style()
        # plotter.enable_terrain_style()
        # plotter.disable_parallel_projection()
        # # NOTE: This is expensive.  Other anti-aliasing options are available.
        # # plotter.enable_anti_aliasing('ssaa')
        # plotter.show_axes()
        # # plotter.show_bounds(use_2d=True, location='origin', all_edges=True, padding=0.1, use_3d_text=False, ticks='outside', minor_ticks=True)
        # plotter.view_isometric()

        plotter.link_views()

        self.grid_stored_energy_density_y1 = grid_stored_energy_density_y1
        self.grid_stored_energy_density_y2 = grid_stored_energy_density_y2
        self.grid_stored_energy_density = grid_stored_energy_density
        self.grid_cauchy_tensor_invariant_tr = grid_cauchy_tensor_invariant_tr
        self.grid_cauchy_tensor_invariant_det = grid_cauchy_tensor_invariant_det
        self.grid_potential_energy_density = grid_potential_energy_density
        self.grid_lagrangian_density = grid_lagrangian_density
        self.grid_neg_eigenvalue_field = grid_neg_eigenvalue_field
        self.grid_embedded = grid_embedded

        self.plotter = plotter

    def show_nonblocking(self):
        self.plotter.show(interactive=True, interactive_update=True)

    def show_blocking(self):
        self.plotter.show(interactive=True, interactive_update=False)

    def update(self, *, neg_eigenvalue_field: Optional[ndarray] = None):
        update_grid_from_body(self.grid_stored_energy_density_y1, self.b)
        update_grid_from_body(self.grid_stored_energy_density_y2, self.b)
        update_grid_from_body(self.grid_stored_energy_density, self.b)
        update_grid_from_body(self.grid_cauchy_tensor_invariant_tr, self.b)
        update_grid_from_body(self.grid_cauchy_tensor_invariant_det, self.b)
        update_grid_from_body(self.grid_lagrangian_density, self.b)
        update_grid_from_body(self.grid_potential_energy_density, self.b)
        update_grid_from_body(self.grid_neg_eigenvalue_field, self.b)
        update_grid_from_body(self.grid_embedded, self.b, embedding_z=self.b.spatial_manifold.spatial_embedding_z)

        # body_point_v should be the same for all grids in this case.
        body_point_v = self.grid_stored_energy_density.point_data['body_point_v']
        # space_point_v = np.apply_along_axis(lambda body_point: phi(body_configuration.reshape(self.b.function_space.vector_function_shape), self.h.domain_corner_v, body_point), 1, body_point_v)

        # This is used to avoid log10(0).
        log_epsilon = 1.0e-10
        # self.grid_stored_energy_density_y1.point_data['Stored Energy Density y1'] = np.log10(np.apply_along_axis(lambda body_point: self.b.stored_energy_density(body_point, y1_component=True, y2_component=False), 1, body_point_v) + log_epsilon)
        # self.grid_stored_energy_density_y2.point_data['Stored Energy Density y2'] = np.log10(np.apply_along_axis(lambda body_point: self.b.stored_energy_density(body_point, y1_component=False, y2_component=True), 1, body_point_v) + log_epsilon)
        self.grid_stored_energy_density_y1.point_data['Stored Energy Density y1'] = np.apply_along_axis(lambda body_point: self.b.stored_energy_density(body_point, y1_component=True, y2_component=False), 1, body_point_v)
        self.grid_stored_energy_density_y2.point_data['Stored Energy Density y2'] = np.apply_along_axis(lambda body_point: self.b.stored_energy_density(body_point, y1_component=False, y2_component=True), 1, body_point_v)
        self.grid_stored_energy_density.point_data['Stored Energy Density'] = np.apply_along_axis(lambda body_point: self.b.stored_energy_density(body_point), 1, body_point_v)
        self.grid_cauchy_tensor_invariant_tr.point_data['Cauchy tensor invariant tr (log 2 scale)'] = np.log2(np.apply_along_axis(lambda body_point: self.b.cauchy_tensor_invariants(body_point)[0], 1, body_point_v) + log_epsilon)
        self.grid_cauchy_tensor_invariant_det.point_data['Cauchy tensor invariant det (log 2 scale)'] = np.log2(np.apply_along_axis(lambda body_point: self.b.cauchy_tensor_invariants(body_point)[1], 1, body_point_v) + log_epsilon)
        self.grid_lagrangian_density.point_data['Lagrangian Density'] = np.apply_along_axis(lambda body_point: self.b.lagrangian_density(body_point), 1, body_point_v)
        self.grid_potential_energy_density.point_data['Potential Energy Density'] = np.apply_along_axis(lambda body_point: self.b.potential_energy_density(body_point), 1, body_point_v)

        # Compute the embedding.
        # self.grid_embedded.points[:,0:2] = space_point_v
        # self.grid_embedded.points[:,2] = np.apply_along_axis(self.h.spatial_embedding_z, 1, space_point_v)

        stored_energy_density_y1_range = self.grid_stored_energy_density_y1.point_data['Stored Energy Density y1'].min(), self.grid_stored_energy_density_y1.point_data['Stored Energy Density y1'].max()
        stored_energy_density_y2_range = self.grid_stored_energy_density_y2.point_data['Stored Energy Density y2'].min(), self.grid_stored_energy_density_y2.point_data['Stored Energy Density y2'].max()
        stored_energy_density_range = self.grid_stored_energy_density.point_data['Stored Energy Density'].min(), self.grid_stored_energy_density.point_data['Stored Energy Density'].max()
        cauchy_tensor_invariant_tr_range = self.grid_cauchy_tensor_invariant_tr.point_data['Cauchy tensor invariant tr (log 2 scale)'].min(), self.grid_cauchy_tensor_invariant_tr.point_data['Cauchy tensor invariant tr (log 2 scale)'].max()
        cauchy_tensor_invariant_det_range = self.grid_cauchy_tensor_invariant_det.point_data['Cauchy tensor invariant det (log 2 scale)'].min(), self.grid_cauchy_tensor_invariant_det.point_data['Cauchy tensor invariant det (log 2 scale)'].max()
        lagrangian_density_range = self.grid_lagrangian_density.point_data['Lagrangian Density'].min(), self.grid_lagrangian_density.point_data['Lagrangian Density'].max()
        potential_energy_density_range = self.grid_potential_energy_density.point_data['Potential Energy Density'].min(), self.grid_potential_energy_density.point_data['Potential Energy Density'].max()

        # # This symmetrizes the range around 0, so that colormaps are more intutive.
        # # NOTE: This doesn't work very well, it tends to decrease the visual detail.
        # def symmetrized_range(range: Tuple[float, float]) -> Tuple[float, float]:
        #     return (min(range[0], -range[1]), max(-range[0], range[1]))

        # stored_energy_density_y1_range = symmetrized_range(stored_energy_density_y1_range)
        # stored_energy_density_y2_range = symmetrized_range(stored_energy_density_y2_range)
        # stored_energy_density_range = symmetrized_range(stored_energy_density_range)
        # cauchy_tensor_invariant_tr_range = symmetrized_range(cauchy_tensor_invariant_tr_range)
        # cauchy_tensor_invariant_det_range = symmetrized_range(cauchy_tensor_invariant_det_range)
        # lagrangian_density_range = symmetrized_range(lagrangian_density_range)

        # TODO: Consider using a CDF-based colormap, so that there is more visual detail.

        # exp_range = (-5.0, 3.0)
        # tick_value_v = np.linspace(exp_range[0], exp_range[1], 7, endpoint=True)
        # tick_label_v = [f'10^{tick_value}' for tick_value in tick_value_v]

        self.plotter.update_scalar_bar_range(stored_energy_density_y1_range, 'Stored Energy Density y1')
        self.plotter.update_scalar_bar_range(stored_energy_density_y2_range, 'Stored Energy Density y2')
        self.plotter.update_scalar_bar_range(stored_energy_density_range, 'Stored Energy Density')
        # self.plotter.update_scalar_bar_range(cauchy_tensor_invariant_tr_range, 'Cauchy tensor invariant tr (log 2 scale)')
        # self.plotter.update_scalar_bar_range(cauchy_tensor_invariant_det_range, 'Cauchy tensor invariant det (log 2 scale)')
        self.plotter.update_scalar_bar_range(lagrangian_density_range, 'Lagrangian Density')
        self.plotter.update_scalar_bar_range(potential_energy_density_range, 'Potential Energy Density')
        # self.plotter.update_scalar_bar_range((-10.0, 10.0), 'Potential Energy Density')
        # self.plotter.update_scalar_bar_range((-10.0, 10.0), 'Lagrangian Density')

        # self.plotter.update_scalar_bar_range(stored_energy_density_range, 'Stored Energy Density')
        # self.plotter.update_scalar_bar_range(potential_energy_density_range, 'Potential Energy Density')
        # self.plotter.update_scalar_bar_range(lagrangian_density_range, 'Lagrangian Density')

        # Compute the gradients of each component of the Lagrangian.
        def stored_energy_y1(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            b_body_cfg = self.b.with_cfg(body_cfg)
            stored_energy_y1_density = lambda X: b_body_cfg.stored_energy_density(X, y1_component=True, y2_component=False)
            return self.b.measure.integrate(stored_energy_y1_density)
        
        def stored_energy_y2(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            b_body_cfg = self.b.with_cfg(body_cfg)
            stored_energy_y2_density = lambda X: b_body_cfg.stored_energy_density(X, y1_component=False, y2_component=True)
            return self.b.measure.integrate(stored_energy_y2_density)
        
        def potential_energy(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            b_body_cfg = self.b.with_cfg(body_cfg)
            potential_energy_density = lambda X: b_body_cfg.potential_energy_density(X)
            return self.b.measure.integrate(potential_energy_density)
        
        def lagrangian(body_cfg: ndarray) -> Any:
            if not isinstance(body_cfg, ndarray):
                body_cfg = array(body_cfg)
            b_body_cfg = self.b.with_cfg(body_cfg)
            lagrangian_density = lambda X: b_body_cfg.lagrangian_density(X)
            return self.b.measure.integrate(lagrangian_density)
        
        neg_grad_stored_energy_y1 = -self.b.function_space.vector_field_gradient(stored_energy_y1, self.b.cfg, self.b.measure, self.b.spatial_manifold)
        neg_grad_stored_energy_y2 = -self.b.function_space.vector_field_gradient(stored_energy_y2, self.b.cfg, self.b.measure, self.b.spatial_manifold)
        neg_grad_potential_energy = -self.b.function_space.vector_field_gradient(potential_energy, self.b.cfg, self.b.measure, self.b.spatial_manifold)
        neg_grad_lagrangian = -self.b.function_space.vector_field_gradient(lagrangian, self.b.cfg, self.b.measure, self.b.spatial_manifold)

        # Need to sample the negative gradients at the grid body points.
        body_point_v = self.grid_stored_energy_density.point_data['body_point_v']

        def determine_arrow_scale_factor(max_arrow_length: float) -> float:
            if max_arrow_length <= 1.0e-8:
                # Below a certain threshold, use a fixed scale factor of 1.
                return 1.0
            else:
                return 0.25 / max_arrow_length

        # Update the stored energy y1 negative gradient arrows.
        self.plotter.subplot(0, 0)
        self.grid_stored_energy_y1_neg_grad_polydata.points = self.grid_stored_energy_density_y1.points
        self.grid_stored_energy_y1_neg_grad_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi(neg_grad_stored_energy_y1.reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
        max_arrow_length = np.max(np.linalg.norm(self.grid_stored_energy_y1_neg_grad_polydata['vectors'][:,0:2], axis=1))
        grid_stored_energy_y1_neg_grad_arrow_glyph = self.grid_stored_energy_y1_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(max_arrow_length))
        print(f'stored_energy_y1_neg_grad max arrow length: {max_arrow_length}')
        # Update the arrow actor's mesh in place
        self.grid_stored_energy_y1_neg_grad_arrow_actor.GetMapper().SetInputData(grid_stored_energy_y1_neg_grad_arrow_glyph)

        # Update the stored energy y2 negative gradient arrows.
        self.plotter.subplot(0, 1)
        self.grid_stored_energy_y2_neg_grad_polydata.points = self.grid_stored_energy_density_y2.points
        self.grid_stored_energy_y2_neg_grad_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi(neg_grad_stored_energy_y2.reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
        max_arrow_length = np.max(np.linalg.norm(self.grid_stored_energy_y2_neg_grad_polydata['vectors'][:,0:2], axis=1))
        grid_stored_energy_y2_neg_grad_arrow_glyph = self.grid_stored_energy_y2_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(max_arrow_length))
        print(f'stored_energy_y2_neg_grad max arrow length: {max_arrow_length}')
        # Update the arrow actor's mesh in place
        self.grid_stored_energy_y2_neg_grad_arrow_actor.GetMapper().SetInputData(grid_stored_energy_y2_neg_grad_arrow_glyph)

        # Make stored energy, potential energy, and (maybe) Lagrangian arrows use the same scale factor.

        self.grid_stored_energy_neg_grad_polydata.points = self.grid_stored_energy_density.points
        self.grid_stored_energy_neg_grad_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi((neg_grad_stored_energy_y1 + neg_grad_stored_energy_y2).reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
        stored_energy_neg_grad_max_arrow_length = np.max(np.linalg.norm(self.grid_stored_energy_neg_grad_polydata['vectors'][:,0:2], axis=1))
        print(f'stored_energy_neg_grad max arrow length: {stored_energy_neg_grad_max_arrow_length}')

        self.grid_lagrangian_neg_grad_polydata.points = self.grid_lagrangian_density.points
        self.grid_lagrangian_neg_grad_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi(neg_grad_lagrangian.reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
        lagrangian_neg_grad_max_arrow_length = np.max(np.linalg.norm(self.grid_lagrangian_neg_grad_polydata['vectors'][:,0:2], axis=1))
        print(f'lagrangian_neg_grad max arrow length: {lagrangian_neg_grad_max_arrow_length}')

        self.grid_potential_energy_neg_grad_polydata.points = self.grid_potential_energy_density.points
        self.grid_potential_energy_neg_grad_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi(neg_grad_potential_energy.reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
        potential_energy_neg_grad_max_arrow_length = np.max(np.linalg.norm(self.grid_potential_energy_neg_grad_polydata['vectors'][:,0:2], axis=1))
        print(f'potential_energy_neg_grad max arrow length: {potential_energy_neg_grad_max_arrow_length}')

        common_max_arrow_length = max(stored_energy_neg_grad_max_arrow_length, lagrangian_neg_grad_max_arrow_length, potential_energy_neg_grad_max_arrow_length)

        # Update the stored energy negative gradient arrows.
        self.plotter.subplot(0, 2)
        grid_stored_energy_neg_grad_arrow_glyph = self.grid_stored_energy_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(common_max_arrow_length))
        # Update the arrow actor's mesh in place
        self.grid_stored_energy_neg_grad_arrow_actor.GetMapper().SetInputData(grid_stored_energy_neg_grad_arrow_glyph)

        # Update the lagrangian negative gradient arrows.
        self.plotter.subplot(1, 0)

        grid_lagrangian_neg_grad_arrow_glyph = self.grid_lagrangian_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(common_max_arrow_length))
        # Update the arrow actor's mesh in place
        self.grid_lagrangian_neg_grad_arrow_actor.GetMapper().SetInputData(grid_lagrangian_neg_grad_arrow_glyph)

        # Update the potential energy negative gradient arrows.
        self.plotter.subplot(1, 1)
        grid_potential_energy_neg_grad_arrow_glyph = self.grid_potential_energy_neg_grad_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(common_max_arrow_length))
        # Update the arrow actor's mesh in place
        self.grid_potential_energy_neg_grad_arrow_actor.GetMapper().SetInputData(grid_potential_energy_neg_grad_arrow_glyph)

        # Update the Lagrangian's hessian's (largest-magnitude) negative eigenvalue arrows.
        if neg_eigenvalue_field is not None:
            self.plotter.subplot(1, 2)
            self.grid_neg_eigenvalue_field_polydata.points = self.grid_neg_eigenvalue_field.points
            self.grid_neg_eigenvalue_field_polydata['vectors'][:,0:2] = np.apply_along_axis(lambda body_point: phi(neg_eigenvalue_field.reshape(self.b.function_space.vector_function_shape), self.b.function_space.domain_corner_v, body_point), 1, body_point_v)
            max_arrow_length = np.max(np.linalg.norm(self.grid_neg_eigenvalue_field_polydata['vectors'][:,0:2], axis=1))
            grid_neg_eigenvalue_field_arrow_glyph = self.grid_neg_eigenvalue_field_polydata.glyph(orient='vectors', scale='vectors', factor=determine_arrow_scale_factor(max_arrow_length))
            print(f'neg_eigenvalue_field max arrow length: {max_arrow_length}')
            if self.grid_neg_eigenvalue_field_arrow_actor is None:
                self.grid_neg_eigenvalue_field_arrow_actor = self.plotter.add_mesh(grid_neg_eigenvalue_field_arrow_glyph, color='red')
            else:
                # Update the arrow actor's mesh in place
                self.grid_neg_eigenvalue_field_arrow_actor.GetMapper().SetInputData(grid_neg_eigenvalue_field_arrow_glyph)
        else:
            print('neg_eigenvalue_field is None -- removing negative eigenvalue field arrows')
            if self.grid_neg_eigenvalue_field_arrow_actor is not None:
                self.plotter.remove_actor(self.grid_neg_eigenvalue_field_arrow_actor)
                self.grid_neg_eigenvalue_field_arrow_actor = None
                self.grid_neg_eigenvalue_field_arrow_glyph = None

        self.plotter.update()

def plot_stuff():
    import time

    from .fem2d import FEFunctionSpace2D
    from .material import HyperelasticMaterial
    from .spatial_manifold import SpatialManifold
    from .body import HyperelasticBody
    from numpy import array, ndarray
    from typing import Any, Callable, Optional

    f = FEFunctionSpace2D(
        domain_corner_v=array([[-1.0, -1.0], [1.0, 1.0]]),
        vertex_count_v=array([4, 4]),
    )
    M = HyperelasticMaterial(
        alpha=1.0,
        uniform_mass_density=1.0,
    )
    S = SpatialManifold(
        spatial_metric=lambda Y: np.eye(2, dtype=np.float64),
        spatial_embedding_z=lambda Y: Y.dot(Y),
    )
    mass_specific_potential = lambda Y: Y.dot(Y)
    b = HyperelasticBody.identity_embedding(f, M, S, mass_specific_potential)

    body_cfg_base = b.cfg.copy()
    body_cfg_axis0 = 0.1 * np.random.randn(*b.cfg.shape)
    body_cfg_axis1 = 0.1 * np.random.randn(*b.cfg.shape)

    hbp = HyperelasticBodyPlotter(b)
    hbp.show_nonblocking()

    # Animate
    while True:
        try:
            t = time.time()

            # Simple arbitrary motion.  Use an irrational frequency ratio to avoid periodic behavior.
            b.cfg[...] = body_cfg_base + np.cos(t) * body_cfg_axis0 + np.sin(0.5 * np.pi * t) * body_cfg_axis1

            hbp.update()

            time.sleep(0.1)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    plot_stuff()
