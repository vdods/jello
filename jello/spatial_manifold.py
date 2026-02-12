import numpy as np
import num_dual as nd
import vtk
import vtk.util.numpy_support

from numpy import array, ndarray
from typing import Any, Callable

class SpatialManifold:
    def __init__(
        self,
        *,
        surface_name: str,
        spatial_embedding_z: Callable[[ndarray], Any],
        spatial_metric: Callable[[ndarray], ndarray],
    ):
        self.surface_name = surface_name
        self.spatial_embedding_z = spatial_embedding_z
        self.spatial_metric = spatial_metric

    @staticmethod
    def flamms_paraboloid(*, schwarzschild_radius: float) -> 'SpatialManifold':
        """
        Create the Flamms paraboloid surface, which is a representation of the spatial curvature of a slice of
        the Schwarzschild spacetime.
        """
        assert schwarzschild_radius > 0.0

        def z(Y: ndarray) -> Any:
            r = Y.dot(Y)**0.5
            return 2 * (schwarzschild_radius * (r - schwarzschild_radius))**0.5

        def spatial_metric(Y: ndarray) -> ndarray:
            """
            The metric comes from the Schwarzschild metric, but the embedding into Euclidean R^3 is chosen such that it is an isometric embedding.
            """
            retval = np.outer(Y, Y)
            r_squared = Y.dot(Y)
            r = r_squared**0.5
            retval *= (schwarzschild_radius / ((r - schwarzschild_radius) * r_squared))
            retval[0,0] += 1
            retval[1,1] += 1
            return retval

        return SpatialManifold(
            surface_name='flamms-paraboloid',
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def funnel() -> 'SpatialManifold':
        def z(Y: ndarray) -> Any:
            return -1.0 / np.linalg.norm(Y)

        def spatial_metric(Y: ndarray) -> ndarray:
            """Metric induced on the z = -1/r funnel surface by its embedding into Euclidean R^3."""
            retval = np.outer(Y, Y)
            retval /= Y.dot(Y)**3
            retval[0,0] += 1
            retval[1,1] += 1
            return retval

        return SpatialManifold(
            surface_name='funnel',
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def gaussian(*, orientation: float) -> 'SpatialManifold':
        """
        Gaussian distribution surface: z = orientation * exp(-r^2 / 2).
        Orientation should be 1.0 or -1.0, indicating which way the Gaussian
        distribution is oriented (1.0 for conventional definition, -1.0 for upside-down).
        """
        assert orientation == 1.0 or orientation == -1.0

        def z(Y: ndarray) -> Any:
            return orientation * np.exp(-Y.dot(Y) / 2)

        def spatial_metric(Y: ndarray) -> ndarray:
            """Metric induced on the z = exp(-r^2 / 2) Gaussian distribution surface by its embedding into Euclidean R^3."""
            retval = np.outer(Y, Y)
            r_squared = Y.dot(Y)
            retval *= np.exp(-r_squared)
            retval[0,0] += 1
            retval[1,1] += 1
            return retval

        if orientation == 1.0:
            surface_name = 'gaussian'
        else:
            surface_name = 'gaussian-upside-down'

        return SpatialManifold(
            surface_name=surface_name,
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def paraboloid(*, orientation: float) -> 'SpatialManifold':
        """
        Orientation should be 1.0 or -1.0, indicating which way the paraboloid is oriented (1.0 for up, -1.0 for down).
        """

        assert orientation == 1.0 or orientation == -1.0

        def z(Y: ndarray) -> Any:
            return orientation * Y.dot(Y) / 2

        def spatial_metric(Y: ndarray) -> ndarray:
            """Metric induced on the z = x^2 + y^2/2 paraboloid surface by its embedding into Euclidean R^3."""
            retval = np.outer(Y, Y)
            retval[0,0] += 1
            retval[1,1] += 1
            return retval
        
        if orientation == 1.0:
            surface_name = 'paraboloid-upward'
        else:
            surface_name = 'paraboloid-downward'

        return SpatialManifold(
            surface_name=surface_name,
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def parabolic_trough() -> 'SpatialManifold':
        """
        Parabolic trough surface: z = x^2 / 2.
        """

        def z(Y: ndarray) -> Any:
            return Y[0]**2 / 2

        def spatial_metric(Y: ndarray) -> ndarray:
            """Metric induced on the z = x^2 / 2 parabolic trough surface by its embedding into Euclidean R^3."""
            retval = np.eye(2, dtype=Y.dtype)
            retval[0,0] += 1
            return retval
        
        surface_name = 'parabolic-trough'

        return SpatialManifold(
            surface_name=surface_name,
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def spherical_cup(radius: float) -> 'SpatialManifold':
        def z(Y: ndarray) -> Any:
            return -(radius**2 - Y.dot(Y))**0.5

        def spatial_metric(Y: ndarray) -> ndarray:
            """Metric induced on the z = x^2 + y^2/2 paraboloid surface by its embedding into Euclidean R^3."""
            retval = np.outer(Y, Y)
            retval /= (radius**2 - Y.dot(Y))
            retval[0,0] += 1
            retval[1,1] += 1
            return retval

        return SpatialManifold(
            surface_name='spherical-cup',
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    @staticmethod
    def flat_plane() -> 'SpatialManifold':
        """Euclidean 2-space."""
        def z(Y: ndarray) -> Any:
            # TEMP HACK to get a zero of the correct type
            return Y[0]*0.0

        def spatial_metric(Y: ndarray) -> ndarray:
            return np.eye(2, dtype=Y.dtype)

        return SpatialManifold(
            surface_name='flat-plane',
            spatial_metric=spatial_metric,
            spatial_embedding_z=z,
        )

    def tangency_residual_of_vector(self, space_point_xyz: ndarray, space_vector_xyz: ndarray) -> Any:
        def F(xyz: ndarray) -> Any:
            assert np.shape(xyz) == (3,)
            x, y, z = xyz
            return self.spatial_embedding_z(np.array([x, y])) - z
        
        DF = array(nd.gradient(F, space_point_xyz)[1])
        return DF.dot(space_vector_xyz)

    def write_surface_vtp(self, filename: str, *, surface_name: str, inner_radius: float, outer_radius: float):
        # Currently this function is hardcoded to cylindrical coordinates.

        if surface_name == 'flamms-paraboloid':
            r_v = np.linspace(0.0, np.sqrt(outer_radius-inner_radius), 64, endpoint=True)**2 + inner_radius
        else:
            r_v = np.linspace(inner_radius, outer_radius, 64, endpoint=True)
        theta_v = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=True)
        r_mesh_v, theta_mesh_v = np.meshgrid(r_v, theta_v, indexing='ij')

        x_mesh_v = (r_mesh_v * np.cos(theta_mesh_v)).flatten()
        y_mesh_v = (r_mesh_v * np.sin(theta_mesh_v)).flatten()
        z_mesh_v = np.apply_along_axis(self.spatial_embedding_z, 1, np.column_stack([x_mesh_v, y_mesh_v]))

        point_v = np.column_stack([x_mesh_v, y_mesh_v, z_mesh_v])
        quad_v = []
        for i in range(len(r_v) - 1):
            for j in range(len(theta_v) - 1):
                ij = i * len(theta_v) + j
                iJ = i * len(theta_v) + j + 1
                Ij = (i + 1) * len(theta_v) + j
                IJ = (i + 1) * len(theta_v) + j + 1
                quad_v.append([ij, iJ, IJ, Ij])
        quad_v = np.array(quad_v, dtype=np.int32)

        # Set points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(point_v, deep=True))
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        
        # Set cells (quads as polygons)
        cells = vtk.vtkCellArray()
        for quad in quad_v:
            cell = vtk.vtkQuad()
            cell.GetPointIds().SetId(0, int(quad[0]))
            cell.GetPointIds().SetId(1, int(quad[1]))
            cell.GetPointIds().SetId(2, int(quad[2]))
            cell.GetPointIds().SetId(3, int(quad[3]))
            cells.InsertNextCell(cell)
        polydata.SetPolys(cells)
        
        # Write file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        print(f'Successfully wrote SpatialManifold to file {filename}')
