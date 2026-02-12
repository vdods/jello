from .material import HyperelasticMaterial
from .spatial_manifold import SpatialManifold
from numpy import ndarray
from typing import Any, Callable, Tuple

class ProblemContext:
    def __init__(
        self,
        *,
        problem_name: str,
        M: HyperelasticMaterial,
        S: SpatialManifold,
        gravitational_constant: float,
        mass_specific_potential: Callable[[ndarray], Any],
        surface_radius_range: Tuple[float, float],
        body_center_r_initial: float,
        body_angle_initial: float,
        mesh_refinement_level_count: int,
    ):
        self.problem_name = problem_name
        self.M = M
        self.S = S
        self.gravitational_constant = gravitational_constant
        self.mass_specific_potential = mass_specific_potential
        self.surface_radius_range = surface_radius_range
        self.body_center_r_initial = body_center_r_initial
        self.body_angle_initial = body_angle_initial
        self.mesh_refinement_level_count = mesh_refinement_level_count
    
    @staticmethod
    def create(problem_name: str) -> 'ProblemContext':
        if problem_name == 'flamms-paraboloid-with-gravity':
            S = SpatialManifold.flamms_paraboloid(schwarzschild_radius=1.0)
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=10000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(1.00001, 8.0),
                body_center_r_initial=3.0,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'funnel-with-gravity':
            S = SpatialManifold.funnel()
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(1.0/8.0, 8.0),
                body_center_r_initial=1.5,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'gaussian-with-gravity':
            assert False, 'not implemented yet; requires body manifold with non-Euclidean metric'
            # S = SpatialManifold.gaussian(orientation=1.0)
            # gravitational_constant = 1.0
            # mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            # return ProblemContext(
            #     problem_name=problem_name,
            #     M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
            #     S=S,
            #     gravitational_constant=gravitational_constant,
            #     mass_specific_potential=mass_specific_potential,
            #     surface_radius_range=(0.0, 8.0),
            #     body_center_r_initial=1.5,
            #     body_angle_initial=0.0,
            #     mesh_refinement_level_count=3,
            # )
        elif problem_name == 'gaussian-upside-down-with-gravity':
            S = SpatialManifold.gaussian(orientation=-1.0)
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(0.0, 8.0),
                body_center_r_initial=1.5,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'paraboloid-upward-with-gravity':
            S = SpatialManifold.paraboloid(orientation=1.0)
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(0.0, 2.0),
                body_center_r_initial=1.5,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'paraboloid-downward-with-gravity':
            assert False, 'not implemented yet; requires body manifold with non-Euclidean metric'
            # S = SpatialManifold.paraboloid(orientation=-1.0)
            # gravitational_constant = 1.0
            # mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            # return ProblemContext(
            #     problem_name=problem_name,
            #     M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
            #     S=S,
            #     gravitational_constant=gravitational_constant,
            #     mass_specific_potential=mass_specific_potential,
            #     surface_radius_range=(0.0, 8.0),
            #     body_center_r_initial=1.5,
            #     body_angle_initial=0.0,
            #     mesh_refinement_level_count=3,
            # )
        elif problem_name == 'parabolic-trough-with-gravity':
            S = SpatialManifold.parabolic_trough()
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(0.0, 2.0),
                body_center_r_initial=1.5,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'spherical-cup-with-gravity':
            S = SpatialManifold.spherical_cup(radius=5.0)
            gravitational_constant = 1.0
            mass_specific_potential = lambda Y: gravitational_constant * S.spatial_embedding_z(Y)
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=S,
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(0.0, 8.0),
                body_center_r_initial=1.5,
                body_angle_initial=0.0,
                mesh_refinement_level_count=3,
            )
        elif problem_name == 'flat-plane-no-gravity':
            S = SpatialManifold.flat_plane()
            gravitational_constant = 0.0
            # Dumb but effective way to get a zero of the correct type.
            mass_specific_potential = lambda Y: 0.0*Y[0]
            return ProblemContext(
                problem_name=problem_name,
                M=HyperelasticMaterial(alpha=5000.0, uniform_mass_density=1.0),
                S=SpatialManifold.flat_plane(),
                gravitational_constant=gravitational_constant,
                mass_specific_potential=mass_specific_potential,
                surface_radius_range=(0.0, 8.0),
                body_center_r_initial=0.0,
                body_angle_initial=0.0,
                mesh_refinement_level_count=0,
            )
        else:
            raise ValueError(f"Unknown problem name: {problem_name}")

    def write_surface_vtp(self):
        filename = f'surface.vtp'
        self.S.write_surface_vtp(filename, surface_name=self.S.surface_name, inner_radius=self.surface_radius_range[0], outer_radius=self.surface_radius_range[1])

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, *args, **kwargs):
        for writer in self.writers:
            writer.write(*args, **kwargs)

    def flush(self):
        for writer in self.writers:
            writer.flush()

    def close(self):
        for writer in self.writers:
            writer.close()

if __name__ == "__main__":
    import numpy as np
    import os
    import sys

    from .spatial_manifold import SpatialManifold
    from .solve import solve_static_problem

    def print_help_message():
        print('Usage: uv run python -m jello <problem-name> <output-directory>')
        print('This will create the output directory (if not already present), write the surface vtp file, and then')
        print('solve the FEM problem and write the solution data to vtu files.')
        print()
        print('Problem names, with surfaces given in cylindrical coordinates:')
        print('    flamms-paraboloid-with-gravity    : z = 2 * (r_s * (r - r_s))^(1/2) (r_s := Schwarzschild radius = 1)')
        print('    funnel-with-gravity               : z = -1/r')
        # print('    gaussian-with-gravity             : z = exp(-r^2 / 2)')
        # print('    gaussian-upside-down-with-gravity : z = -exp(-r^2 / 2)')
        # print('    parabolic-trough-with-gravity     : z = x^2 / 2')
        print('    paraboloid-upward-with-gravity    : z = x^2 + y^2 / 2')
        # print('    paraboloid-downward-with-gravity  : z = -x^2 - y^2 / 2')
        print('    spherical-cup-with-gravity        : z = -(r^2 - x^2 - y^2)^0.5')
        print('    flat-plane-no-gravity             : z = 0')
        sys.exit(1)

    if len(sys.argv) < 3:
        print(f'Error: Not enough arguments.')
        print()
        print_help_message()
        sys.exit(1)
    
    problem_name = sys.argv[1]
    output_directory = sys.argv[2]

    # Ensure the output directory doesn't already exist.
    # TODO: Make it so that it checks for existing files, and if there are existing vtu files, it reads
    # the latest one in and uses that as the initial body configuration.
    assert not os.path.exists(output_directory), f'output directory {output_directory} already exists -- specify a directory that doesn\'t exist instead.'

    # Create the output directory if it doesn't exist, then cd to it.
    os.makedirs(output_directory, exist_ok=True)
    os.chdir(output_directory)

    # Open a log file that console output will also be written to.
    # The use of buffering=1 means that a flush will happen after each line.
    log_file = open('log.txt', 'w', buffering=1)
    # A bit hacky, but it means I don't have to change all the print statements everywhere.
    sys.stdout = TeeWriter(sys.__stdout__, log_file)
    sys.stderr = TeeWriter(sys.__stderr__, log_file)

    problem_context = ProblemContext.create(problem_name)
    problem_context.write_surface_vtp()

    solve_static_problem(
        M=problem_context.M,
        S=problem_context.S,
        mass_specific_potential=problem_context.mass_specific_potential,
        body_center_r_initial=problem_context.body_center_r_initial,
        body_angle_initial=problem_context.body_angle_initial,
        vtu_filename_base='solution',
        mesh_refinement_level_count=problem_context.mesh_refinement_level_count,
        # show_plot=True,
    )
