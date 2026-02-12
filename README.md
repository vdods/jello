# Curvature-Induced Force Fields in Hyperelasticity

Copyright 2026 Victor Dods

This repository contains all the code that was used to produce the numerical solutions given in the paper 'Curvature-Induced Force Fields in Hyperelasticity', as well as all solution data from which in particular all the plots in the paper were produced (excepting the two images from the 2008 project).

This repository will also contain updates and other work relevant to the paper.  The git branch `curvature-induced-force-fields-in-hyperelasticity` will always point to the state of this repo corresponding to the publication of the paper, regardless of if the `main` git branch has advanced to include additional work.

## Software License

The software in this repository is licensed under the [MIT license](LICENSE).  If you use this code, I request that you include a visible attribution to the effect of:

    Victor Dods, 2026, 'Curvature-Induced Force Fields in Hyperelasticity', (TODO: link to paper), https://github.com/vdods/jello

## Citing This Work

TODO

## Instructions for Viewing Data

The data can be viewed using the data visualization program [`paraview`](https://paraview.org).  This repository contains the solution data presented in the paper, organized under the following directory structure.  `jello/` is the repository root directory.

    jello/
        data/
            flamms-paraboloid-with-gravity/
            funnel-with-gravity/
            paraboloid-upward-with-gravity/
            spherical-cup-with-gravity/

To view a solution's data, run the `paraview` program.  Then click `File > Load State`, navigate to the directory for the solution you want to view, and select the `state.pvsm` file in that directory.  It will present a dialog with a selection dropdown for 'Load State Data File Options'.  From this, select 'Search files under specified directory', ensure the directory shown in the one containing the `state.pvsm` that you opened, and click the checkbox for 'Only Use Files in Data Directory' (I'm not sure if this is actually necessary).  This will load two (or more) "layouts", each of which may have multiple views.  Each view is set up to present a different aspect of the solution -- energy densities (colormapped) and various vector fields (black, semi-transparent arrows).  Please refer to the paper for descriptions.

Paraview is a complicated program, and I will not give a tutorial on how to use it here.  [There is extensive documentation online](https://docs.paraview.org/en/latest/index.html), but asking a chatbot is definitely the best avenue for learning specific ways to use it.

## Instructions for Running Solution Computations

### Setting up Python Environment and Installing Dependencies

These instructions use the `uv` Python package manager.  Installation instructions can be found [here](https://docs.astral.sh/uv/) under "Installation".

From this repository's root directory run the following command to download and install or update all dependencies.

    uv sync

This will create the `.venv` subdirectory which contains the whole "virtual environment" that will run the code.  The `.venv` directory can safely be deleted at any time and reinstalled using the `uv sync` command as above.

### Running Solution Computations

The code is structured as a Python module (Python version 3 required) called `jello` that can be run from this repository's root directory via

    uv run python -m jello

This will print a help message showing usage information.

    Usage: uv run python -m jello <problem-name> <output-directory>
    This will create the output directory (if not already present), write the surface vtp file, and then
    solve the FEM problem and write the solution data to vtu files.

    Problem names, with surfaces given in cylindrical coordinates:
        flamms-paraboloid-with-gravity    : z = 2 * (r_s * (r - r_s))^(1/2) (r_s := Schwarzschild radius = 1.0)
        funnel-with-gravity               : z = -1/r
        paraboloid-upward-with-gravity    : z = x^2 + y^2 / 2
        spherical-cup-with-gravity        : z = -(r^2 - x^2 - y^2)^0.5
        flat-plane-no-gravity             : z = 0

To compute a solution, for example, `funnel-with-gravity`, run the following command

    uv run python -m jello funnel-with-gravity newdata/funnel-with-gravity

This will create the directory `newdata/funnel-with-gravity` if it doesn't exist, run solution computations for a long time, printing a log of the optimization and mesh refinement processes, and gradually fill the directory with the following artifacts.

    initial_body_configuration.vtu  -- mesh data for the initial configuration of the hyperelastic body before optimization.
    log.txt                         -- a log of all the output of the solution process.
    solution.01.vtu                 -- mesh data for level 1 of mesh refinement.
    solution.02.vtu                 -- mesh data for level 2 of mesh refinement.
    solution.03.vtu                 -- mesh data for level 3 of mesh refinement.
    surface.vtp                     -- mesh data for the surface that the body is embedded in.

Note that the `.vtu` files also contain the scalar and vector field data.

This data can be viewed using a data visualization program called [`paraview`](https://paraview.org).  It's a very complicated program and is not beginner friendly, and loading and visualization data from scratch is somewhat difficult.  Thus the full data of the solutions with `state.pvsm` (which includes all the views used to generate the plots in the paper) is provided in this repo.

In order to compute the solutions and generate the data used in the paper, run the following commands, each one producing a directory that contains the solution artifacts.

    uv run python -m jello flamms-paraboloid-with-gravity generated/flamms-paraboloid-with-gravity
    uv run python -m jello funnel-with-gravity generated/funnel-with-gravity
    uv run python -m jello paraboloid-upward-with-gravity generated/paraboloid-upward-with-gravity
    uv run python -m jello spherical-cup-with-gravity generated/spherical-cup-with-gravity

## Other Stuff

There are some tests that run a bunch of the code through its paces.  It can be run via

    uv run python -m jello.test

It will print that all tests pass, otherwise will print out a very very ugly error message.

## To-dos (For the Author)

-   Add citation info, BibTeX entry
-   Add link to arXiv, and then published paper when available
