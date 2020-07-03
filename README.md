# README

Replication of results presented in:
"Adjoint optimization of pressurized membrane structures using automatic differentiation tools"

## REPLICATION OF RESULTS
You must have dolfin-adjoint with FEniCS and IPOPT installed:
The easiest way is to use Docker (or Singularity on a research cluster)
See http://www.dolfin-adjoint.org/en/latest/download/index.html for detailed instructions

and also 
https://fenics.readthedocs.io/projects/containers/en/latest/introduction.html

To generate the graphs, you will need to have installed matplotlib and pandas 

Some visualizations were made externally using Paraview 
www.paraview.org

## Quickstart

You will need Docker installed on your machine to continue.

First, clone the repo:

`cd FenicsMembranesCMAME`

Then start a Docker session with access to current folder:

`docker run -ti -v $(pwd):/home/fenics/shared quay.io/dolfinadjoint/pyadjoint`

Enter the repo:

`cd shared`

Install the membrane library (use `-e` (editable) flag if you want to make any changes):

`pip3 install --user  -e .`

We also need these dependencies:
`pip3 install --user matplotlib`
`pip3 install --user pandas`

The examples are in the following directory:

`cd CMAME/examples`

Run a problem:

`python3 main_results.py p1`

then plot the results:

`python3 main_results.py p1 --plot`

The first time runnning the code will take significantly longer becuase of compilation.

## Singularity
The code can also be run on a research cluster with resource management ie SLURM
using Singularity. To convert the docker image to singularity:

`singularity pull docker://quay.io/dolfinadjoint/pyadjoint:2019.1.0`

To run batch jobs:

`python3 main_results.py p1 --slurm`

## Details

The problem data from the paper are configured in the convenience script: `main_results.py`.

If running on a cluster, this script will automatically create and submit slurm scripts. If so,
the `path_to_sif` variable in 'main_results.py' may need to be edited to provide the correct path to the `.sif` file
 
If running on a personal machine, stdout is redirected to the output file specified by the `--path` variable

The script is called from the command line with a command corresponding to the problem ie
` python3 main_results.py p1`

The above command will recreate the results for the plane strain 
pressure minimization problem (Problem 1). 

Once the optimization terminates, the results can be plotted:
` python3 main_results.py p1 --plot`

Problems 1 and 2A have the option to also calculate and save the gradient for the chosen control bounds.
` python3 main_results.py p1a --deriv`

To run the Taylor tests:
` python3 main_results.py p1 --tt`

To disable optimization (if you only want to run the taylor test and/or gradient study)
` python3 main_results.py p1 --tt --deriv --no-optimize `
` python3 main_results.py p1 --tt --no-optimize `

See `--help` for more choices.

For convenience the main results can be replicated with one command:
` python3 main_results.py all `
    
To experiment with the code, you can also pass other args to the individual problem scripts directly:
`python3 problem1.py [<options>]`

See each problem for details.

