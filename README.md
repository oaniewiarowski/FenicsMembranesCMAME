# README

## REPLICATION OF RESULTS
You must have dolfin-adjoint with FEniCS and IPOPT installed:
The easiest way is to use Docker (or Singularity on a research cluster)
See http://www.dolfin-adjoint.org/en/latest/download/index.html for detailed instructions

and also 
https://fenics.readthedocs.io/projects/containers/en/latest/introduction.html

To generate the graphs, you will need to have installed matplotlib and pandas 

pip3 install matplotlib
pip3 install pandas

Other visualizations were made externally using Paraview 
www.paraview.org

## Singularity
The code can also be run on a research cluster with resource management ie SLURM
using Singularity. To convert the docker image to singularity:

>singularity pull docker://quay.io/dolfinadjoint/pyadjoint:2019.1.0

## Use

The problem data from the paper are configured in the script: `main_results.py`.
If on a cluster, this script will automatically create and submit the slurm scripts.
 
If running on a personal machine, the script needs to be edited by changing
`SLURM = True` to `SLURM = False` at the top of the script. 

The script is called from the command line with a command corresponding to the problem ie
> python3 main_results.py p1

The above command will recreate the results for the plane strain 
pressure minimization problem (Problem 1). 

Once the otpimization terminates, the results can be plotted:
> python3 main_results.py p1 --plot

Problems 1 and 2A have the option to also calculate and save the gradient for the chosen control bounds.
> python3 main_results.py p1a--deriv

To run the Taylor tests:
> python3 main_results.py p1 --tt

To disable optimization (if you only want to run the taylor test and/or gradient )
> python3 main_results.py p1 --tt --deriv --no-optimize
> python3 main_results.py p1 --tt --no-optimize

See --help for more choices.

For convenience the main results can be replicated with one command:
> python3 main_results.py all
    
To experiment with the code, you can also pass other args to the individual problem scripts. 
See each problem for details.


##

clone the repo


cd FenicsMembranesCMAME


start docker session with access to current folder:

docker run -ti -v $(pwd):/home/fenics/shared quay.io/dolfinadjoint/pyadjoint

enter the repo:

cd shared

install the membrane library

pip3 install --user  -e .

pip3 install --user matplotlib
pip3 install --user pandas
