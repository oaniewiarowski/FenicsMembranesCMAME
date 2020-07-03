#!/bin/bash
#SBATCH -N 1  # nodes
#SBATCH --cpus-per-task 8 #  # of processes
#SBATCH -o HD_3D_plane_strain/log.out # stdout is redirected to that file
#SBATCH -e HD_3D_plane_strain/error.err # stderr is redirected to that file
#SBATCH -t 08:01:00 # time required
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=aan2@princeton.edu

module load anaconda3
source activate fenics2019
srun python3 w_eta_cost_func_script.py --directory "HD_3D_plane_strain" --constant_t --num_w 50 --num_eta 25 --dim 3 --res 40 40  --bc "Pinned"