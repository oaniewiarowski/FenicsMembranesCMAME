#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:30:25 2020

@author: alexanderniewiarowski
"""

import os
import argparse


OPTIMIZE = True
path_to_sif = '../../../'  # make sure this is the right path to singularity .sif file (assuming in home dir)

def format_slurm_file(command, path):
    ''' for running slurm jobs from singularity containers '''
    N = 1  # nodes
    num_cpu = 4  # number of proc
    time = '04:01:00'  # enter max time
    email = ''  # enter your email here

    header = ['#!/bin/bash \n',
             f'#SBATCH -N {N}  # number of nodes \n',
             f'#SBATCH --cpus-per-task {num_cpu} #  assign # processes \n',
             f'#SBATCH -o {path}/out.txt # stdout is redirected to that file \n',
             f'#SBATCH -e {path}/error.txt # stderr is redirected to that file \n',
             f'#SBATCH -t {time} # time required \n',
              '#SBATCH --mail-type=begin \n',
              '#SBATCH --mail-type=end \n',
             f'#SBATCH --mail-user={email} \n',
              'cd \n',
              # current issue with dolfin adjoint docker container on singularity 
              'export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/lib \n',
              'cd FenicsMembranesCMAME/CMAME/examples/ \n',
              f'singularity exec {path_to_sif}pyadjoint_2019.1.0.sif {command}']
    return header


def p1(args):
    RESX = 100
    RESY = 1
    itr = 20
    diag = 'crossed'
    pressure = 18.5  # initial pressure
    H = 1
    bcs = "Pinned"
    force_list = ['USDS', 'US']



    for DIM in [3]:
        for f in force_list:
            for p in [600]:
                if DIM == 2:
                    res_str = f'{RESX}'
                else:
                    res_str = f'{RESX}x{RESY}'

                if f == 'USDS':
                    SURROUND = '--surround'
                    forcing = 'constant'
                elif f == 'US':
                    SURROUND = '--no-surround'
                    forcing = 'constant'
                else:
                    forcing = 'var_head'

                path = f'PressureOpt/{bcs}/{f}/{DIM}/{p}/res{res_str}_itr{itr}/'
                
                if args.plot:
                    print("Plotting problem p1a")
                    os.system(f"python3 plotting/pressure_opt_plotting.py --path {path}")
                    if os.path.isfile('./PressureOpt/Pinned/US_initial_stretches.csv'):
                        os.system("python3 plotting/pressure_stretch_plot.py --path {path}")
                    else:
                        print('***'*50)
                        print("To plot stretches, need to process data in paraview first!")
                        print("Load the initial and final xdmf files, select last time step...")
                        print("Select 'Plot over line' and then File > Save Data to csv.")
                        print("save the US (upstream - one side loading) files as:")
                        print("\t 'PressureOpt/Pinned/US_initial_stretches'")
                        print("\t 'PressureOpt/Pinned/US_final_stretches'")
                        print("save the USDS (upstream/downstream - two side loading) files as:")
                        print("\t 'PressureOpt/Pinned/USDS_initial_stretches'")
                        print("\t 'PressureOpt/Pinned/USDS_final_stretches'")  
                        print('***'*50)
                    if os.path.isfile(path+'J_dJdm.json'):
                        os.system(f"python3 plotting/p1_graphical_soln.py --p {p} --itr {itr} --res {res_str}")
                    else:
                        print('***'*50)
                        print("To plot the graphical solution you first need to run:")
                        print("python3 main_results.py p1 --deriv")
                        print('***'*50)
                    return
                
                if not os.path.exists(path):
                    os.system(f'mkdir -p {path}')

                command = 'python3 problem1.py ' +\
                             f' --resx {RESX} ' +\
                             f'--resy {RESY} ' +\
                             f'--dim {DIM} ' +\
                             f'--diag {diag} ' +\
                             f'--itr {itr} ' +\
                             f'--pressure {pressure} ' +\
                             f'--height {H} ' +\
                             f'{SURROUND} ' +\
                             f'--forcing {forcing} ' +\
                             f' --p {p} ' +\
                             f' --path {path} '
                if args.deriv:
                    command += '--deriv '
                if not args.opt:
                    command += '--no-optimize '
                if args.tt:
                    command += '--tt '
                print(command)
                if args.SLURM:
                    header = format_slurm_file(command, path)
                    with open(f'p1a_{p}.sh', 'w') as file:
                        file.writelines(header)
                    os.system(f'sbatch p1a_{p}.sh')
                else:
                    os.system(f'touch {path}p1a.txt')
                    command += f' > {path}p1a.txt'  # redirect stdout
                    os.system(command)


def p2a(args):
    '''
    shape opt problem plane strain
    100x16 mesh (crossed cells are default)
    Pinned BCs
    '''
    res = [100, 16]
    bcs = "Pinned"

    if args.plot:
        print("Plotting problem 2a")
        for p in [300]:  # only plot sigma 300 path
            path = f'ShapeOpt/{bcs}/p{p}/res{res[0]}x{res[1]}/'
            os.system(f'python3 plotting/problem2a_plots.py --path {path} --p {p} --resx {res[0]} --resy {res[1]}')
            os.system('python3 plotting/problem2a_plot_grads.py')
            print('***'*50)
            print("To plot the graphical solution you first need to run:")
            print("python3 main_results.py p2a --deriv")
            print('***'*50)
        return

    if OPTIMIZE:
        for p in [300, 600]:  # we are using KS sigma 300 and 600
            path = f'ShapeOpt/{bcs}/p{p}/res{res[0]}x{res[1]}/'
            if not os.path.exists(path):
                os.system(f'mkdir -p {path}')

            command = 'python3 problem2.py ' +\
                      f'--path {path} ' +\
                      f'--bcs {bcs} ' +\
                      f'--p {p} ' +\
                      f'--resx {res[0]} ' +\
                      f'--resy {res[1]} '
            if args.deriv:
                command += '--deriv '
            if not args.opt:
                command += '--no-optimize '
            if args.tt:
                command += '--tt '
            print(command)

            if args.SLURM:
                header = format_slurm_file(command, path)
                with open(f'p2a_{p}.sh', 'w') as file:
                    file.writelines(header)
                os.system(f'sbatch p2a_{p}.sh')
            else:
                os.system(f'touch {path}p2a.txt')
                command += f'> {path}p2a.txt'  # redirect stdout
                os.system(command)


def p2b(args):
    '''
    shape opt problem general plane stress
    40x40 mesh (crossed cells are default)
    Pinned BCs
    '''
    res = [40, 40]
    p = 300
    bcs = "Capped"
    path = f'ShapeOpt/{bcs}/p{p}/res{res[0]}x{res[1]}/'

    if args.plot:
        print("Plotting problem 2b")
        os.system(f'python3 plotting/problem2b_plot.py --path {path}')
        return

    if OPTIMIZE:
        if not os.path.exists(f'{path}'):
            os.system(f'mkdir -p {path}')

        command = f'python3 problem2.py --path {path} --p {p} --resx {res[0]} --resy {res[1]} --bcs {bcs} '
        if args.deriv:
            command += '--deriv '
        if not args.opt:
            command += '--no-optimize '
        if args.tt:
            command += '--tt '
        print(command)

        if args.SLURM:
            header = format_slurm_file(command, path)
            with open(f'p2b_{p}.sh', 'w') as file:
                file.writelines(header)
            os.system(f'sbatch p2b_{p}.sh')
        else:
            os.system(f'touch {path}p2b.txt')
            command += f' > {path}p2b.txt'  # redirect stdout
            os.system(command)


def p3a(args):
    resx = 100
    resy = 1
    itr = 50
    lmax = 1.02

    diag = 'left'
    if args.plot:
        method = 'lsq'
        path = f'ThickOpt/plane_strain/{lmax}/{diag}/{method}/res{resx}x{resy}/itr{itr}/'
        print("Plotting problem 3a")
        if os.path.isfile(path+'initial.csv'):
                        os.system(f'python3 plotting/thick_opt_plotting.py --path {path} --method {method}')
        else:
            print('***'*50)
            print("To plot stretches, need to process data in paraview first!")
            print("Load the initial and final xdmf files, select last time step...")
            print("Select 'Plot over line' and then File > Save Data to csv.")
            print("save the initial state data as:")
            print(f"\t '{path}initial.csv'")
            print("save the final state data as:")
            print(f"\t '{path}final.csv'")
            print('***'*50)
        return

    if OPTIMIZE:
        for method in ['lsq']: # ['abs', 'lsq', 'KS']: # 3 objectives were tried
            path = f'ThickOpt/plane_strain/{lmax}/{diag}/{method}/res{resx}x{resy}/itr{itr}/'
            if not os.path.exists(path):
                os.system(f'mkdir -p {path}')

            command = 'python3 p3a.py ' +\
                      f'--path {path} ' +\
                      f'--itr {itr} ' +\
                      f'--resx {resx} ' +\
                      f'--resy {resy} ' +\
                      f'--lmax {lmax} ' +\
                      f'--diag {diag} ' +\
                      f'--method {method} '
            if not args.opt:
                command += '--no-optimize '
            if args.tt:
                command += '--tt '
            print(command)

            if args.SLURM:
                header = format_slurm_file(command, path)
                with open(f'p3a_{method}.sh', 'w') as file:
                    file.writelines(header)
                os.system(f'sbatch p3a_{method}.sh')
            else:
                os.system(f'touch {path}p3a.txt')
                command += f'> {path}p3a.txt'  # redirect stdout
                os.system(command)


def p3b(args):
    res = [100, 100]
    lmax = 1.01
    bcs = "Capped"
    diag = 'crossed'
    itr = 300
    tdeg = 0   # interpolation degree for thickness
    element = 'DG'  # for thickness field
    method = 'lsq'
    path = f'ThickOpt/plane_stress/{bcs}/lmax{lmax}/res{res[0]}x{res[1]}/{diag}/{method}/{element}{tdeg}_itr{itr}/'
    if args.plot:
        print("Plotting problem 3b")
        os.system(f'python3 plotting/3d_thick_opt_plotting.py --method {method} --path {path}')
        return
    if OPTIMIZE:
        if not os.path.exists(path):
            os.system(f'mkdir -p {path}')

        command = 'python3 p3b.py ' +\
                  f'--resx {res[0]} ' +\
                  f'--resy {res[1]} ' +\
                  f'--lmax {lmax} ' +\
                  f'--bcs {bcs} ' +\
                  f'--diag {diag} ' +\
                  f'--method {method} ' +\
                  f'--itr {itr} ' +\
                  f'--tdeg {tdeg} ' +\
                  f'--element {element} ' +\
                  f'--path {path} '
        if not args.opt:
            command += '--no-optimize '
        if args.tt:
            command += '--tt '
        print(command)
        if args.SLURM:
            header = format_slurm_file(command, path)
            with open(f'3b{method}.sh', 'w') as file:
                file.writelines(header)
            os.system(f'sbatch 3b{method}.sh')
        else:
            os.system(f'touch {path}p3b.txt')
            command += f' > {path}p3b.txt'  # redirect stdout
            os.system(command)


def run_all(args):
    '''
    to reproduce the main figures, run the following:
    p1 --tt --deriv
    '''
    p1(args)
    p2a(args)
    p2b(args)
    p3a(args)
    p3b(args)


if __name__ == "__main__":

    PROBLEM_MAP = {'p1': p1,     # pressure opt
                   'p2a': p2a,   # shape opt plane strain
                   'p2b': p2b,   # thick opt plane stress
                   'p3a': p3a,   # thick opt plane strain
                   'p3b': p3b,   # thick opt plane stress
                   'all': run_all}

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--string', type=str)
    parser.add_argument('--plot',
                        dest='plot',
                        action='store_true',
                        default=False,
                        help= "plot this problem, need latex or disable plt rc file")

    parser.add_argument('--no-optimize',
                        dest='opt',
                        action='store_false',
                        default=True,
                        help="do not run the optimization -\
                        ie just run taylor test and/or gradient study")

    parser.add_argument('--deriv',
                        dest='deriv',
                        action='store_true',
                        default=False,
                        help="if applicable, run the derivative study \
                        (eval and store J, dJdm for range of m)? \
                        this can take a long. time default is False")

    parser.add_argument('--tt',
                        dest='tt',
                        action='store_true',
                        default=False,
                        help="run the taylor test? default is False")

    parser.add_argument('command', choices=PROBLEM_MAP.keys())

    parser.add_argument('--slurm',
                        dest='SLURM',
                        action='store_true',
                        default=False,
                        help="generate slurm scripts if running on a cluster")

    args = parser.parse_args()

    problem = PROBLEM_MAP[args.command]
    problem(args)
