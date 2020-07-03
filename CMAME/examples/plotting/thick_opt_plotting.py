#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:49:33 2020

@author: alexanderniewiarowski

Plotting script for Problem 2 (thickness optimization)
Select the data with the method flag and MAX_ITER and RES.

method refers to the objective functional used
    KS = Kreisselmeier Steinhausser of abs(lambda - lambda allowable)
    abs = integral of abs(lambda - lambda allowable)

Currently, the Paraview data is saved manually to csv files
(This can be automated, but installing paraview via conda would have been risky)
** Thickness function is saved as the last timestep so need to fstforward before saving the data!!!
"""
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rc_file
import argparse

rc_file('journal_rc_file.rc')
#plt.style.use('seaborn-whitegrid')

out_path = 'submission/rev1/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)


# For convenience 
labels = {
        'KS': r'$J = KS\left( (\lambda_1 - \bar{\lambda})^2\right)$',
        'abs': r'$J = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$',
        'lsq': r'$\hat{J} = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$'
        }
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='plot problem 3a')
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    fname = args.path
    method = args.method

    #%% Plot inital v optimized thicknesses and stretches....
    
    # csv file saved in Paraview. File > Save Data
    df_0 = pd.read_csv(f'{fname}initial.csv')
    df_1 = pd.read_csv(f'{fname}final.csv')
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=[6.5, 3])
    
    ax0.plot(df_0['Points:0'], df_0.l1, label='Initial stretch')
    ax0.plot(df_1['Points:0'], df_1.l1, label='Optimized stretch')
    
    ax0.set_xlabel(r'$\xi^1$')
    ax0.set_ylabel(r'$\lambda_1$')
    ax0.legend()
    
    ax1.plot(df_0['Points:0'] ,df_0.thickness, label='Initial thickness')
    ax1.plot(df_1['Points:0'], df_1.thickness, label='Optimized thickness')
    
    ax1.set_xlabel(r'$\xi^1$')
    ax1.set_ylabel('Thickness')
    ax1.legend()
    
    ax0.grid(True)
    ax1.grid(True)
    ax0.text(.5, 1.1, '(a)', horizontalalignment='center', verticalalignment='top', transform=ax0.transAxes)
    ax1.text(.5, 1.1, '(b)', horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(f'{fname}_results.pdf', dpi=600)
    plt.savefig(out_path+f'P3A_{method}_rev1.pdf', dpi=600)

    #%% Plot converence history
    with open(f'{fname}_conv_history.json') as json_file:
        results = json.load(json_file)
        
    fig, ax = plt.subplots(figsize=[3.25,3])
    ax.plot(results['j'][0:150], label=labels[method])
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$J$')
    ax.grid(True)
    ax.text(.5, 1.1, '(c)', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(out_path+f'P3A_conv_{method}_rev1.pdf', dpi=600)

