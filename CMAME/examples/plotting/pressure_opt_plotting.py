#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:24:53 2020

@author: alexanderniewiarowski


Plotting script for Problem 1 (pressure optimization)


"""
import pandas as pd
import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rc_file

rc_file('journal_rc_file.rc')

#%% Plot converence history and pressure
def plot_2_in_1(path_list, load_names):
    fig, axs = plt.subplots(1, 2, figsize=[6.5,3], sharey='all')
    axs[0].set_ylabel(r'$\hat{J}$')
    axs[0].text(.5, 1.1, '(a)', horizontalalignment='center', verticalalignment='top', transform=axs[0].transAxes)
    axs[1].text(.5, 1.1, '(b)', horizontalalignment='center', verticalalignment='top', transform=axs[1].transAxes)
    ax0 = axs[0]
    ax1 = ax0.twinx()
    ax2 = axs[1]
    ax3 = ax2.twinx()
    ax0.get_shared_y_axes().join(ax0, ax2)
    ax1.get_shared_y_axes().join(ax1, ax3)
    
    axs = [ax0, ax1, ax2, ax3]
    [ax.grid(True) for ax in axs[:3:2]]
    for i, f in enumerate(path_list):
        
        with open(f'{f}_conv_history.json') as json_file:
            results = json.load(json_file)

        color_cycle = axs[i*2]._get_lines.prop_cycler
        label = r'$J$'
        ln1 = axs[i*2].plot(results['j'][0:150], label=label)
        axs[i*2].set_xlabel("Iterations")

        ln2 = axs[i*2+1].plot(results['m'], '*', label=r'$p_0$', color=next(color_cycle)['color'])
        # put labels on one legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        
        s0 = r'$p_{opt=}$'+f'{results["pressure"]:.3f}'    
        s1 = f'Height={results["HEIGHT"]:.3f}'
        s2 = f'Height (KS)={results["HEIGHT_KS"]:.3f}'
#        s3 = r'$\lambda_{min}=$' + f'{results["true_lambda_min"]:.3f}' 
#        s4 = r'$KS(\lambda_{min})=$' + f'{results["lambda_min_KS"]:.3f}' 
        s5 = f'iter={len(results["m"])}'
        s = s0 + '\n' + s1 + '\n' + s2 + '\n' + s5#+ '\n' + s4 + '\n' + s5
        text = axs[i*2].text(0.95, .35, s,
                             transform=axs[i*2+1].transAxes,
                             multialignment='right',
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=9,
                             zorder=10)
        text.set_bbox(dict(facecolor='white', alpha=1, edgecolor='black'))
        
        print(results['j'][-1])

    axs[3].legend(lns, labs, loc='best')
    axs[3].set_ylabel('Inflation pressure')   


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='plot problem 1')
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    fname = args.path

    bcs = 'Pinned'
    p = 600
    paths = [f'PressureOpt/{bcs}/US/3/{p}/res100x1_itr20/',
             f'PressureOpt/{bcs}/USDS/3/{p}/res100x1_itr20/']
    
    plot_2_in_1(paths, ['(a)','(b)'])
    plt.tight_layout()
    
    out_path = 'submission/rev1/figures/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(out_path+'pressure_convergence.pdf', dpi=600)


