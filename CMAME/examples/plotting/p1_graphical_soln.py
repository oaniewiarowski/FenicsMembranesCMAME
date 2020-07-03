#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:48:32 2020

@author: alexanderniewiarowski
"""

import pandas as pd
import argparse
import os

from matplotlib import rc_file

rc_file('journal_rc_file.rc')
import matplotlib.pyplot as plt


out_path = 'submission/rev1/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--itr', type=int, required=True)
    parser.add_argument('--res', type=str, required=True)
    args = parser.parse_args()

    p = args.p
    itr = args.itr
    res = args.res
    path_list = [f'PressureOpt/Pinned/US/3/{p}/res{res}_itr{itr}/',
           f'PressureOpt/Pinned/USDS/3/{p}/res{res}_itr{itr}/']

    labels = [r'$\hat{J} = \int \lambda_1 d\xi$',
              r'$\hat{J} =KS(height)$',
              r'$\hat{J} =KS(min(\lambda_2))$']

    fig, axs = plt.subplots(1, 2, figsize=[6.5, 3], sharey=True)

    for i, df in enumerate(path_list):
        df = pd.read_json(df+'J_dJdm.json')
        cols = df.columns
        for j in range(3):
            axs[i].hlines(1, df['p'].iloc[0], df['p'].iloc[-1],
                          linestyle='--',
                          linewidth=1,
                          alpha=0.5)

            axs[i].scatter(df['p'],
                           df[cols[j+1]],
                           marker='o',
                           s=2,
                           label=labels[j])

        axs[i].set_prop_cycle(None)
        axs[i].set_xlabel(r'$p_0$')
    axs[0].set_ylabel(r'$\hat{J}$')
    axs[0].text(.5, 1.1, '(a)',
                horizontalalignment='center',
                verticalalignment='top',
                transform=axs[0].transAxes)
    axs[1].text(.5, 1.1, '(b)',
                horizontalalignment='center',
                verticalalignment='top',
                transform=axs[1].transAxes)
    plt.legend()   

    plt.tight_layout()
    plt.savefig(out_path+'pres_opt_graphical_soln.pdf', dpi=600)
