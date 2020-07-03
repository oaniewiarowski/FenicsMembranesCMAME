#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:34:44 2020

@author: alexanderniewiarowski
"""
import pandas as pd
import argparse
import os
from matplotlib import rc_file
rc_file('journal_rc_file.rc')
import matplotlib.pyplot as plt


def plot_results(dfs, labels):

    # rows for table with results
    rw = []
    re = []
    raw = []
    rae = []
    rj = []
    rows = [rw, re, raw, rae, rj]

    fig, axs = plt.subplots(3, 2, figsize=(6.5, 4))
    for i, df in enumerate(dfs):
        df = pd.read_json(df)
        axs[0][0].plot(df.w, label=labels[i])
        axs[0][0].set_title(r'$W_{0}$')

        axs[0][1].plot(df.eta, label=labels[i])
        axs[0][1].set_title(r'$\eta_{0}$')

        axs[1][0].plot(df.a_w, label=labels[i])
        axs[1][0].set_title(r'$\alpha_{w}$')

        axs[1][1].plot(df.a_e, label=labels[i])
        axs[1][1].set_title(r'$\alpha_{\eta}$')

        axs[2][0].plot(df.J, label=labels[i])
        axs[2][0].set_title(r'$\hat{J}_{min}$')

        axs[2][1].set_title("Final values")

        rw.append(f'{df.w.iloc[-1]:.3f}')
        re.append(f'{df.eta.iloc[-1]:.3f}')
        raw.append(f'{df.a_w.iloc[-1]:.3f}')
        rae.append(f'{df.a_e.iloc[-1]:.3f}')
        rj.append(f'{df.J.iloc[-1]:.3f}')

    axs[2][1].axis('off')

    # List final values in table
    row_lables = [r'$W_{0}$',
                  r'$\eta_{0}$',
                  r'$\alpha_{w}$',
                  r'$\alpha_{\eta}$',
                  r'$\hat{J}_{min}$']

    table = plt.table(cellText=rows,
                      rowLabels=row_lables,
                      colLabels=["1 side", "2 sides"],
                      loc='center',
                      edges='open')
    table.auto_set_column_width([0, 1])

    axs[2][1].add_table(table)
    handles, labels = axs[2][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    return fig, axs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot problem 2b')

    parser.add_argument('--path', type=str, required=False)
    args = parser.parse_args()

    path = args.path
    #RES = '40x40'
    #sub_dir = 'Capped/4var'
    #labels = ["1 side", "2 sides"]
    #dfs_all = [f"3D_shape_opt/{sub_dir}/res{RES}/4var_NO_SURROUND_iterates.json",
    #           f"3D_shape_opt/{sub_dir}/res{RES}/4var_SURROUND_iterates.json"]
    #
    #fig, axs = plot_results(dfs_all, labels)
    #plt.tight_layout()
    #
    #plt.savefig(f"3D_shape_opt/{sub_dir}/res{RES}/results.pdf")
    #plt.savefig('./submission/figures/p1b.pdf', dpi=600)
    

    labels = ["1 side", "2 sides"]
    #3D_shape_opt/Pinned/p300/res40x40
    dfs_all = [f"{path}/4var_NO_SURROUND_iterates.json",
               f"{path}/4var_SURROUND_iterates.json"]
    
    fig, axs = plot_results(dfs_all, labels)
    plt.tight_layout()
    #%%
    out_path = 'submission/rev1/figures'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(f'{out_path}/p2b.pdf', dpi=600)