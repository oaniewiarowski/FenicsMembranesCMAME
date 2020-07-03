#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:41:44 2020
Created on Sat Jan 25 22:41:48 2020
Created on Sat Jan 25 19:40:17 2020
Created on Thu Nov 14 14:52:51 2019
@author: alexanderniewiarowski
"""
import argparse
import pandas as pd
import numpy as np
import sympy as sp
import os

from mpl_toolkits.mplot3d.art3d import LineCollection
from matplotlib import rc_file
rc_file('journal_rc_file.rc')
import matplotlib.pyplot as plt

H = 1  # depth
# DATA FOR PLOTTING BOUNDS BETWEEN H=1 H=2
ws = []
etah1 = []
etah2 = []


def get_eta(W, H):
    '''
    W = width
    H = minimum height ie depth
    '''
    assert W > 0.001, "does not work for small W"
    eta = sp.symbols('eta', negative=True)
    z_c = 0.5 * W * (sp.sin(eta)/(1-sp.cos(eta)))
    R = W/sp.sqrt(2*(1 - sp.cos(eta)))
    # tried many different settings, this seems to work for wide range of W
    e = -sp.nsolve(z_c + R - H, -6.2, bisect=True)  # - bc want positve eta
    return e


for w in np.linspace(.25, 4, 100):
    try:
        e1 = get_eta(w, 1)
        e2 = get_eta(w, 2)
        ws.append(float(w))
        etah1.append(float(e1))
        etah2.append(float(e2))
    except ValueError:
        continue


def plot_2D(df_ref,
            df_opt, 
            SURROUND,
            w_bounds,
            eta_bounds,
            ax,
            ref_val='lambda_max'):
    '''
    PLOT CONTOURS (FILLED AND LINES) OF REFERENCE VAL FROM GRIDSEARCH
    PLOT COORDS OF GRIDSEARCH
    PLOT PATH OF OPT SOLVER

    df_ref: the pd dataframe with reference values
    df_opt: the pd dataframe with optimization results
    ref_val: string to indicate which z val we are plotting
    '''
    # ref_col = the column name in df_ref with the reference function
    ref_col = f'USDS_{ref_val}' if SURROUND else f'US_{ref_val}'

    # the reference function values
    z_vals = df_ref[ref_col]

    # plot grid search coordinates
    dots = ax.scatter(df_ref.W,
                      np.abs(df_ref.eta),
                      c=z_vals,
                      cmap="coolwarm",
                      alpha=0.9,
                      s=1,
                      zorder=2,
                      edgecolors='k',
                      linewidths=0.3)

    # plot the reference function (filled contours)
    contour = ax.tricontourf(df_ref.W,
                             np.abs(df_ref.eta),
                             z_vals,
                             levels=80,
                             cmap='coolwarm',
                             alpha=1)

    # plot the reference function contour lines
    ax.tricontour(df_ref.W,
                  np.abs(df_ref.eta),
                  z_vals,
                  linewidths=0.5,
                  colors='k')

    # optimization iterates
    cols = df_opt.columns
    x = df_opt[cols[0]]  # w
    y = df_opt[cols[1]]  # eta
    z = df_opt[cols[2]]  # J
    print(np.count_nonzero(~np.isnan(x)))

    # plot optimization path 2D lines on groundplane
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    width_ln = np.linspace(0.4, 3, len(x))
    lc = LineCollection(segments, color='k', linewidths=width_ln, zorder=1)
    ax.add_collection(lc)

    # color out the convex hull that tricontour puts in above H=2
    ax.fill_between(ws, etah2, np.max(etah2), color='white', zorder=2)

    # OPTION TO PLOT CONSTRAINT BOUNDS CORRESPONDING TO H=1 and H=2
#    ax.plot(ws, etah1, color='black', label='H=1', linewidth=1)
#    ax.plot(ws, etah2, color='red', label='H=2', linewidth=1)
#    ax.legend()
#    fill = ax.fill_between(ws, etah1, etah2, alpha=0.1)
#    ax.add_collection(fill)

    # labels
    ax.set_xlabel(r'$W$')
    ax.set_ylabel(r'$\eta=\frac{L}{R}$')

    # Display optimization results in text
    s0 = 'Optimization:'
    s1 = r'$\hat{J}_{min}=$' + rf'${z.values[-1]:.4f}$'  # last J
    s2 = rf'$(W,\eta)=({x.values[-1]:.2f}, {y.values[-1]:.2f})$'  # last w, eta
    s3 = f'Iter: {len(z)-1}'  # iter count starts at 0!
    s = s0 + '\n' + s1 + '\n' + s2 + '\n' + s3
    ax.text(0.97, 0.97, s,
            transform=ax.transAxes,
            multialignment='right',
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=9)

    # Display grid search results in text
    # first: get w, eta, ref_col where ref_col is min
    pW_min = df_ref.iloc[df_ref.idxmin(axis=0)[ref_col]].W
    peta_min = -df_ref.iloc[df_ref.idxmin(axis=0)[ref_col]].eta
    pz_min = df_ref.iloc[df_ref.idxmin(axis=0)[ref_col]][ref_col]
    print(pW_min, peta_min)

    if ref_val[0] == 'l':  # if starts with l then it's lambda max
        z_label = r'$minmax(\lambda_1)=$'
    else:  # KS, compose latex str eg KS_p=300(lambda_1)
        z_label = r'$KS_{p=' + rf'{ref_val[-3:]}' + '}(\lambda_1)=$'
    s0 = 'Grid search:'
    s1 = z_label + rf'${pz_min:.4f}$'
    s2 = rf'$(W,\eta)=({pW_min:.2f}, {peta_min:.2f})$'
    s = s0 + '\n' + s1 + '\n' + s2
    ax.text(0.97, 0.75, s,
            transform=ax.transAxes,
            multialignment='right',
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=9)

    # Plot minimizers found by grid search and optimization
    W_min = df_ref.W.iloc[z_vals.idxmin()]
    eta_min = -df_ref.eta.iloc[z_vals.idxmin()]
    print(W_min, eta_min)
    red_marker, = ax.plot(W_min,
                          eta_min,
                          marker='+',
                          c='r',
                          markersize=8,
                          linestyle="None")

    #  Label the optimum value with an black X
    black_marker, = ax.plot(x.values[-1],
                            y.values[-1],
                            marker='x',
                            c='k',
                            markersize=8,
                            linestyle="None")
    ax.legend([black_marker, red_marker],
              ["Optimization", "Grid search"],
              loc='lower left',
              title='Minimum obtained by:')

    # Limits
    ax.set_xlim(w_bounds[0], w_bounds[1])
    return contour, dots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot problem 1a')
    parser.add_argument('--resx', type=int, required=True)
    parser.add_argument('--resy', type=int, required=True)
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--path', type=str, required=False)
    args = parser.parse_args()

    RESX = args.resx
    RESY = args.resy
    p = args.p

    # Get the reference data
    # The results of the grid search are found here: (see w_eta_cost_func_script.py)
    df_ref = pd.read_json("../cost_func_plots/HD_3D_plane_strain_KS/df_hydro.json")

    fname = args.path # f'3D_shape_opt/Pinned/p{p}/res{RESX}x{RESY}/' if args.path is None else

    # iterate over difference background values...
    for ref_val in ['lambda_max', f'KS_lambda_{p}']:

        # Setup the figure with 2 subplots and colorbar
        fig = plt.figure(figsize=(6.5, 3.5))
        from mpl_toolkits.axes_grid1 import AxesGrid
        axs = AxesGrid(fig,
                       111,
                       nrows_ncols=(1, 2),
                       axes_pad=0.1,
                       cbar_mode='single',
                       cbar_location='right',
                       cbar_pad=0.1,
                       cbar_size='5%')

        contours = []  # for colorbar
        dots = []  # for colorbar
        for i, SURROUND in enumerate([False, True]):
            name_str = 'SURROUND' if SURROUND else 'NO_SURROUND'
            w_bounds = (0.25, 4)
            eta_bounds = (-6.2, -1)
            df = pd.read_json(fname + f'{name_str}_iterates.json')
            contour, dot = plot_2D(df_ref,
                                   df,
                                   SURROUND,
                                   w_bounds,
                                   eta_bounds,
                                   axs[i],
                                   ref_val=ref_val)
            contours.append(contour)
            dots.append(dot)

        # the min and max J values in the two subplots
        cmin = min(contours[0].get_clim()[0], contours[1].get_clim()[0])
        cmax = max(contours[0].get_clim()[1], contours[1].get_clim()[1])

        contours[0].set_clim(cmin, cmax)
        contours[1].set_clim(cmin, cmax)
        contours[0].changed()
        contours[1].changed()

        dots[0].set_clim(cmin, cmax)
        dots[1].set_clim(cmin, cmax)
        dots[0].changed()
        dots[1].changed()

        # The most frustrating plot ever:
        # using axesGrid, the colorbar cbar wouldnt update...
        # ...(only reflected data ranges for the subplot data range, no clim range)
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                   norm=plt.Normalize(vmin=cmin, vmax=cmax))
        sm.set_array(np.linspace(cmin, cmax, 82))
        cbar = axs.cbar_axes[0].colorbar(sm)

        from matplotlib.ticker import MaxNLocator
        my_locator = MaxNLocator(8)
        cbar.cbar_axis.set_major_locator(my_locator)
        cbar.changed()
        cbar.ax.get_yaxis().labelpad = 15
        
        if ref_val[0] == 'l':  # if starts with l then it's lambda max
            cbar_label = r'$max(\lambda_1)$'
        else:  # KS, compose latex str eg KS_p=300(lambda_1)
            cbar_label = r'$KS_{p=' + rf'{ref_val[-3:]}' + '}(\lambda_1)$'
        cbar.ax.set_ylabel(cbar_label, rotation=270)

        axs[0].text(0.5, 1.1, '(a)',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=axs[0].transAxes)
        axs[1].text(0.5, 1.1, '(b)',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=axs[1].transAxes)

        out_path = 'submission/rev1/figures'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

#        plt.savefig(fname + f'{RESX}x{RESY}_results2d_{ref_val}.pdf')
        plt.savefig(f'{out_path}/p2a_{ref_val}{p}.pdf', dpi=600)
# %%
    '''
    Now make the figure inspecting the active contraint at H=1
    '''
    df = df_ref

    def minw(df, zvals):
        # helper function to get W, J values at lowest zvals (J)
        return df.W.iloc[zvals.idxmin()], zvals.min()

    # setup the figure
    fig, axs = plt.subplots(1, 2, figsize=[6.5, 3], sharey=True)
    axs[0].set_ylabel(r'$J$')
    axs[0].text(0.5, 1.1, '(a)',
                horizontalalignment='center',
                verticalalignment='top',
                transform=axs[0].transAxes)
    axs[1].text(0.5, 1.1, '(b)',
                horizontalalignment='center',
                verticalalignment='top',
                transform=axs[1].transAxes)

    #axs[0].set_title("1 side")  # option to
    #axs[1].set_title("2 sides")

    loads = ["US", "USDS"]
    for k, load in enumerate(loads):
        KS300 = []
        KS500 = []
        KS600 = []
        lmax = []
        W = []
        # Here we plot the values of true max lambda, and the KS approximations:
        # We are plotting values along the bottom constraint, 50 W x 25 eta
        # First of each (W,eta) pair is on the contraint)
        # Count by number of eta's in df, here 25.
        for i in range(0, len(df), 25):
            KS300.append(df[f'{load}_KS_lambda_300'].iloc[i])
    #        KS500.append(df[f'{load}_KS_lambda_500'].iloc[i])  # too crowded w 500
            KS600.append(df[f'{load}_KS_lambda_600'].iloc[i])
            lmax.append(df[f'{load}_lambda_max'].iloc[i])
            W.append(df['W'].iloc[i])

        # Now plot the extracted data
        ax = axs[k]
        ax.plot(W, lmax, '-', label=r'$max(\lambda_1)$')
        ax.plot(W, KS300, '-', label=r'$KS(\lambda_1), \sigma=300$')
    #    ax.plot(W, KS500,'-', label=r'$KS(\lambda_1), \sigma=500$')  # too crowded w 500
        ax.plot(W, KS600, '-', label=r'$KS(\lambda_1), \sigma=600$')
        ax.legend()
        ax.set_prop_cycle(None)  # reset color cycle
        top_legend = ax.legend()
        # mark the minimum value in each series with a dot
        for zvals in [df[f'{load}_lambda_max'],
                      df[f'{load}_KS_lambda_300'],
    #                  df[f'{load}_KS_lambda_500'],  # too crowded w 500
                      df[f'{load}_KS_lambda_600']]:
            x, y = minw(df, zvals)
            # mark reference optimizers with a red cross
            red_marker, = ax.plot(x, y,
                                  marker='+',
                                  c='r',
                                  markersize=8,
                                  linestyle="None")
        ax.set_xlabel(r'$W$')

    # now mark the returned optimizers for each load case and KS sigma
    for i, name_str in enumerate(['NO_SURROUND', 'SURROUND']):
        for p in [300, 600]:
#            fname = f'ShapeOpt/Pinned/p{p}/res{RESX}x{RESY}/'
            df_opt = pd.read_json(fname+f'{name_str}_iterates.json')
            cols = df_opt.columns
            x = df_opt['w']
            y = df_opt['w']
            z = df_opt['J']
            # mark optimizers with black X
            black_marker, = axs[i].plot(x.values[-1],
                                        z.values[-1],
                                        marker='x',
                                        c='k',
                                        markersize=8,
                                        linestyle="None")

        ax.legend([black_marker, red_marker],
                  ["Optimization", "Grid search"],
                  loc='best',
                  title='Minimum obtained by:')
    plt.tight_layout()

    plt.savefig(f'{out_path}/p2a_detail_constraint.pdf', dpi=600)
