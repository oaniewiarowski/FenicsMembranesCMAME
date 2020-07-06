#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:48:32 2020

@author: alexanderniewiarowski
"""

import pandas as pd
import os

from matplotlib import rc_file

rc_file('journal_rc_file.rc')
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')

out_path = 'submission/rev1/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

l10 = r'$\lambda_1$ initial'
l11 = r'$\lambda_1$ final'
l20 = r'$\lambda_2$ initial'
l21 = r'$\lambda_2$ final'
if __name__ == "__main__":
    
    fname = './PressureOpt/Pinned/'
    
    # Plot inital v optimized stretches....
    
    # csv file saved in Paraview. File > Save Data
    US0 = pd.read_csv(f'{fname}US_initial_stretches.csv')
    US1 = pd.read_csv(f'{fname}US_opt_stretches.csv')
    USDS0 = pd.read_csv(f'{fname}USDS_initial_stretches.csv')
    USDS1= pd.read_csv(f'{fname}USDS_opt_stretches.csv')
    
    
    fig, (US, USDS) = plt.subplots(1,2, figsize=[6.5, 3], sharey=True)
    len_ = len(US0['Points:0'])
    skip = 40
    
    US.plot(US0['Points:0'], US0.l1, '-*', label=l10, linewidth=3, markersize=2)
    US.plot(US1['Points:0'], US1.l1,  '-.',label=l11, linewidth=2)
    US.plot(US0['Points:0'], US0.l2, '-', label=l20, linewidth=3)
    US.plot(US1['Points:0'][:len_:skip], US1.l2[:len_:skip], '--o', label=l21, linewidth=1, markersize=2, alpha=0.5)
    
    US.set_xlabel(r'$\xi^1$')
    US.set_ylabel(r'$\lambda$')
    US.legend()

    len_ = len(USDS0['Points:0'])
    skip = 40
    USDS.plot(USDS0['Points:0'], USDS0.l1, '-*', label=l10, linewidth=3, markersize=2)
    USDS.plot(USDS1['Points:0'], USDS1.l1, '-.', label=l11, linewidth=2)
    USDS.plot(USDS0['Points:0'], USDS0.l2, '-', label=l20, linewidth=3)
    USDS.plot(USDS1['Points:0'][:len_:skip], USDS1.l2[:len_:skip], '--o',label=l21, linewidth=1, markersize=2, alpha=0.75)
    
    USDS.set_xlabel(r'$\xi^1$')
    USDS.legend()
    
    US.grid(True)
    USDS.grid(True)
    US.text(.5, 1.1, '(a)', horizontalalignment='center', verticalalignment='top', transform=US.transAxes)
    USDS.text(.5, 1.1, '(b)', horizontalalignment='center', verticalalignment='top', transform=USDS.transAxes)
    plt.tight_layout()
    
    plt.savefig(out_path+'Fig_pressure_stretch.pdf', dpi=600)
    
