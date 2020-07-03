#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:00:57 2020

@author: alexanderniewiarowski
"""
import os
import pandas as pd
from matplotlib import rc_file

rc_file('journal_rc_file.rc')
import matplotlib.pyplot as plt

out_path = 'submission/rev1/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

#%%
ms = 3
#res = [r'$\sigma=300, (40 \times 40)$', r'$\sigma=300, (40 \times 40)$', r'$\sigma=300, (100 \times 25)$']
res = [r'$\sigma=300$', r'$\sigma=600$']
df300 = pd.read_json('./ShapeOpt/Pinned/p300/res40x40/SURROUND_J_dJdm.json')
df600 = pd.read_json('./ShapeOpt/Pinned/p600/res40x40/SURROUND_J_dJdm.json')
#dfhd = pd.read_json('3D_shape_opt/Pinned/5steps/p300/res100x25/deriv_study/JdJdebug.json')

fig, axs = plt.subplots(1, 2,  figsize=[6.5, 3])
axs[0].set_xlabel(r'$W$')
axs[0].set_ylabel(r'$\frac{\partial \hat{J}}{\partial W}$')
axs[1].set_xlabel(r'$W$')
axs[1].set_ylabel(r'$\frac{\partial \hat{J}}{\partial \eta}$')

for i, df in enumerate([df300, df600]):

    axs[0].plot(df['w'], df['dJdw'], '.',
       markersize=ms,
               label= res[i])  # r'$\frac{\partial \hat{J}}{\partial W}$' +
    
#    color_cycle = axs[0]._get_lines.prop_cycler
    
    axs[1].plot(df['w'],df['dJdeta'],'.',
              markersize=ms,
              label= res[i]) # r'$\frac{\partial \hat{J}}{\partial \eta}$' +
#              color=next(color_cycle)['color'])

axs[0].legend()
axs[1].legend()

axs[0].text(0.5, 1.1, '(a)',
            horizontalalignment='center',
            verticalalignment='top',
            transform=axs[0].transAxes)
axs[1].text(0.5, 1.1, '(b)',
            horizontalalignment='center',
            verticalalignment='top',
            transform=axs[1].transAxes)
plt.tight_layout()

plt.savefig(out_path+'shape_opt_grads.pdf', dpi=600)