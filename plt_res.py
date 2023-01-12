#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date:   Fri, 06 Jan 2023, 11:15

@author: pciuh

Postprocessing of simulation results

"""
import os
import sys
sys.path.append(os.getcwd()+'/modules/')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

rDir = 'results/sim/'

#### Case conditions
MU  = 60.0
VS  = 10.0
US  = (VS-0)*1852/3600
PRF = 'OMG'
PRF = 'NOS'
SEED = 200278
SEED = 291176
nam = 'D%.3dV%.3d-'%(MU,10*VS)+PRF
suf = '-simT-S%.6d'%SEED
fnam = rDir + nam + suf + '.ftr'

nd = 6

df = pd.read_feather(fnam)

chn = df.columns

t = df.Time.values
zw = df.Wave.values

nb = 2

chnm = [x.replace('mot','m') for x in chn[nb:nb+3]]
chna = [x.replace('mot','deg') for x in chn[nb+3:nb+nd]]
chnp = chnm + chna

gam = 1
Hs = 4*np.std(zw)
cth = 4.565-0.87*np.log(gam)
Tp  = cth*np.sqrt(Hs)

fig,ax = plt.subplots(nd,figsize=(6,6))
fig.tight_layout(pad=1)
fig.suptitle('$H_S$=%4.1fm, $T_P$=%4.1fs'%(Hs,Tp),va='bottom')

for i in range(nd):
    ic = i+nb
    ax[i].plot(t,df[chn[ic]],alpha=.7)
    ax[i].set_ylabel(chnp[i])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_position('zero')

ax[-1].set_xlabel('Time (s)')
fig.savefig(nam+'.png',dpi=300,bbox_inches='tight')
fig.savefig(nam+'.pdf',dpi=300,bbox_inches='tight')
