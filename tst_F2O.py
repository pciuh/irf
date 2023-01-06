#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date:   Tue, 3 Jan 2023, 06:37

@author: pciuh

Compute 2nd Order Forces from AQWA

"""
import os
import sys
sys.path.append(os.getcwd()+'/modules/')

import numpy as np
import scipy.signal as scs
import scipy.interpolate as sci
import matplotlib.pyplot as plt

import matplotlib.animation as ani

import time

from module_simul import synth,synth2
from module_aqwa import read_qtf

print('Current Version:-',sys.version)

start_time = time.time()

G = 9.80665

iDir = 'inputs/'
pDir = 'png/'
cDir = 'coeff/'
rDir = 'results/'

chn = ['X','Y','Z','RX','RY','RZ']

#### Case conditions
MU  = 60.0
VS  = 14.0
US  = (VS-0)*1852/3600
PRF = 'OMG'
PRF = 'NOS'
#### Time conditions
Ts,Fs = (900,10)
SEED = 291176
#SEED = 200278
#### Wave conditions
Hs,gam  = 5.0,1.0
cth = 4.565-0.87*np.log(gam)
Tp  = cth*np.sqrt(Hs)
wave = {'HS':Hs,'TP':Tp,'GAM':gam}

nam = 'D%.3dV%.3d-'%(MU,10*VS)+PRF
fnam = iDir + nam + '.QTF'

Few = np.load(cDir+nam+'-Fw.npy')

om = np.real(Few[-1])
fkda = np.abs(Few[:-1])
fkdp = np.angle(Few[:-1])

IND = 2

fe,_ = synth(Ts,Fs,(om,fkda[IND],fkdp[IND]),wave,SEED)

################### READ QTF
qtf = read_qtf(fnam,IND)
y,ts = synth2(Ts,Fs,qtf,wave,SEED)
ym = np.mean(y)

print('Runtime: %3.2fs\n'%(float(time.time() - start_time)))

fig,ax = plt.subplots(figsize=(9,2.7))
ax.set_title('$F^{(2)}_{%i}$'%(IND+1))
ax.plot(ts,fe+ym,lw=2,alpha=.5,label='$O^{(1)}$')
ax.plot(ts,y+fe,'--r',lw=.75,label='$O^{(2)}$')
ax.legend()
ax.hlines(ym,0,Ts,ls='--',lw=2,color='tab:red')
ax.annotate('%.2e'%ym,(Ts,ym),ha='left',va='center')
ax.set_xlim(right=1.25*Ts)
fig.savefig('2nd_order.png',dpi=300)

def animmm():

    F2d = np.sqrt(Pd**2+Qd**2)
    F2s = np.sqrt(Ps**2+Qs**2)
    Phd = np.arctan2(Qd,Pd)
    Phs = np.arctan2(Qs,Ps)

    #stop
    omc,omr = np.meshgrid(om,om)
    #Fd = Fd.reshape(nom,-1)
    #Fs = Fs.reshape(nom,-1)
    #Phd = Phd.reshape(nom,-1)
    #Phs = Phs.reshape(nom,-1)

    Ts,fs = (30,10)
    t = np.arange(0,Ts,1/fs)
    ns = len(t)
    F = np.zeros((nom,nom,ns))

    for i,w1 in enumerate(om):
        for ii,w2 in enumerate(om):
            F[i,ii] = F2d[i,ii]*np.cos((w2-w1)*t+Phd[i,ii])+F2s[i,ii]*np.cos((w1+w2)*t+Phs[i,ii])

    rao = (om,(Pd,Ps),(Qd,Qs))

    fig,ax = plt.subplots()
    fig.suptitle('$F_{%i}$'%(IND+1))

    cnt = ax.contourf(omr,omc,F[:,:,0])
    fig.colorbar(cnt,ax=ax)

    def animate(i):
        ax.clear()
        cnt = ax.contourf(omr,omc,F[:,:,i])
        ax.set_title('%12s%8.1f'%('Time:',t[i]))
        ax.set_aspect(1)

    anim = ani.FuncAnimation(fig,animate,ns,interval=1000/fs,blit=False)
    #cbr = fig.colorbar(cnt)

    #cbr.formatter.set_powerlimits((0,0))
    #cbr.formatter.set_useMathText(True)
    #ticklabel_format(axis='y',useMathText=True,scilimits=(0,0))
    #cbr.update_ticks()
    plt.show()
