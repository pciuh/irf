#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date:   Fri, 16 Dec 2022, 08:01

@author: pciuh

This script simulate 6DOF motions of single frequency wave excitation.

"""
import os
import sys
sys.path.append(os.getcwd()+'/modules/')

import numpy as np
import scipy.signal as scs
import scipy.interpolate as sci
import matplotlib.pyplot as plt

import time

from module_aqwa import *

def plores(om,rao,raoi,TIT,chn,vdeg):

    nd = len(vdeg)
    fig,ax = plt.subplots(nd,figsize=(10,2*nd))
    fig.suptitle(TIT,va='bottom')
    for i,m in enumerate(vdeg):

        ax[i].plot(om,rao[m],lw=3,alpha=.4,label='AQWA')
        ax[i].plot(om,raoi[m],label='IRF')
        ax[i].set_ylabel(chn[m])
        ax[i].ticklabel_format(axis='y',useMathText=True,scilimits=(-5,5))
    ax[0].legend()
    fig.tight_layout(pad=.96)
    return fig

def plocoef(om,A,Ak,TIT,chn,rdeg,cdeg):
    nd = len(cdeg)
    fig,ax = plt.subplots(nd,figsize=(6,2*nd))
    fig.suptitle(TIT)
    i = 0
    for ir,mr in enumerate(rdeg):
        for ic,mc in enumerate(cdeg):
            if ir==ic:
                ax[i].plot(om,A[:,mr,mc],lw=3,alpha=.4,label='AQWA')
                ax[i].plot(om,Ak[:,mr,mc],label='IRF')
                ax[i].set_ylabel(chn[mr]+'-'+chn[mc])
                ax[i].ticklabel_format(axis='y',useMathText=True,scilimits=(-5,5))
                i += 1
    ax[0].legend()
    fig.tight_layout(pad=.96)
    return fig

def calcrao(om,*args):
    M,A,B,C,few = args
    nom,nd = om.shape[0],M.shape[0]
    rao = np.zeros((nom,nd),dtype=complex)
    for i,o in enumerate(om):
        Frh = C-(M+A[i])*o**2-1j*o*B[i]
        rao[i] = np.linalg.inv(Frh)@few[i]
    return rao.T

start_time = time.time()

G = 9.80665

iDir = 'inputs/'
pDir = 'png/'
cDir = 'coeff/'
rDir = 'results/'

chn = ['X','Y','Z','RX','RY','RZ']

#### Case conditions
MU  = 120.0
VS  = 14.0
US  = (VS-0)*1852/3600
PRF = 'OMG'
PRF = 'NOS'
#### Time conditions
Ts,Tc,Fs = (900,30,10)
SEED = 291176
#SEED = 200278
#### Wave conditions
Hs,gam  = 5.0,1.0
cth = 4.565-0.87*np.log(gam)
Tp  = cth*np.sqrt(Hs)
wave = {'HS':Hs,'TP':Tp,'GAM':gam}

nam = 'D%.3dV%.3d-'%(MU,10*VS)+PRF
fnam = iDir + nam + '.LIS'

if os.path.exists(fnam)==False:
    print('\nFile %s does not exist!'%fnam)
    sys.exit()
else:
    print('\n File %s loaded!'%fnam)


Ainf,Ar,Br,Cr,Dr,M,A,B,C,Few,RAO = (np.load(cDir+nam+'-Ainf.npy'),
                                    np.load(cDir+nam+'-Ar.npy'),
                                    np.load(cDir+nam+'-Br.npy'),
                                    np.load(cDir+nam+'-Cr.npy'),
                                    np.load(cDir+nam+'-Dr.npy'),
                                    np.load(cDir+nam+'-M.npy'),
                                    np.load(cDir+nam+'-A.npy'),
                                    np.load(cDir+nam+'-B.npy'),
                                    np.load(cDir+nam+'-C.npy'),
                                    np.load(cDir+nam+'-Fw.npy'),
                                    np.load(cDir+nam+'-rao.npy')
                                    )

nss = Ar.shape[2]
nd = Ainf.shape[0]

tc = np.arange(0,Tc+1/Fs/2,1/Fs)
n = len(tc)
om = np.real(Few[-1])

few = Few[:-1].T

nom = len(om)

nc = len(tc)

Bk,Ak = np.zeros((nom,nd,nd)),np.zeros((nom,nd,nd))
for ir in range(nd):
    for ic in range(nd):
        _,kt = scs.impulse((Ar[ir,ic],Br[ir,ic],Cr[ir,ic],Dr[ir,ic]),T=tc)
#        kt = ktfn(tc,om,B[:,ir,ic])
        for i,o in enumerate(om):
            ckt =  kt*np.cos(o*tc)
            skt = kt*np.sin(o*tc)
            Bk[i,ir,ic] = np.trapz(ckt,tc)
            Ak[i,ir,ic] = Ainf[ir,ic] - np.trapz(skt,tc)/o

rao = calcrao(om,M,A,B,C,few)
raoi = calcrao(om,M,Ak,Bk,C,few)
raoa = RAO[:-1]

ns = int(Ts*Fs)
zw,zwi,zwa = np.zeros((nd,ns)),np.zeros((nd,ns)),np.zeros((nd,ns))

MN = np.ones(nd)
MN[3:] = 180./np.pi

for i in range(nd):
    zw[i],ts = synth(Ts,Fs,(om,np.abs(rao[i]),np.angle(rao[i])),wave,SEED)
    zwi[i],ts = synth(Ts,Fs,(om,np.abs(raoi[i]),np.angle(raoi[i])),wave,SEED)
    zw[i] = zw[i]*MN[i]
    zwi[i] = zwi[i]*MN[i]

    zwa[i],ts = synth(Ts,Fs,(om,np.abs(raoa[i]),np.angle(raoa[i])),wave,SEED)
    zwa[i] = zwa[i]*MN[i]

zww,_ = synth(Ts,Fs,(om,np.ones_like(om),np.zeros_like(om)),wave,SEED)

figa = plocoef(om,A,Ak,'Added Mass',chn,[2,3,4],[2,3,4])
figa.savefig(pDir + 'syn/' + nam+'-adm.png',dpi=300)

figb = plocoef(om,B,Bk,'Damping',chn,[2,3,4],[2,3,4])
figb.savefig(pDir + 'syn/' + nam+'-dmp.png',dpi=300)

figt = plores(ts,zw,zwi,'SYNTHESIS',chn,[2,3,4])
figt.savefig(pDir + 'syn/' + nam+'-syn3D.png',dpi=300)

key = ['Time','Wave']+[x+'(aqw)' for x in chn]+[x+'(irf)' for x in chn]
vec = np.concatenate((ts.reshape(-1,1),zww.reshape(-1,1),zw.T,zwi.T),axis=1)
df = pd.DataFrame(vec,columns=key)

#df.to_csv('sim/'+nam+'-synT.csv',sep=';')
df.to_feather(rDir + 'syn/'+nam+'-synT.ftr')

vec = np.array([[np.std(x) for x in zwi],[np.std(x) for x in zw]])
df = pd.DataFrame(vec.T,columns=['IRF','AQWA'],index=chn)
#df.round(2).to_csv('sim/'+nam+'-synRMS.csv',sep=';')

print('%6s%8s%8s'%('','IRF','AQWA'))
for i in range(nd):
    print('%6s:%8.2f%8.2f'%(chn[i],np.std(zwi[i]),np.std(zw[i])))



