#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 20:01:58 2022

@author: pciuh
"""
import os
import sys
sys.path.append(os.getcwd()+'/modules/')

import numpy as np
import scipy.signal as scs
import scipy.interpolate as sci
import matplotlib.pyplot as plt

import time

from module_aqwa import countomega, readomega, readrao, readcoeff
from module_simul import *

def mirmat(At):
    Ar = np.rot90(np.fliplr(At),1)
    Ad = np.diag(Ar)
    An = Ad*np.eye(len(Ad))
    return Ar+At-An

start_time = time.time()

iDir = 'inputs/'
pDir = 'png/'
cDir = 'coeff/'

MIR = False

NS = 13    #### Maximum degree of State Space Model

MU = 120.0
VS = 14.0

SUF = 'OMG'
SUF = 'NOS'

nam = 'D%.3dV%.3d-'%(MU,10*VS)+SUF
fnam = iDir + nam + '.LIS'

chn= ['XX','YY','ZZ','RX','RY','RZ']
keya = [x+'A' for x in chn]
keyp = [x+'P' for x in chn]
vec = np.linspace(1,6,6,dtype=np.int8)
chna = dict(zip(vec,keya))
chnp = dict(zip(vec,keyp))

print(fnam)

nom = countomega(fnam)
ome,om=readomega(fnam,nom)

am,bm,cm,mm = readcoeff(fnam,nom)
per = np.array([x for x in am])

zwa = readrao(fnam,'R',nom).values[:,2::2].T
zwp = readrao(fnam,'R',nom).values[:,3::2].T
zwa[3:] = np.radians(zwa[3:])

fwa = readrao(fnam,'FA',nom).values[:,2::2].T
fwp = readrao(fnam,'FA',nom).values[:,3::2].T

A = np.zeros((nom,6,6))
B = np.zeros((nom,6,6))

for i in range(nom):
    A[i,:,:] = np.array(am[per[i]]).reshape(6,6)
    B[i,:,:] = np.array(bm[per[i]]).reshape(6,6)

##################### IRF parameters
Tc  = 20.0
Fs  = 10.

No = 6

omE = No*max(om) #1/2*np.pi*Fs
if omE > No*np.pi:
    omE = No*np.pi

dom = 1e-3
omE = np.round(omE/dom)*dom

print('Omega.Ext:',omE)

t  = np.arange(0,Tc,1/Fs)
w = scs.tukey(len(t),1/6)
w = np.ones(len(t))


M,C = (mm['IM'],cm['SM'])

lbl = ['xx','yy','zz','rx','ry','rz']
Pxx =[]

v = np.array([11,22,33,44,55,66,13,15,24,26,35,42,46,51,53,62,64])
print(v)
v =[]
for ir in range(6):
    for ic in range(6):
        ter,tec = ir%2,ic%2
        if ter==tec:
            v = np.append(v,(ir+1)*10+ic+1)
v = v.astype('int32')

nv = len(v)
nc = 3
nr = int(nv/nc)
ind = np.concatenate((np.tile(np.arange(nr),nc),np.repeat(np.arange(nc),nr)),axis=0).reshape(2,-1).T

Px,Qx = (np.zeros((6,6,NS+1),dtype=complex),np.zeros((6,6,NS+1),dtype=complex))
x = tuple([6,6,NS,NS])

Ar,Br,Cr,Dr = (np.zeros((6,6,NS,NS)),np.zeros((6,6,NS,1)),np.zeros((6,6,1,NS)),np.zeros((6,6,1,1)))
Ainf = np.zeros((6,6))

Fe,RAO = np.zeros((7,nom),dtype=complex),np.zeros((7,nom),dtype=complex)
for i in range(6):
    Fe[i] = fwa[i]*np.exp(1j*np.radians(fwp[i]))
    RAO[i] = zwa[i]*np.exp(1j*np.radians(zwp[i]))


Fe[-1] = om
RAO[-1] = om
i = 0
ii = 0
#print(B[:,2,3])
#### Insert values at om=0
#om = np.append([1e-6],om)
#idx = np.append([0],np.arange(0,nom))
#A = A[idx]
#B = B[idx]
#B[0] = 0
#print(B[:,2,3])
#print(om)
#stop
fig,ax = plt.subplots(nr,nc,figsize=(10,10))
fif,af = plt.subplots(nr,nc,figsize=(10,10))
fib,ab = plt.subplots(nr,nc,figsize=(10,10))

fig.suptitle('Impulse Response Function Matrix')
fif.suptitle('Added Mass Matrix')
fib.suptitle('Damping Matrix')
aPer,bPer = .1,0.4
for ir in range(6):
    for ic in range(6):

        num = (ir+1)*10+ic+1

        if np.any(v==num):

            Ax,Bx = A[:,ic,ir],B[:,ic,ir]

            #### Estimation of infinit added mass
            Ainfty,Aint = adminf(om,Ax,Bx)

            print('%i%i:'%(ir+1,ic+1))
            #oma,An = approx(om,Ax-Ainfty,dom,omE,per=aPer)
            #An = An + Ainfty
            #### Impulse response function in time domain
            #if ir!=ic:
            #    Bx = 1/2*(B[:,ir,ic]+B[:,ir,ic])
            #else:
            #if num == 35:

            #else:
            #PER = 0.0
            omn,Bn = approx(om,Bx,dom,omE,per=bPer)

#            omo = np.linspace(min(om),omE-dom,101)
#            fa,fb = sci.interp1d(oma,An,kind='quadratic'),sci.interp1d(omn,Bn,kind='quadratic')

            Ainf[ir,ic] = Ainfty

            ktn = ktfn(t,omn,Bn)
            ktd = ktf(omn,Bn,t)
            TF,SS = era_sys(ktd,Fs,NS,False)

            th,kth = scs.impulse(SS,T=t)
            Ak,Bk = komfn(th,om,kth,Ainfty,0)

            enp,enq = TF[0].shape[1],TF[1].shape[0]
            en = SS[0].shape[0]
            Ar[ir,ic,:en,:en] = SS[0]
            Br[ir,ic,:en] = SS[1]
            Cr[ir,ic,:,:en] = SS[2]
            Dr[ir,ic] = SS[3]
            Px[ir,ic,:enp],Qx[ir,ic,:enq]=TF

            idx = tuple(ind[i])
            ax[idx].plot(t,ktn,label='AQWA+extrap',alpha=.5,lw=2.5)
            ax[idx].plot(t,ktd,lw=1.25)
            ax[idx].plot(th,kth,'--r',label='ERA',lw=1.)
            ax[idx].set_ylabel(r'$k_{%i%i}$'%(ir+1,ic+1))
            ax[idx].ticklabel_format(axis='y',useMathText=True,scilimits=(0,0))
            fig.tight_layout(w_pad=.5,h_pad=.5)

            af[idx].scatter(om,Ax,alpha=.3)
            #af[idx].plot(oma,An,'--r',label='AQWA+extrap',lw=1.5)
            af[idx].plot(om,Ak,'--r',label='Approx',lw=1.5)
            #af[idx].plot(Aint[0],Aint[1],label='Interp',lw=1.)
            af[idx].hlines(Ainf[ir,ic],min(omn),max(omn),ls='--',color='#000000bb',lw=.5)
            af[idx].annotate(r'$A_{\infty}$:%4.2e'%Ainf[ir,ic],(max(omn),Ainf[ir,ic]),va='bottom',ha='right')
            af[idx].set_ylabel(r'$A_{%i%i}$'%(ir+1,ic+1))
            af[idx].set_xscale('log')
            af[idx].ticklabel_format(axis='y',useMathText=True,scilimits=(0,0))
            fif.tight_layout()

            ab[idx].scatter(om,Bx,alpha=.3,label='AQWA')
            ab[idx].plot(omn,Bn,c='tab:blue',label='AQWA+extrap',lw=.5)
            ab[idx].plot(om,Bk,'--r',label='Approx',lw=1.5)
            ab[idx].set_ylabel(r'$B_{%i%i}$'%(ir+1,ic+1))
            ab[idx].set_xscale('log')
            ab[idx].ticklabel_format(axis='y',useMathText=True,scilimits=(0,0))
            fib.tight_layout()

            i+=1

        ii+=1

        print('%6.1f'%(100*ii/36)+'%',end='\r')


#print('Ori:',Ar[:,:,0,0])
if MIR:
    m = Ar.shape[2]
    Dr[:,:,0,0] = mirmat(Dr[:,:,0,0])
    for ir in range(m):
        Br[:,:,ir,0] = mirmat(Br[:,:,ir,0])
        Cr[:,:,0,ir] = mirmat(Cr[:,:,0,ir])
        for ic in range(m):
            Ar[:,:,ir,ic] = mirmat(Ar[:,:,ir,ic])

#print('Mir:',Ar[:,:,0,0])
af[0,0].legend()
ax[0,0].legend()
fig.savefig(pDir+'irf/D%.3dV%.3d'%(MU,10*VS)+'-irf.png',dpi=300)
fif.savefig(pDir+'ainf/D%.3dV%.3d'%(MU,10*VS)+'-ainf.png',dpi=300)
fib.savefig(pDir+'ainf/D%.3dV%.3d'%(MU,10*VS)+'-bapp.png',dpi=300)
plt.close(fig)
plt.close(fif)
plt.close(fib)

np.save(cDir+nam+'-M.npy',M)
np.save(cDir+nam+'-A.npy',A)
np.save(cDir+nam+'-B.npy',B)
np.save(cDir+nam+'-C.npy',C) 
np.save(cDir+nam+'-Ainf.npy',Ainf)
np.save(cDir+nam+'-Ar.npy',Ar)
np.save(cDir+nam+'-Br.npy',Br)
np.save(cDir+nam+'-Cr.npy',Cr)
np.save(cDir+nam+'-Dr.npy',Dr)
np.save(cDir+nam+'-Px.npy',Px)
np.save(cDir+nam+'-Qx.npy',Qx)
np.save(cDir+nam+'-Fw.npy',Fe)
np.save(cDir+nam+'-rao.npy',RAO)
print('Execution time:%12.1f'%(float(time.time() - start_time))+' seconds\n')
