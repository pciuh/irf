#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date:   Sun, 11 Dec 2022, 10:29

@author: pciuh

This script simulate 6DOF motions of single frequency wave excitation.

"""
import os
import sys
sys.path.append(os.getcwd()+'/modules/')

import numpy as np
import pandas as pd

import scipy.signal as scs
import scipy.interpolate as sci
import matplotlib.pyplot as plt

import time

from module_aqwa import countomega, readomega, readrao, readcoeff
from module_simul import *

def plocmp(om,rao,raoi,TIT,chn,vdeg):

    nd = len(vdeg)
    fig,ax = plt.subplots(nd,figsize=(10,2*nd))
    fig.suptitle(TIT)
    for i,m in enumerate(vdeg):

        ax[i].plot(om,rao[m],lw=3,alpha=.4,label='AQWA')
        ax[i].plot(om,raoi[m],label='IRF')
        ax[i].set_ylabel(chn[m])
    ax[0].legend()
    return fig

def psd(z,Fs):
    n = int(len(z)/2)
    Z = np.fft.fft(z)
    ZA = np.abs(Z[:n])/n
    ZP = np.angle(Z[:n])
    f  = np.linspace(0,Fs/2,n)
    return(f,ZA,ZP)


def funmot(x,args):
    fe,M1,B,C = args
    dxdt1 = x[1]
    b = np.zeros_like(B)
    dxdt2 = M1@(fe - B@x[1] - C@x[0])
#    dxdt2 = M1@(fe - C@x[0])
    return(np.array([dxdt1,dxdt2]))

def radforce(u,kt):
    return np.convolve(u,w*kt,mode='same')[-1]

def radforce_ss(tc,u,sys):
    _,fr,_ = sys.output(U=u,T=tc)
    return fr

def radforce_l(u,Tc,Fs,SS):
    tc = np.arange(0,2*Tc+1/Fs/2,1/Fs)
    _,fr,_ = scs.lsim(SS,U=u,T=tc)
    return fr[-1]

def rk2(dt,u,fun,args):
#### Ralston's method.
    k1 = fun(u,args)
    alf = 2/3
    k2 = fun(u+alf*dt*k1,args)
    alf1,alf2 = 1-1/2/alf,1/2/alf
    return dt*(alf1*k1+alf2*k2)

def rk4(dt,u,fun,args):
#### Classical 4th order method
    k1 = fun(u,args)
    k2 = fun(u+dt/2*k1,args)
    k3 = fun(u+dt/2*k2,args)
    k4 = fun(u+dt*k3,args)
    return dt/6*(k1+2*k2+2*k3+k4)

#def solve(ns,Fs,Tc,few,M,B,C,SS):
def solve(ns,Fs,Tc,few,M,B,C,sys):

    fe = few.T
    nd = M.shape[0]
    u = np.zeros((2,nd))
    mot,vel,fr = (np.zeros((ns,nd)),np.zeros((ns,nd)),np.zeros((ns,nd)))
    dt = 1/Fs
    fr = np.zeros((ns,nd))
    M1 = np.linalg.inv(M)
    nc = int(Tc*Fs)
    MET = 'RK2'

    for i in range(1,ns):
        print('%4.1f'%(100*i/ns)+'%',end='\r')

        ft = fe[i-1] - fr[i-1]
        args = (ft,M1,B,C)

        if MET == 'RK4':
            u += rk4(dt,u,funmot,args)
        else:
            u += rk2(dt,u,funmot,args)

        mot[i] = u[0]
        vel[i] = u[1]

 #       if i>nc+1:
 #           nb = nc+1
 #       else:
 #           nb = 1
 #       if i>nc+1:
            #B = np.zeros_like(M)
        ii = 0
        for ir in range(nd):
            for ic in range(nd):
#                    v = vel[i-nb:i,ic]
#                    fr[i,ir] += radforce(v,Tc,Fs,(Ar[ir,ic],Br[ir,ic],Cr[ir,ic],Dr[ir,ic]))
                fr[i,ir] += radforce_ss(Fs*i,vel[i,ic],sys[ii])
                ii+=1
                #fr[i,ir] += radforce(v,ktr[ir,ic])
    return(mot.T,vel.T,fr.T)

start_time = time.time()

G = 9.80665

iDir = 'inputs/'
pDir = 'png/'
cDir = 'coeff/'
rDir = 'results/'

#### Case conditions
MU  = 120.0
VS  = 14.0
US  = (VS-0)*1852/3600
PRF = 'OMG'
PRF = 'NOS'
#### Wave conditions
Hs,gam  = 5.0,1.0
cth = 4.565-0.87*np.log(gam)
Tp  = cth*np.sqrt(Hs)
wave = {'HS':Hs,'TP':Tp,'GAM':gam}

#### Simulation parameters
Ts,Tc,Fs = (900.,30.,10.)
SEED = 291176
#SEED = 5112007
#SEED = 200278

print('Simulation Conditions:')
print('VS(knots) %8.1f\n'%VS)
print('Wave:')
print('mu(deg) %8.1f'%MU)
print(' Hs(m) %8.1f'%Hs)
print(' Tp(s) %8.2f'%Tp)
print('Gamma %8.1f'%gam)
print('omp(rad/s) %8.3f'%(2*np.pi/Tp))

nam = 'D%.3dV%.3d-'%(MU,10*VS)+PRF
fnam = iDir + nam + '.LIS'

if os.path.exists(fnam)==False:
    print('\nFile %s does not exist!'%fnam)
    sys.exit()
else:
    print('\n File %s loaded!'%fnam)

nom = countomega(fnam)

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

#cKa,cKp = np.load(cDir+nam+'-cka.npy'),np.load(cDir+nam+'-ckp.npy')

nd = Ainf.shape[0]

ts = np.arange(0,Ts,1/Fs)
tc = np.arange(0,Tc+1/2/Fs,1/Fs)

ns,nd = len(ts),6

om = np.real(Few[-1])
fkda = np.abs(Few[:-1])
fkdp = np.angle(Few[:-1])
zwa  = np.abs(RAO[:-1])
zwp  = np.angle(RAO[:-1])

chn = ['X','Y','Z','RX','RY','RZ']

df = pd.DataFrame(Ainf,columns=chn,index=chn)
#print(df)


nc = int(2*Tc*Fs) + 1

### Exitation force and RAO motions synthesis
fe,zr = np.zeros((nd,ns+nc)),np.zeros((nd,ns))
for i in range(nd):
    fe[i,nc:],ts = synth(Ts,Fs,(om,fkda[i],fkdp[i]),wave,SEED)
    zr[i],_      = synth(Ts,Fs,(om,zwa[i],zwp[i]),wave,SEED)

zw,_ = synth(Ts,Fs,(om,np.ones_like(om),np.zeros_like(om)),wave,SEED)

print('Hs(m)%8.1f:'%(4*np.std(zw)))

As,Bs = Ainf,np.zeros((nd,nd))

tc = np.arange(-Tc,Tc+1/Fs/2,1/Fs)
nc = len(tc)
ktr = np.zeros((nd,nd,nc))
w = scs.tukey(nc,1/2)
w = 1
#omE,dom = 10.,1e-2
#omo = np.arange(0,omE,dom)
#            Kxo = fka(omo)*np.exp(1j*fkp(omo))

#            Axo = Kxo.imag/omo
#            Bxo = Kxo.real

#            kt  = ktfn(tc,om,Bx-Bnf)
#            kto = ktfn(tc,omo,Bxo-Bnf)

omE,dom = 4*np.pi,1e-3
tt = np.arange(0,Tc+1/Fs/2,1/Fs)
ss = []
for ir in range(nd):
    for ic in range(nd):
        num = (ir+1)*10+ic+1
        _,ktt = scs.impulse((Ar[ir,ic],Br[ir,ic],Cr[ir,ic],Dr[ir,ic]),T=tt)
#        ktt = np.append(np.zeros(len(tt)-1),ktt)
        ktt = np.append(np.flip(ktt[1:]),ktt)
#        ktr[ir,ic] = 1/Fs*ktt
#        fb = sci.interp1d(om,B[:,ir,ic])
        #omn,_,Bmn = approx(om,A[:,ir,ic],B[:,ir,ic],dom,omE)
        #ktt = ktf(omn,Bmn,tc)
        #if ir == ic:
            #fig,ax = plt.subplots()
            #ax.plot(tc,ktt,label='SS')
            #ax.plot(tc,ktf(omn,Bmn,tc),label='DFT')
            #ax.set_ylabel('k%i%i'%(ir+1,ic+1))
        ss.append(scs.lti(Ar[ir,ic],Br[ir,ic],Cr[ir,ic],Dr[ir,ic]))
        #ktr[ir,ic] = 1/Fs*ktt

#ax.legend()
#plt.show()
#### Solver
print('Number of time steps: %i'%ns)
#mot,vel,fr = solve(ns+nc+1,Fs,Tc,fe,M+As,Bs,C,ktr)
mot,vel,fr = solve(ns+nc+1,Fs,Tc,fe,M+As,Bs,C,ss)

print('Runtime: %3.2fs\n'%(float(time.time() - start_time)))
zr[3:] = np.degrees(zr[3:])
mot[3:] = np.degrees(mot[3:])

fig = plocmp(ts,zr,mot[:,-ns:],'SiMULATiON',chn,[2,3,4])
fig.savefig(pDir + 'sim/' + nam + '-sim3D.png',dpi=300)


print('%6s%8s%8s'%('','Simul','Synth'))
for i in range(nd):
    print('%6s:%8.2f%8.2f'%(chn[i],np.std(mot[i]),np.std(zr[i])))


#### Storage to file (FEATHER)
mot,vel,fe,fr = mot.T[-ns:],vel.T[-ns:],fe.T[-ns:],fr.T[-ns:]

key = ['Time','Wave']+[x+'(mot)' for x in chn]+[x+'(vel)' for x in chn]+[x+'(fkd)' for x in chn]+[x+'(rad)' for x in chn]
vec = np.concatenate((ts.reshape(-1,1),zw.reshape(-1,1),mot,vel,fe,fr),axis=1)
df = pd.DataFrame(vec,columns=key)
df.to_feather(rDir + 'sim/'+nam+'-simT.ftr')
df.to_csv(rDir + 'sim/'+nam+'-simT.csv')
