
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

from module_aqwa import *

def synth(Ts,Fs,rao,wave,SEED):

    if SEED:
        rnds = np.random.seed(SEED)

    dom = 2*np.pi/Ts
    om,P,Q = rao

    omc,omr = np.meshgrid(om,om)
    omd,oms = omr-omc,omr+omc

    Pd,Ps = P
    Qd,Qs = Q

    phd = np.arctan2(Qd,Pd)
    phs = np.arctan2(Qs,Ps)

    oMin,oMax = np.ceil(min(om)/dom)*dom,np.floor(max(om)/dom)*dom

    print(Ts,Fs)
    ts = np.arange(0,Ts,1/Fs)

    omi = np.arange(oMin,oMax,dom)
    omci,omri = np.meshgrid(omi,omi)
    nomi = len(omi)

    fpd = sci.RegularGridInterpolator((om,om),Pd)
    fps = sci.RegularGridInterpolator((om,om),Ps)

    fqd = sci.RegularGridInterpolator((om,om),Qd)
    fqs = sci.RegularGridInterpolator((om,om),Qs)

    fphd = sci.RegularGridInterpolator((om,om),Phd)
    fphs = sci.RegularGridInterpolator((om,om),Phs)

    Pdi = fpd((omri,omci))
    Psi = fps((omri,omci))

    Qdi = fqd((omri,omci))
    Qsi = fqs((omri,omci))

    phi = (1-2*np.random.random(nomi))*np.pi

    phdi = fphd((omri,omci))
    phsi = fphs((omri,omci))

    fi = omi/2/np.pi
    s,f = Jonswap(wave,fi)

    sc,sr = np.meshgrid(s,s)

    wa   = np.sqrt(2*s/Ts)
    wac  = np.sqrt(2*sc/Ts)
    war  = np.sqrt(2*sr/Ts)

    yp,yq = 0,0

    sMN = 1

    for ir in range(sc.shape[0]):
        for ic in range(sc.shape[1]):
            yp += wa[ir]*wa[ic]*(Pdi[ir,ic]*np.cos(-(omi[ir]-omi[ic])*ts+phdi[ir,ic]+phi[ir]-phi[ic])+
                                 sMN*Psi[ir,ic]*np.cos(-(omi[ir]+omi[ic])*ts+phsi[ir,ic]+phi[ir]-phi[ic]))
            yq += wa[ir]*wa[ic]*(Qdi[ir,ic]*np.sin(-(omi[ir]-omi[ic])*ts+phdi[ir,ic]+phi[ir]-phi[ic])+
                                 sMN*Qsi[ir,ic]*np.sin(-(omi[ir]+omi[ic])*ts+phsi[ir,ic]+phi[ir]-phi[ic]))
    return (yp+yq,ts)

start_time = time.time()

G = 9.80665

iDir = 'inputs/'
pDir = 'png/'
cDir = 'coeff/'
rDir = 'results/'

chn = ['X','Y','Z','RX','RY','RZ']

#### Case conditions
MU  = 60.0
VS  = 10.0
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
fnam = iDir + nam + '.QTF'

if os.path.exists(fnam)==False:
    print('\nFile %s does not exist!'%fnam)
    sys.exit()
else:
    print('\n File %s loaded!'%fnam)

################### READ QTF
IND = 1

file1 = open(fnam, 'r') 
Lines = file1.readlines() 
file1.close()

cs=((8,80))

line=str(Lines[1:2])
nf=int(line[6:8])


nl=np.size(Lines)
nc=5
nt=int((nl-nc)/4)

count=int(np.ceil(nf/6))+2


qtf=np.zeros((nl-4))
om=np.array([])
for line in Lines[2:count]:
    om=np.append(om,np.array(line[cs[0]:cs[1]].split(),dtype=np.float64))



i = 0
for line in Lines[count:]:
    sline=np.array(line[cs[0]:cs[1]].split(),dtype=np.float64)
    qtf[i] = sline[IND]
    i+=1

nom = len(om)

Pd=qtf[0:-4:4]
Qd=qtf[1:-3:4]
Ps=qtf[3:-2:4]
Qs=qtf[4::4]

Pd = Pd.reshape(nom,-1)
Qd = Qd.reshape(nom,-1)
Ps = Ps.reshape(nom,-1)
Qs = Qs.reshape(nom,-1)

F2d = np.sqrt(Pd**2+Qd**2)
F2s = np.sqrt(Ps**2+Qs**2)
Phd = np.arctan2(Qd,Pd)
Phs = np.arctan2(Qs,Ps)

qtf = (om,(Pd,Ps),(Qd,Qs))

y,ts = synth(Ts,Fs,qtf,wave,SEED)

print('Runtime: %3.2fs\n'%(float(time.time() - start_time)))

fig,ax = plt.subplots()
ax.set_title('$F^{(2)}_{%i}$'%(IND+1))
ax.plot(ts,y)
plt.show()

stop
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
