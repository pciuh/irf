
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

import time

from module_aqwa import *

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

print(fnam)
