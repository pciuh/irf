import numpy as np
from numba import jit,njit

import time

@njit(fastmath=True)
def repmat(A,ns):
    nr,nc = A.shape
    return A.repeat(ns).reshape(nr,nc,ns)

@njit(fastmath=True)
def mulmat(B,Bo,Bp,Bpr,flag):
    if flag == 'c':
        return B*np.cos(Bo+Bp+Bpr)
    else:
        return B*np.sin(B+B+B)

@njit(fastmath=True)
def calsum(t,P,amp,om,ph,phr,flag):
    n = len(t)
    t = t.reshape((ns,1))
    imat = np.ones((ns,1))
    yp = np.zeros(ns)
    for ir in range(amp.shape[0]):
        if flag == 'P':
            c = amp[ir]*imat*np.cos(om[ir]*t+ph[ir]*imat+phr[ir]*imat)
        else:
            c = amp[ir]*imat*np.sin(om[ir]*t+ph[ir]*imat+phr[ir]*imat)
        yp += P[ir]@c.T
    return yp


stim = time.time()

nr,nc = 256,256
Ts,Fs = 900,10

t= np.arange(0,Ts,1/Fs)
ns = len(t)

A = np.ones((nr,nc))

#B = repmat(A,ns)
C = calsum(t,A,A,A,A,A,'P')
#D = (A@C).sum(axis=1).sum(axis=0)

print(C.shape)

print('Time: ',time.time()-stim)
