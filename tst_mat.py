import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs

def psd(z,Fs):
    n = int(len(z)/2)
    Z = np.fft.fft(z)
    ZA = np.abs(Z[:n])/n
    ZP = np.angle(Z[:n])
    f  = np.linspace(0,Fs/2,n)
    return(f,ZA,ZP)

iDir = 'results/'
nam = 'D120V140-NOS'

mode = 3
chn = mode + 1

dfsi = pd.read_feather(iDir + 'sim/' + nam + '-simT.ftr')
dfsy = pd.read_feather(iDir + 'syn/' + nam + '-synT.ftr')

nam_si = dfsi.columns
nam_sy = dfsy.columns

tsi = dfsi.Time
tsy = dfsy.Time

Fs = 1/(tsi[2]-tsi[1])
zsi = dfsi[nam_si[chn]]
zsy = dfsy[nam_sy[chn+6]]

fsi,A_sim,P_sim = psd(zsi,Fs)
fsy,A_syn,P_syn = psd(zsy,Fs)

fig = plt.figure(figsize=(9,6))
fig.suptitle(nam_si[chn])
gs0 = grs.GridSpec(2,3,figure=fig)
axt = fig.add_subplot(gs0[:-1,:])
axa = fig.add_subplot(gs0[1,0])
axp = fig.add_subplot(gs0[1,1])

axt.plot(tsy,zsy,label='Synthetisized')
axt.plot(tsi,zsi,alpha=.75,label='Simulate')
axt.legend()

axa.plot(fsy,A_syn)
axa.plot(fsi,A_sim,alpha=.75)
axa.set_xscale('log')

#axp.plot(fsi,P_sim)
axp.plot(fsy,P_syn-P_sim)
axp.set_xscale('log')
plt.show()
