import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,1,.1)

w = [1,1/4,0]

T = 10
om = 2*np.pi/T
y = 1/2*(1-t)**2

n = len(w)

cf = np.convolve(w,y,mode='valid')
cs = np.convolve(w,y,mode='same')

vec = [w,y,cf]

fig,ax = plt.subplots(3)
for i,yp in enumerate(vec):
    ax[i].plot(yp,label='valid')
    ax[i].set_xlim(0,len(w)+len(y))



N = len(cf)-1
ax[2].scatter(N,cf[N])
ax[2].scatter(N,cs[N])
ax[2].plot(cs,label='same')
ax[2].legend()
plt.show()
