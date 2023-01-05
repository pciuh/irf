import numpy as np


a = np.ones((4,1))*2
b = np.ones((1,12))*3

a[1] = -1
a[2] = 4

c = a@b

print(c)
