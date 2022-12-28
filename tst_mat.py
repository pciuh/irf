import numpy as np

A0 = np.zeros((3,3))
A1 = np.ones((3,3))

D1 = np.ones((2,2))

B = np.array([[A0,A1],[A1,A0]])

C = B@D1

print(C.shape)
