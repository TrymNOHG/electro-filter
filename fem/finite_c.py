import numpy as np
from scipy.linalg import solve
import scipy.sparse.linalg as sp



rsize = 3 # r grid size
zsize = 3 # z grid size
tsize = 5 # amount of time iterations

Result = np.zeros((tsize, zsize, rsize)) # 3d matrix with radius, z, and time dimension

Result[0] = np.ones((zsize, rsize)) # initial condition


D = 3 # Value of D
A = 4 # Value of A
B = 5 # Value of B

rarray = np.linspace(0, 1, rsize) + 0.0001  # r array from 0 to 1 (added +0.0001 to avoid zero division error. this will be removed by boundary conditions)
zarray = np.linspace(0, 1, rsize) # z array from 0 to 1

R = rarray[-1] - rarray[0] # r length (from 0 to 1)
L = rarray[-1] - rarray[0] # z length (from 0 to 1)

dr = rarray[1] - rarray[0] # r step size
dz = rarray[1] - rarray[0] # z step size


tarray = np.linspace(0, 1, tsize) # t array (here from 0 to 1. change if nessecary)

T = rarray[-1] - rarray[0] # T length (here from 0 to 1. change if nessecary)

dt = rarray[1] - rarray[0] # t step size


# Extend the r array in a way so it can be used in the differential equation for 2 dimensions, and not just one
rones = np.ones(rsize)

r1 = np.reshape(rones, (rsize, 1))
r2 = np.reshape(rarray, (1, rsize))

rmatrix = np.dot(r1, r2)
rmatrix = np.concatenate(rmatrix)


# Different parts of the matrix, diagonalized with the i(r), j(z), and n(t) (see chatgpt for what this means)
Dpart = 1/dr**2 * np.diag((-2) * np.ones(rsize**2)) + np.diag(np.ones(rsize**2 - 1), 1) + np.diag(np.ones(rsize**2 - 1), -1)
RLpart = (R/L)**2 / dz**2 * np.diag((-2) * np.ones(zsize**2)) + np.diag(np.ones(zsize**2 - zsize), zsize) + np.diag(np.ones(zsize**2 - zsize), -zsize)
ARpart = A / rmatrix / dr * np.diag(np.ones(rsize**2)) + np.diag(np.ones(rsize**2 - 1), -1)
Bpart = B * (1 - rmatrix**2) / dz * np.diag(np.ones(zsize**2)) + np.diag(np.ones(zsize**2 - zsize), -zsize)

AA = dt * (D * (Dpart + RLpart) + ARpart + Bpart) # Full matrix

# get each next time step from inverting the matrix:
for i in range(tsize-1):
    vector = np.concatenate(Result[i]) # 2d matrix to 1d vector
    solve = sp.spsolve(AA, vector) # invert matrix, and solve
    print(i)
    Result[i+1] = np.reshape(solve, (rsize, zsize)) # return back to 2d matrix grid

    
print(Result)
