from py_ped import pyPED
from personalFunctions import *

case = 2

if case == 1:
    """
    test power iteration for single J
    """
    ped = pyPED()
    n = 4
    J = rand(n, n)
    Q = rand(n, n)
    q, r, d, c = ped.PowerIter(J, Q, 1000, 1e-15, True)
    print eig(J.T)[0]
    print r.T
    print d.T

if case == 2:
    """
    test power iteration for multiple J
    """
    ped = pyPED()
    n = 4
    m = 30
    J = rand(m*n, n)
    Q = rand(n, n)
    
    q, r, d, c = ped.PowerIter(J, Q, 1000, 1e-15, True)
    JJ = eye(n)
    rr = eye(n)
    for i in range(m):
        JJ = dot(JJ, J[i*n:(i+1)*n, :].T)  # be cautious of the order
        rr = dot(rr, r[i*n:(i+1)*n, :].T)  # use dot for multiplication
    print eig(JJ)[0]
    print diag(rr)
    print d.T
