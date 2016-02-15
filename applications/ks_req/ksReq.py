from personalFunctions import *
from py_ks import *

case = 10

if case == 10:
    """
    test ks.stab() function
    """
    N = 32
    d = 22
    h = 0.1
    ks = pyKS(N, h, d)
    
    a0 = KSreadE('/usr/local/home/xiong/00git/research/data/ksReqx32.h5', 3)
    print norm(ks.velocity(a0))

    A = ks.stab(a0)
    e, v = eig(A.T)
    idx = argsort(e.real)[::-1]
    e = e[idx]
    v = v[:, idx]
    print e
    
