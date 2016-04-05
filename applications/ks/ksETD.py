from py_ks import pyKSETD, pyKS
from personalFunctions import *

case = 1

if case == 1:
    N = 64
    L = 22
    ksetd = pyKSETD(N, L)
    
    poType = 'rpo'
    poId = 1
    a0, T, nstp, r, s = KSreadPO('../../data/ks22h001t120x64.h5', poType, poId)

    ksetd.setRtol(1e-10)
    tt, aa = ksetd.etd(a0, T, 0.01, 1, 2, True)
    hs = ksetd.hs()[1:]
    duu = ksetd.duu()[1:]
    plot1dfig(duu, yscale='log')
    plot1dfig(hs)
    print ksetd.etdParam()
