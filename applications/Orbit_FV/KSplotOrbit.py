from personalFunctions import *
from py_ks import *

case = 20

N, L = 64, 22
eqFile =  '../../data/ks22Reqx64.h5'
poFile = '../../data/ks22h001t120x64EV.h5'

if case == 10:
    """
    plot the configuration profile of eq and req
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    for i in range(1, 4):
        a, err = ksp.readEq(eqFile, i)
        ksp.oneConfig(a, axisLabelSize=25, tickSize=18)
    for i in range(1, 3):
        a, w, err = ksp.readReq(eqFile, i)
        ksp.oneConfig(a, axisLabelSize=25, tickSize=18)
    
if case == 20:
    """
    plot the 2 req with time and the symmetry reduced figues 
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    h = 2e-3
    T = 100
    for i in range(1, 3):
        a, w, err = ksp.readReq(eqFile, i)
        aa = ks.intg(a, h, np.int(T/h), 5)
        raa, ths = ks.redSO2(aa, 1, False)
        ksp.config(aa, [0, L, 0, T], axisLabelSize=25, tickSize=16)
        ksp.config(raa, [0, L, 0, T], axisLabelSize=25, tickSize=16)
    

if case == 30:
    """
    Plot rpo and ppo configuration
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    for i in range(1, 4):
        ksp.oneConfig(a, axisLabelSize=25, tickSize=18)
