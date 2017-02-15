from personalFunctions import *
from py_ks import *

case = 30

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
    Plot rpo and ppo configuration in the full state space and 
    in the reduced space. Also plot without labels.
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    Ts = 100
    for poType in ['ppo', 'rpo']:
        for i in range(1, 4):
            a0, T, nstp, r, s = ksp.readPO(poFile, poType, i)
            h = T / nstp
            aa = ks.intg(a0, h, np.int(Ts/h), 5)
            raa, ths = ks.redSO2(aa, 1, False)
            name = 'ks' + poType + str(i) + 'T100'
            ksp.config(aa, [0, L, 0, Ts], axisLabelSize=25, tickSize=16, save=True, name=name)
            ksp.config(raa, [0, L, 0, Ts], axisLabelSize=25, tickSize=16,save=True, name=name+'Red')
            ksp.config(aa, [0, L, 0, Ts], axisLabelSize=25, tickSize=16, save=True, name=name+'NoLabel', 
                       labs=[None, None])
            ksp.config(raa, [0, L, 0, Ts], axisLabelSize=25, tickSize=16,save=True, name=name+'NoLabel'+'Red',
                       labs=[None, None])

