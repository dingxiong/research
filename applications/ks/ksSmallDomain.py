from ksHelp import *
from py_ks import *

case = 10

if case == 10:
    N, L = 64, 21.7
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    
    a0 = rand(N-2)
    a0 = ks.intgC(a0, 0.01, 100, 1000000)
    for i in range(10):
        aa = ks.intgC(a0, 0.01, 100, 10)
        a0 = aa[-1]
        # raa, ths = ks.redSO2(aa, 1, False)
        ksp.config(aa, [0, L, 0, 100])
    
if case == 20:
    """
    visulize rpo with different L
    """
    N = 64
    L0, dL = 21.95, 0.05
    ppType = 'rpo'
    isRPO = ppType == 'rpo'
    ppId = 2
    fileName = '../../data/ksh01x64.h5'
    
    Ts, ths = [], []
    for i in range(20):
        L = L0 - i * dL
        ks = pyKS(N, L)
        ksp = KSplot(ks)
        groupName = ksp.toStr(ppType, ppId, L, flag=1)
        if ksp.checkExist(fileName, groupName):
            a, T, nstp, theta, err = ksp.readPO(fileName, groupName, isRPO)[:5]
            Ts.append(T)
            ths.append(theta)
            if i%1 == 0:
                aa = ks.intgC(a, T/nstp, 4*T, 5)
                ksp.config(aa, [0, L, 0, 4*T])
    print Ts
    print ths

if case == 30:
    """
    visulize the state space for small L
    """
    N, L = 64, 21.95
    ks = pyKS(N, L)
    ksp = KSplot(ks)

    ppType = 'ppo'
    isRPO = ppType == 'rpo'
    ppId = 1
    
    a, T, nstp, theta, err = ksp.readPO('../../data/ksh01x64.h5', ksp.toStr(ppType, ppId, L, flag=1),
                                        isRPO)[:5]
    ap = ks.intgC(a, T/nstp, T, 10)
    apH = ks.redSO2(ap, 2, True)[0]

    aE = rand(N-2)*0.1
    aE = ks.intgC(aE, 0.01, 100, 100000)
    aa = ks.intgC(aE, 0.01, 1000, 10)
    aaH = ks.redSO2(aa, 2, True)[0]

    fig, ax = pl3d(size=[8, 6])
    ax.plot(apH[:, 1], apH[:, 5], apH[:, 3], c='r', lw=3)
    ax.plot(aaH[:, 1], aaH[:, 5], aaH[:, 3], c='b')
    ax3d(fig, ax)

if case == 40:
    """
    have a look at stability of rpo/ppo for different L
    """
    N = 64
    L0, dL = 21.95, 0.05
    ppType = 'rpo'
    isRPO = ppType == 'rpo'
    ppId = 2
    fileName = '../../data/ksh01x64EV.h5'
    
    Ls, Es, Ts = [], [], []
    for i in range(20):
        L = L0 - i * dL
        ksp = KSplot()
        groupName = ksp.toStr(ppType, ppId, L, flag=1)
        if ksp.checkExist(fileName, groupName):
            a, T, nstp, theta, err, e = ksp.readPO(fileName, groupName, isRPO, flag=1)[:6]
            Ts.append(T)
            Ls.append(L)
            Es.append(e)

    print Ts
    print Ls
    print Es
