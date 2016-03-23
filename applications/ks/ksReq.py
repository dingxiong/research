from personalFunctions import *
from py_ks import *

case = 40

if case == 10:
    """
    test ks.stab() function
    """
    N = 32
    d = 22
    h = 0.1
    ks = pyKS(N, h, d)
    
    a0 = KSreadEq('/usr/local/home/xiong/00git/research/data/ksReqx32.h5', 3)
    print norm(ks.velocity(a0))

    A = ks.stab(a0)
    e, v = eig(A.T)
    idx = argsort(e.real)[::-1]
    e = e[idx]
    v = v[:, idx]
    print e


if case == 20:
    """
    Have a look at the flow in the state space
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)
    
    a0, w, err = KSreadReq('../../data/ks22Reqx64.h5', 1)
    es, vs = KSstabReqEig(ks, a0, w)
    # aa = ks.intg(a0 + 1e-1*vs[0].real, 200000, 100)
    aa = ks.intg(rand(N-2)*0.1, 200000, 100)
    aaH = ks.orbitToSlice(aa)[0]
    # plot3dfig(aa[:, 0], aa[:, 3], aa[:, 2])
    plot3dfig(aaH[:, 0], aaH[:, 3], aaH[:, 2])
    raa, ths = rSO3(ks, aa)
    plot3dfig(raa[:, 0], raa[:, 3], raa[:, 2])


if case == 25:
    """
    test the 2nd slice
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    x = rand(N-2)
    y = ks.Reflection(x)
    t0 = np.pi*2.47
    x1, t1 = ks.redO2(x)
    x2, t2 = ks.redO2(ks.Rotation(x, t0))
    y1, r1 = ks.redO2(y)
    y2, r2 = ks.redO2(ks.Rotation(y, t0))
    print (t2-t0-t1) / (np.pi/3), norm(x2-x1), norm(y2-y1), norm(y2-x2)


def loadRE(fileName, N):
    req = np.zeros((2, N-2))
    ws = np.zeros(2)
    reqr = np.zeros((2, N-2))
    for i in range(2):
        a0, w, err = KSreadReq(fileName, i+1)
        req[i] = a0
        ws[i] = w
        tmp = ks.redO2(a0)
        reqr[i] = tmp[0]
        print tmp[1]
        
    eq = np.zeros((3, N-2))
    eqr = np.zeros((3, N-2))
    for i in range(3):
        a0, err = KSreadEq(fileName, i+1)
        eq[i] = a0
        tmp = ks.redO2(a0)
        eqr[i] = tmp[0]
        print tmp[1]

    return req, ws, reqr, eq, eqr


def loadPO(fileName, types, poIds):
    aas = []
    for poType in types:
        for poId in poIds:
            a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
            h = T / nstp
            ks = pyKS(N, h, 22)
            aa = ks.intg(a0, nstp, 5)
            aa = ks.redO2(aa)[0]
            aas.append(aa)
    
    return aas


def plotRE(ax, reqr, eqr, ii):
    c1 = ['r', 'b']
    for i in range(2):
        ax.scatter(reqr[i, ii[0]], reqr[i, ii[1]], reqr[i, ii[2]],
                   c=c1[i], s=70, edgecolors='none', label='TW'+str(i+1))
    c2 = ['c', 'k', 'y']
    for i in range(3):
        ax.scatter(eqr[i, ii[0]], eqr[i, ii[1]], eqr[i, ii[2]], c=c2[i], s=70,
                   edgecolors='none', label='E'+str(i+1))
    

def getBases(ks, etype, a, ii, w=0):

    if etype == 'eq':
        es, evt = KSstabEig(ks, a)
        ev = Tcopy(realve(evt))
        pev = ks.redV2(ev, a)
        v1, v2, v3 = orthAxes(pev[ii[0]], pev[ii[1]], pev[ii[2]])
        bases = np.vstack((v1, v2, v3))

    if etype == 'req':
        es, evt = KSstabReqEig(ks, a, w)
        ev = Tcopy(realve(evt))
        pev = ks.redV2(ev, a)
        v1, v2, v3 = orthAxes(pev[ii[0]], pev[ii[1]], pev[ii[2]])
        bases = np.vstack((v1, v2, v3))

    return pev, bases

if case == 30:
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

    k = 1
    es, evt = KSstabEig(ks, eq[k])
    ev = Tcopy(realve(evt))
    aas = []
    nn = 30
    for i in range(nn):
        a0 = eq[k] + 1e-5 * (i+1) * ev[0]
        # a0 = rand(N-2) * 0.1
        aa = ks.intg(a0, 150000, 100)
        raa, ths = ks.redO2(aa)
        aas.append(raa)

    ns = -1
    ii = [7, 8, 11]
 
    doProj = True
    if doProj:
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
        paas = []
        for i in range(len(aas)):
            paas.append(aas[i].dot(bases.T))

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)

        ii = [0, 1, 2]
   
    fig, ax = pl3d()
    plotRE(ax, reqr, eqr, ii)
    if doProj:
        for i in range(nn):
            ax.plot(paas[i][:ns, ii[0]], paas[i][:ns, ii[1]],
                    paas[i][:ns, ii[2]], alpha=0.5)
    else:
        for i in range(nn):
            ax.plot(aas[i][:ns, ii[0]], aas[i][:ns, ii[1]], aas[i][:ns, ii[2]],
                    alpha=0.5)
    ax3d(fig, ax, labs=[r'$v_1$', r'$v_2$', r'$v_3$'])


if case == 40:
    N = 64
    L = 22
    h = 0.001
    ks = pyKS(N, h, L)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    types = ['rpo', 'ppo']
    poIds = range(1, 51)
    aas = loadPO('../../data/ks22h001t120x64EV.h5', types, poIds)

    ii = [0, 3, 4]
    # ii = [7, 8, 11]

    doProj = True
    if doProj:
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
        # pev, bases = getBases(ks, 'eq', eq[0], [2, 3, 5])
        # pev, bases = getBases(ks, 'req', req[0], [0, 1, 3], ws[0])

        paas = []
        for i in range(len(aas)):
            paas.append(aas[i].dot(bases.T))

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)

        ii = [0, 1, 2]

    fig, ax = pl3d()
    plotRE(ax, reqr, eqr, ii)
    if doProj:
        for i in range(len(aas)):
            ax.plot(paas[i][:, ii[0]], paas[i][:, ii[1]], paas[i][:, ii[2]],
                    alpha=0.2)
        ax.plot(paas[0][:, ii[0]], paas[0][:, ii[1]], paas[0][:, ii[2]], c='k',
                label=r'$rpo_{16.31}$')
    else:
        for i in range(len(aas)):
            ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], aas[i][:, ii[2]],
                    alpha=0.2)
    ax3d(fig, ax, labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
