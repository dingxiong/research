from personalFunctions import *
from py_ks import *
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV


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
        # print tmp[1]
        
    eq = np.zeros((3, N-2))
    eqr = np.zeros((3, N-2))
    for i in range(3):
        a0, err = KSreadEq(fileName, i+1)
        eq[i] = a0
        tmp = ks.redO2(a0)
        eqr[i] = tmp[0]
        # print tmp[1]

    return req, ws, reqr, eq, eqr


def plotRE(ax, reqr, eqr, ii):
    c1 = ['r', 'b']
    for i in range(2):
        ax.scatter(reqr[i, ii[0]], reqr[i, ii[1]], reqr[i, ii[2]],
                   c=c1[i], s=70, edgecolors='none', label='TW'+str(i+1))
    c2 = ['c', 'k', 'y']
    for i in range(3):
        ax.scatter(eqr[i, ii[0]], eqr[i, ii[1]], eqr[i, ii[2]], c=c2[i], s=70,
                   edgecolors='none', label='E'+str(i+1))
    

def plotRE2d(ax, reqr, eqr, ii):
    c1 = ['r', 'b']
    for i in range(2):
        ax.scatter(reqr[i, ii[0]], reqr[i, ii[1]],
                   c=c1[i], s=70, edgecolors='none', label='TW'+str(i+1))
    c2 = ['c', 'k', 'y']
    for i in range(3):
        ax.scatter(eqr[i, ii[0]], eqr[i, ii[1]], c=c2[i], s=70,
                   edgecolors='none', label='E'+str(i+1))


def loadPO(fileName, poIds):
    aas = []
    types = ['rpo', 'ppo']
    for i in range(2):
        poType = types[i]
        for poId in poIds[i]:
            a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
            h = T / nstp
            ks = pyKS(N, h, 22)
            aa = ks.intg(a0, nstp, 5)
            aa = ks.redO2(aa)[0]
            aas.append(aa)
    
    return aas


def loadPO2(fileName, poIds, bases, x0):
    """
    aas : symmetry reduced state space
    pas : projected state space
    """
    aas = []
    pas = []
    types = ['rpo', 'ppo']
    for i in range(2):
        poType = types[i]
        for poId in poIds[i]:
            a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
            h = T / nstp
            ks = pyKS(N, 22)
            aa = ks.intg(a0, h, nstp, 5)
            aa = ks.redO2(aa)[0]
            aas.append(aa)
            pas.append(aa.dot(bases.T) - x0)
    
    return aas, pas


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


def getPoinc(data, theta):
    x, y = rotz(data[:, 0], data[:, 1], theta)
    aa = np.vstack((x, y, data[:, 2])).T
    n, m = aa.shape
    pc = np.zeros((n, m))
    pcf = np.zeros((n, m))
    num = 0
    for i in range(n-1):
        if aa[i, 0] < 0 and aa[i+1, 0] >= 0:
            p = int2p(aa[i], aa[i+1])
            pc[num] = p
            x, y = rotz(p[0], p[1], -theta)
            pcf[num] = np.array([x, y, p[2]])
            num += 1
    pc = pc[:num]
    pcf = pcf[:num]

    return pc, pcf


def getPoinc2(data, theta1, theta2):
    x1, y1 = rotz(data[:, 0], data[:, 1], theta1)
    x2, y2 = rotz(data[:, 0], data[:, 1], theta2)
    aa1 = np.vstack((x1, y1, data[:, 2])).T
    aa2 = np.vstack((x2, y2, data[:, 2])).T
    n, m = aa1.shape
    pc = np.zeros((n, m))
    pcf = np.zeros((n, m))
    coe = np.zeros(n)
    ixs = np.zeros(n, dtype=np.int)
    num = 0
    for i in range(n-1):
        if aa1[i, 0] < 0 and aa1[i+1, 0] >= 0:
            p, c1 = int2p(aa1[i], aa1[i+1])
            if p[1] >= 0:
                pc[num] = p
                coe[num] = c1
                ixs[num] = i
                x, y = rotz(p[0], p[1], -theta1)
                pcf[num] = np.array([x, y, p[2]])
                num += 1
        if aa2[i, 0] < 0 and aa2[i+1, 0] >= 0:
            p, c1 = int2p(aa2[i], aa2[i+1])
            if p[1] <= 0:
                pc[num] = p
                coe[num] = c1
                ixs[num] = i
                x, y = rotz(p[0], p[1], -theta2)
                pcf[num] = np.array([x, y, p[2]])
                num += 1
    pc = pc[:num]
    pcf = pcf[:num]
    coe = coe[:num]
    ixs = ixs[:num]

    return pc, pcf, coe, ixs


def ergoPoinc(ks, bases, x0, theta, si):
    a0 = rand(N-2) * 0.1
    aa = ks.intg(a0, 10000, 10000)
    poinc = np.zeros((0, 3))
    poincf = np.zeros((0, 3))
    paas = np.zeros((0, 3))
    for i in range(15):
        a0 = aa[-1]
        aa = ks.intg(a0, 850000, 100)[1:]
        raa, ths = ks.redO2(aa)
        paa = raa.dot(bases.T)
        paa -= x0
        pc, pcf = getPoinc(paa, theta)
        if si == 'p':
            ix = pc[:, 1] >= 0
            pc = pc[ix]
            pcf = pcf[ix]
        else:
            ix = pc[:, 1] <= 0
            pc = pc[ix]
            pcf = pcf[ix]
        poinc = np.vstack((poinc, pc))
        poincf = np.vstack((poincf, pcf))
        paas = np.vstack((paas, paa))
    return paas, poinc, poincf


def ergoPoinc2(ks, bases, x0, theta1, theta2):
    N = ks.N
    a0 = rand(N-2) * 0.1
    aa = ks.intg(a0, 10000, 10000)
    poinc = np.zeros((0, 3))
    poincf = np.zeros((0, 3))
    poincRaw = np.zeros((0, N-2))
    paas = np.zeros((0, 3))
    
    for i in range(15):
        a0 = aa[-1]
        aa = ks.intg(a0, 850000, 100)[1:]
        raa, ths = ks.redO2(aa)
        paa = raa.dot(bases.T)
        paa -= x0
        pc, pcf, coe, ixs = getPoinc2(paa, theta1, theta2)
        raw = (coe * raa[ixs].T + (1-coe) * raa[ixs+1].T).T
        poincRaw = np.vstack((poincRaw, raw))
        poinc = np.vstack((poinc, pc))
        poincf = np.vstack((poincf, pcf))
        paas = np.vstack((paas, paa))
    return paas, poinc, poincf, poincRaw

    
def poPoinc(fileName, poIds, bases, x0, theta1, theta2):
    """
    pas : projected orbits
    poinc : poincare intersection points on the plane
    poincf : intersection points in the 3d space
    """
    N = 64
    aas, pas = loadPO2(fileName, poIds, bases, x0)
    poinc = np.zeros((0, 3))
    poincf = np.zeros((0, 3))
    poincRaw = np.zeros((0, N-2))
    nums = []
    for i in range(len(pas)):
        pc, pcf, coe, ixs = getPoinc2(pas[i], theta1, theta2)
        nums.append(pc.shape[0])
        raw = (coe * aas[i][ixs].T + (1-coe) * aas[i][ixs+1].T).T
        poincRaw = np.vstack((poincRaw, raw))
        poinc = np.vstack((poinc, pc))
        poincf = np.vstack((poincf, pcf))

    return pas, poinc, poincf, poincRaw, np.array(nums)


def getCurveIndex(x, y):
    """
    for each row of x, try to find the corresponding
    row of y such that these two rows have the minimal
    distance.
    """
    m, n = xf.shape
    minDs = np.zeros(m)
    minIds = np.zeros(m, dtype=np.int)
    for i in range(m):
        dif = x[i]-y
        dis = norm(dif, axis=1)
        minDs[i] = np.min(dis)
        minIds[i] = np.argmin(dis)
    
    return minIds, minDs


def getCurveCoordinate(sortId, poinc):
    """
    get the curve coordinate
    """
    m = sortId.shape[0]
    dis = np.zeros(m)
    coor = np.zeros(m)
    for i in range(1, m):
        dis[i] = dis[i-1] + norm(poinc[sortId[i]]-poinc[sortId[i-1]])

    for i in range(m):
        coor[sortId[i]] = dis[i]

    return dis, coor

##############################################################################################################

case = 110

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

    x = rand(N-2)-0.5
    y = ks.Reflection(x)
    th0 = np.pi*2.47
    th1 = np.pi*4.22
    x1, t1 = ks.redO2(x)
    x2, t2 = ks.redO2(ks.Rotation(x, th0))
    y1, r1 = ks.redO2(y)
    y2, r2 = ks.redO2(ks.Rotation(y, th1))
    print norm(x2-x1), norm(y2-y1), norm(y2-x2)

if case == 30:
    """
    view the unstable manifold of E2
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

    k = 1
    # es, evt = KSstabReqEig(ks, req[k], ws[k])
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
        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        paas = []
        for i in range(len(aas)):
            paas.append(aas[i].dot(bases.T) - eqr[1])

        E3 = np.zeros((0, 3))
        for th in range(100):
            a = ks.Rotation(eq[2], th*np.pi/100)
            a = a.dot(bases.T) - eqr[1]
            E3 = np.vstack((E3, a))

        eqr -= eqr[1]
        ii = [0, 1, 2]
   
    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
    plotRE(ax, reqr, eqr, ii)
    if doProj:
        for i in range(nn):
            ax.plot(paas[i][:ns, ii[0]], paas[i][:ns, ii[1]],
                    paas[i][:ns, ii[2]], alpha=0.5)
        ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
    else:
        for i in range(nn):
            ax.plot(aas[i][:ns, ii[0]], aas[i][:ns, ii[1]], aas[i][:ns, ii[2]],
                    alpha=0.5)
    ax3d(fig, ax)

if case == 40:
    """
    watch an ergodic trajectory after reducing O2 symmetry
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

    a0 = rand(N-2) * 0.1
    aa = ks.intg(a0, 550000, 100)
    raa, ths = ks.redO2(aa)

    ii = [7, 8, 11]
 
    doProj = True
    if doProj:
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
        paas = raa.dot(bases.T)

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        paas -= eqr[1]
        reqr -= eqr[1]
        eqr -= eqr[1]

        ii = [0, 1, 2]
   
    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
    plotRE(ax, reqr, eqr, ii)
    if doProj:
        ax.plot(paas[:, ii[0]], paas[:, ii[1]],
                paas[:, ii[2]], alpha=0.5)
    else:
        ax.plot(raa[:, ii[0]], raa[:, ii[1]], raa[:, ii[2]],
                alpha=0.5)
    ax3d(fig, ax)

    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'],
                   xlim=[-1, 0.4], ylim=[-0.6, 0.6], zlim=[-0.15, 0.15],
                   isBlack=False)
    frame, = ax.plot([], [], [], c='gray', ls='-', lw=1, alpha=0.5)
    frame2, = ax.plot([], [], [], c='r', ls='-', lw=1.5, alpha=1)
    pts, = ax.plot([], [], [], 'co', lw=3)

    def anim(i):
        k = max(0, i-500)
        j = min(i, paas.shape[0])
        frame.set_data(paas[:k, ii[0]], paas[:k, ii[1]])
        frame.set_3d_properties(paas[:k, ii[2]])
        frame2.set_data(paas[k:j, ii[0]], paas[k:j, ii[1]])
        frame2.set_3d_properties(paas[k:j, ii[2]])
        pts.set_data(paas[j, ii[0]], paas[j, ii[1]])
        pts.set_3d_properties(paas[j, ii[2]])
        
        ax.view_init(30, 0.5 * i)
        return frame, frame2, pts

    ani = animation.FuncAnimation(fig, anim, frames=paas.shape[0],
                                  interval=0, blit=False, repeat=False)
    # ax3d(fig, ax)
    ax.legend()
    fig.tight_layout(pad=0)
    # ani.save('ani.mp4', dpi=200, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
    
if case == 50:
    """
    view a collection of rpo and ppo
    """
    N = 64
    L = 22
    h = 0.001
    ks = pyKS(N, h, L)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    poIds = [range(1, 50), range(1, 50)]
    aas = loadPO('../../data/ks22h001t120x64EV.h5', poIds)

    ii = [0, 3, 4]
    # ii = [7, 8, 11]

    doProj = True
    if doProj:
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
        # pev, bases = getBases(ks, 'eq', eq[0], [2, 3, 5])
        # pev, bases = getBases(ks, 'req', req[0], [0, 1, 3], ws[0])
        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        paas = []
        for i in range(len(aas)):
            paas.append(aas[i].dot(bases.T) - eqr[1])
        reqr -= eqr[1]
        eqr -= eqr[1]

        ii = [0, 1, 2]

    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
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
    ax3d(fig, ax)


if case == 60:
    """
    view rpo/ppo pair one at a time
    """
    N = 64
    L = 22
    h = 0.001
    ks = pyKS(N, h, L)

    for i in range(1, 20):
        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

        poIds = [[1] + range(i, i+1), range(i, i+1)]
        aas = loadPO('../../data/ks22h001t120x64EV.h5', poIds)

        ii = [0, 3, 4]

        doProj = True
        if doProj:
            pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
            # pev, bases = getBases(ks, 'eq', eq[0], [2, 3, 5])
            # pev, bases = getBases(ks, 'req', req[0], [0, 1, 3], ws[0])
            reqr = reqr.dot(bases.T)
            eqr = eqr.dot(bases.T)
            paas = []
            for i in range(len(aas)):
                paas.append(aas[i].dot(bases.T) - eqr[1])
            reqr -= eqr[1]
            eqr -= eqr[1]

            ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        plotRE(ax, reqr, eqr, ii)
        if doProj:
            for i in range(1, len(aas)):
                ax.plot(paas[i][:, ii[0]], paas[i][:, ii[1]],
                        paas[i][:, ii[2]],
                        alpha=0.8)
            ax.plot(paas[0][:, ii[0]], paas[0][:, ii[1]], paas[0][:, ii[2]],
                    c='k', ls='--',
                    label=r'$rpo_{16.31}$')
        else:
            for i in range(len(aas)):
                ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], aas[i][:, ii[2]],
                        alpha=0.7)
        ax3d(fig, ax, doBlock=True)

if case == 70:
    """
    construct poincare section in ergodic trajectory
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
 
    reqr = reqr.dot(bases.T)
    eqr = eqr.dot(bases.T)
    reqr -= eqr[1]
    x0 = eqr[1]

    paas, poinc, poincf = ergoPoinc(ks, bases, x0,  2*np.pi/6, 'n')
    eqr -= eqr[1]

    ii = [0, 1, 2]
   
    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
    plotRE(ax, reqr, eqr, ii)
    ax.plot(paas[:, ii[0]], paas[:, ii[1]],
            paas[:, ii[2]], alpha=0.5)
    ax.scatter(poincf[:, 0], poincf[:, 1], poincf[:, 2])
    ax3d(fig, ax)

    scatter2dfig(poinc[:, 1], poinc[:, 2], ratio='equal')

if case == 80:
    """
    construct poincare section with po
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
 
    reqr = reqr.dot(bases.T)
    eqr = eqr.dot(bases.T)
    reqr -= eqr[1]
    x0 = eqr[1]

    i = 40
    poIds = [range(1, i+1), range(1, i+1)]
    # poIds = [[], [2, 4, 8]]
    aas, poinc, nums = poPoinc('../../data/ks22h001t120x64EV.h5', poIds,
                               bases, x0,  0.5 * np.pi/6, 'p')
    eqr -= eqr[1]

    ii = [0, 1, 2]
   
    fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
    plotRE(ax, reqr, eqr, ii)
    for i in range(1, len(aas)):
        ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]],
                aas[i][:, ii[2]],
                alpha=0.2)
    ax.plot(aas[0][:, ii[0]], aas[0][:, ii[1]], aas[0][:, ii[2]],
            c='k', ls='--',
            label=r'$rpo_{16.31}$')
    ax.scatter(poinc[:, 0], poinc[:, 1], poinc[:, 2])
    ax3d(fig, ax)

    fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
    for i in range(1, len(aas)):
        ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]],
                alpha=0.2)
    ax.plot(aas[0][:, ii[0]], aas[0][:, ii[1]],
            c='k', ls='--',
            label=r'$rpo_{16.31}$')
    ax.scatter(poinc[:, 0], poinc[:, 1])
    ax3d(fig, ax)

    scatter2dfig(poinc[:, 1], poinc[:, 2], ratio='equal')
    plot1dfig(nums)


if case == 90:
    """
    construct poincare section in ergodic trajectory and
    try to  find the map
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
 
    reqr = reqr.dot(bases.T)
    eqr = eqr.dot(bases.T)
    reqr -= eqr[1]
    x0 = eqr[1].copy()
    eqr -= eqr[1]

    paas, poinc, poincf, poincRaw = ergoPoinc2(ks, bases, x0,
                                               2*np.pi/6, 2.0/3*np.pi/6)

    ii = [0, 1, 2]
   
    fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
    plotRE2d(ax, reqr, eqr, ii)
    ax.plot(paas[:, ii[0]], paas[:, ii[1]], c='b', alpha=0.5)
    ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
    ax2d(fig, ax)

    fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                   ratio='equal')
    ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
    ax2d(fig, ax)
    
    
if case == 100:
    """
    New version to get Poincare points from pos
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
 
    reqr = reqr.dot(bases.T)
    eqr = eqr.dot(bases.T)
    reqr -= eqr[1]
    x0 = eqr[1].copy()
    eqr -= eqr[1]

    i = 100
    poIds = [range(1, i+1), range(1, i+1)]
    aas, poinc, poincf, poincRaw, nums = poPoinc(
        '../../data/ks22h001t120x64EV.h5',
        poIds, bases, x0,  2*np.pi/6, 2.0/3*np.pi/6)
    ii = [0, 1, 2]
   
    fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
    plotRE2d(ax, reqr, eqr, ii)
    for i in range(len(aas)):
        ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], c='gray', alpha=0.2)
    ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
    ax2d(fig, ax)

    fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                   ratio='equal')
    ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
    ax2d(fig, ax)

    plot1dfig(nums)


if case == 110:
    """
    Get the return map from the Poincare section points
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, d)

    req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
    pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
 
    reqr = reqr.dot(bases.T)
    eqr = eqr.dot(bases.T)
    reqr -= eqr[1]
    x0 = eqr[1].copy()
    eqr -= eqr[1]

    i = 100
    poIds = [range(1, i+1), range(1, i+1)]
    aas, poinc, poincf, poincRaw, nums = poPoinc(
        '../../data/ks22h001t120x64EV.h5', poIds,
        bases, x0,  2*np.pi/6, 2.0/3*np.pi/6)
    ii = [0, 1, 2]
    
    fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
    plotRE2d(ax, reqr, eqr, ii)
    for i in range(len(aas)):
        ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], c='gray', alpha=0.2)
    ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
    ax2d(fig, ax)
    
    fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                   ratio='equal')
    ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
    ax2d(fig, ax)

    plot1dfig(nums)
    
    xf = poinc[:, 1:]
    sel = xf[:, 0] > 0
    # xf = xf[sel]
    # poincRaw = poincRaw[sel]
    scale = 10
    nps = 5000
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5),
                                   "degree": [3]})
    
    svr.fit(xf[:, 0:1], xf[:, 1]*scale)
    xp = linspace(0.43, -0.3, nps)  # start form right side
    xpp = xp.reshape(nps, 1)
    yp = svr.predict(xpp)/scale
    fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                   ratio='equal')
    ax.scatter(poinc[:, 1], poinc[:, 2], c='r', s=10, edgecolors='none')
    ax.plot(xp, yp, c='g', ls='-', lw=2)
    ax2d(fig, ax)

    curve = np.zeros((nps, 2))
    curve[:, 0] = xp
    curve[:, 1] = yp
    minIds, minDs = getCurveIndex(xf, curve)
    sortId = np.argsort(minIds)
   
    dis, coor = getCurveCoordinate(sortId, poincRaw)
    fig, ax = pl2d(size=[6, 4], labs=[r'$S_n$', r'$S_{n+1}$'],
                   axisLabelSize=15)
    ax.scatter(coor[:-1], coor[1:], c='r', s=10, edgecolors='none')
    ax2d(fig, ax)

