from py_CQCGL1d import *
from personalFunctions import *
from sets import Set

case = 16

if case == 6:
    N = 1024
    d = 50
    h = 2e-3
    di = 0.4

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, 3, -0.1, -3.9, -1)
    req = CQCGLreq(cgl)

    a0, wth0, wphi0, err0 = req.readReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    a0 = a0 * (0.1**0.5)
    nstp = 20000
    for i in range(5):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

if case == 8:
    """
    calculate the stability exponents of req
    of different di.
    At the same time, compare it with the linear part of
    the velocity
    """
    N = 1024
    d = 30
    di = 0.06

    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5', di, 1)
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    print eigvalues[:10]
    
    e = eigvalues
    L0 = cgl.L()
    L1 = cgl.C2R(L0)
    L2 = L1[::2] + 1j*L1[1::2]
    L2 = 1 / L2
    L3 = np.zeros(cgl.Ndim)
    L3[::2] = L2.real
    L3[1::2] = L2.imag
    L = cgl.L()[:cgl.Ne/2]
    scatter2dfig(L.real, L.imag)
    scatter2dfig(e.real, e.imag)

    def dp(A, L):
        n = len(L)
        for i in range(n):
            A[:, i] = A[:, i] * L[i]
        return A
    
    A = cgl.stabReq(a0, wth0, wphi0).T
    Ap = dp(A, L3)
    e2, v2 = eig(Ap)
    e2, v2 = sortByReal(e2, v2)
    

if case == 10:
    """
    use the new data to calculate tyhe stability exponents of req
    """
    N = 1024
    d = 50
    Bi = 0.8
    Gi = -0.6
    index = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)

    a0, wth0, wphi0, err0 = req.readReqBiGi('../../data/cgl/reqBiGi.h5',
                                            Bi, Gi, index)
    e, v = req.eigReq(a0, wth0, wphi0)
    print e[:10]

if case == 11:
    """
    calculate the stability exponents and 10 leading vectors
    save them into the database
    """
    N = 1024
    d = 30
    h = 0.0002
    
    for i in range(len(dis)):
        a0, wth0, wphi0, err = cqcglReadReq(files[i], '1')
        cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, dis[i], 4)
        e, v = eigReq(cgl, a0, wth0, wphi0)
        print e[:8]
        cqcglAddEV2Req(files[i], '1', e.real, e.imag,
                       v[:, :10].real, v[:, :10].imag)

if case == 12:
    """
    simply view the heat map of relative equilibria
    """
    N = 1024
    d = 30
    h = 1e-4
    di = 0.96

    a0, wth0, wphi0, err0 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                           di, 2)
    cgl = pyCQCGL1d(N, d, 4.0, 0.8, 0.01, di, -1)

    nstp = 10000
    for i in range(3):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])


if case == 13:
    """
    new size L = 50
    """
    N = 1024
    d = 50
    h = 1e-4
    di = 0.96

    a0, wth0, wphi0, err0 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                           di, 1)
    a1, wth1, wphi1, err1 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                           di, 2)
    cgl = pyCQCGL1d(N, d, 4.0, 0.8, 0.01, di, -1)
    plotOneConfigFromFourier(cgl, a0)
    plotOneConfigFromFourier(cgl, a1)
    
if case == 14:
    N = 1024
    d = 30
    h = 1e-4

    Q0 = []
    Q1 = []
    dis = np.arange(0.08, 0.57, 0.01)
    for i in range(len(dis)):
        di = dis[i]
        cgl = pyCQCGL1d(N, d, 4.0, 0.8, 0.01, di, -1)
        a0, wth0, wphi0, err0 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                               di, 1)
        a1, wth1, wphi1, err1 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                               di, 2)
        Q0.append(cgl.calQ(a0)[0])
        Q1.append(cgl.calQ(a1)[0])
    
    fig, ax = pl2d(labs=[r'$d_i$', r'$Q$'], axisLabelSize=25, tickSize=15)
    ax.plot(dis, Q0, lw=2, c='r', ls='-')
    ax.plot(dis, Q1, lw=2, c='b', ls='--')
    ax2d(fig, ax)

if case == 15:
    """
    to see different profiles
    """
    N = 1024
    d = 30
    h = 1e-3
    di = 0.10
    c = -4.1158

    cgl = pyCQCGL1d(N, d, 4.0, c, 0.01, di, -1)
    req = CQCGLreq(cgl)
    a0, wth0, wphi0, err0 = req.readReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    nstp = 10000
    for i in range(3):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

if case == 16:
    """
    to see different profiles with L = 50
    """
    N = 1024
    d = 50
    h = 2e-3

    Bi = 4.7
    Gi = -4.6
    index = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    req = CQCGLreq()

    a0, wth0, wphi0, err0, e, v = req.readReqBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, index, flag=2)
    print e[:20]
    
    nstp = 20000
    a0 += 0.1*norm(a0)*v[0].real
    for i in range(3):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

if case == 17:
    """
    view the req with new L
    """
    N = 1024
    d = 50
    h = 2e-3
    
    Bi = 2.0
    Gi = -5.6
    index = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)

    a0, wth0, wphi0, err0, e, v = req.readReqBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, index, flag=2)
    plotOneConfigFromFourier(cgl, a0)
    print e[:12]
    
if case == 18:
    """
    Plot the solitons in the Bi-Gi plane to see the transition.
    """
    N = 1024
    d = 50
    h = 2e-3
    
    fileName = '../../data/cgl/reqBiGi.h5'
    
    for i in range(90):
        Bi = -3.2 + i * 0.1
        fig, ax = pl2d(size=[8, 6], labs=[r'$x$', r'$|A|$'], ylim=[-0.5, 3], axisLabelSize=25)
        name = format(Bi, '013.6f') + '.png'
        for j in range(55):
            Gi = -0.2 - 0.1*j
            cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
            req = CQCGLreq(cgl)
            if req.checkExist(fileName, req.toStr(Bi, Gi, 1)):
                a0, wth0, wphi0, err0 = req.readReqBiGi(fileName, Bi, Gi, 1)
                Aamp = np.abs(cgl.Fourier2Config(a0))
                ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp)
            if req.checkExist(fileName, req.toStr(Bi, Gi, 2)):
                a0, wth0, wphi0, err0 = req.readReqBiGi(fileName, Bi, Gi, 2)
                Aamp = np.abs(cgl.Fourier2Config(a0))
                ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp)
        ax2d(fig, ax, save=True, name='ex/'+name)
        

if case == 19:
    """
    plot the stability of req in the Bi-Gi plane
    """
    N = 1024
    d = 50
    h = 2e-3
    
    fileName = '../../data/cgl/reqBiGiEV.h5'
    index = 1
    
    cs = {0: 'g', 1: 'm', 2: 'c', 4: 'r', 6: 'b'}#, 8: 'y', 56: 'r', 14: 'grey'}
    ms = {0: 's', 1: '^', 2: '+', 4: 'o', 6: 'D'}#, 8: '8', 56: '4', 14: 'v'}

    es = np.zeros([90, 55, 1362], dtype=np.complex)
    e1 = np.zeros((90, 55), dtype=np.complex)
    ns = Set([])

    fig, ax = pl2d(size=[8, 6], labs=[r'$G_i$', r'$B_i$'], axisLabelSize=25,
                   xlim=[-6, 0], ylim=[-4, 6])
    for i in range(90):
        Bi = 5.7 - i*0.1
        for j in range(55):
            Gi = -0.2 - 0.1*j
            req = CQCGLreq()
            if req.checkExist(fileName, req.toStr(Bi, Gi, index) + '/vr'):
                a0, wth0, wphi0, err0, e, v = req.readReqBiGi(fileName, Bi, Gi, index, flag=2)
                es[i, j, :] = e
                m, ep = req.numStab(e)
                e1[i, j] = ep[0]
                ns.add(m)
                # print index, Bi, Gi, m
                ax.scatter(Gi, Bi, s=15, edgecolors='none',
                           marker=ms.get(m, '*'), c=cs.get(m, 'k'))
    
    ax.grid(which='both')
    ax2d(fig, ax)

if case == 20:
    """
    check req really converges
    """
    fileName = '../../data/cgl/reqBiGiEV.h5'
    index = 2

    for i in range(90):
        Bi = 3.4 - i*0.1
        for j in range(55):
            Gi = -0.2 - 0.1*j
            req = CQCGLreq()
            if req.checkExist(fileName, req.toStr(Bi, Gi, index)):
                a0, wth0, wphi0, err0= req.readReqBiGi(fileName, Bi, Gi, index)
                if err0 > 1e-6:
                    print err0, Bi, Gi
    
if case == 30:
    """
    Try to locate the Hopf bifurcation limit cycle.
    There is singularity for reducing the discrete symmetry
    """
    N = 1024
    d = 30
    h = 0.0005
    di = 0.37
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    # veTilde = cgl.reflectVe(veHat, a0Hat)
    
    nstp = 10000
    # a0Erg = a0 + eigvectors[0]*1e2
    A0 = 2*centerRand(2*N, 0.2)
    a0Erg = cgl.Config2Fourier(A0)
    for i in range(2):
        aaErg = cgl.intg(a0Erg, nstp, 1)
        a0Erg = aaErg[-1]

    aaErgHat, th, phi = cgl.orbit2slice(aaErg)
    # aaErgTilde = cgl.reduceReflection(aaErgHat)
    # aaErgTilde -= a0Tilde
    aaErgHat -= a0Hat

    # e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])
    e1, e2 = orthAxes2(veHat[0], veHat[1])
    # aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2)).T)
    
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    # ax.plot(aaErgTildeProj[:, 0], aaErgTildeProj[:, 1],
    #         aaErgTildeProj[:, 2], c='r', lw=1)
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            c='r', lw=1)
    ax.scatter(aaErgHatProj[0, 0], aaErgHatProj[0, 1], s=30, facecolor='g')
    ax.scatter(aaErgHatProj[-1, 0], aaErgHatProj[-1, 1], s=30, facecolor='k')
    ax.scatter([0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
 
if case == 31:
    """
    Try to find the guess of the limit cycle for di large enough
    such that the soliton is unstable.
    """
    N = 1024
    d = 30
    h = 0.0004
    di = 0.43

    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    A0 = 2*centerRand(2*N, 0.2)
    a0 = cgl.Config2Fourier(A0)
    nstp = 10000
    x = []
    for i in range(3):
        aa = cgl.intg(a0, nstp, 1)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])
        # plotPhase(cgl, aa, [0, d, 0, nstp*h])
        # plotOneConfigFromFourier(cgl, aa[-1], d)
        # plotOnePhase(cgl, aa[-1], d)
        # plot1dfig(aa[:, 0])
        x.append(aa)
        
    aaHat, th, phi = cgl.orbit2slice(aa)
    i1 = 3000
    i2 = 6800
    plot3dfig(aaHat[i1:i2, 0], aaHat[i1:i2, 1], aaHat[i1:i2, 2])
    plotConfigSpaceFromFourier(cgl, aa[i1:i2], [0, d, 0, nstp*h])
    nstp = i2 - i1
    T = nstp * h
    th0 = th[i1] - th[i2]
    phi0 = phi[i1] - phi[i2]
    err = norm(aaHat[i1] - aaHat[i2])
    print nstp, T, th0, phi0, err
    # cqcglSaveRPOdi('rpot.h5', di, 1, aa[i1], T, nstp, th0, phi0, err)
    
if case == 40:
    """
    compare the Hopf bifurcation limit cycle with slightly different di
    """
    N = 1024
    d = 30
    h = 0.0002

    dis = [-0.07985, -0.0799, -0.08]
    files = ['req07985.h5', 'req0799.h5', 'req08.h5']
    
    orbits = []
    for i in range(len(dis)):
        a0, wth0, wphi0, err = cqcglReadReq(files[i], '1')
        cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, dis[i], 4)

        eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
        print eigvalues[:8]
        eigvectors = Tcopy(realve(eigvectors))
        a0Hat = cgl.orbit2slice(a0)[0]
        veHat = cgl.ve2slice(eigvectors, a0)
        
        nstp = 20000
        a0Erg = a0 + eigvectors[0]*1e-3
        for i in range(10):
            aaErg = cgl.intg(a0Erg, nstp, 1)
            a0Erg = aaErg[-1]

        aaErgHat, th, phi = cgl.orbit2slice(aaErg)
        aaErgHat -= a0Hat
        e1, e2 = orthAxes2(veHat[0], veHat[1])
        aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2)).T)
        orbits.append(aaErgHatProj)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    for i in range(len(orbits)):
        ax.plot(orbits[i][:, 0], orbits[i][:, 1], lw=1,
                label=str(dis[i]))
    ax.scatter([0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.legend(loc='upper right', frameon=False)
    plt.show(block=False)

if case == 50:
    """
    visualize a few explosion examples in the covariant coordinate.
    Only care about the part after explosion.
    """
    N = 1024
    d = 30
    h = 0.0002

    di = -0.0799
    a0, wth0, wphi0, err = cqcglReadReq('req0799.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)
    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[4])

    aa1 = load('0799_v6.npz')['x']
    aa1Hat, th1, phi1 = cgl.orbit2slice(aa1)
    aa1Hat -= a0Hat
    aa1HatProj = np.dot(aa1Hat, np.vstack((e1, e2, e3)).T)

    plotConfigSpaceFromFourier(cgl, aa1, [0, d, 0, aa1.shape[0]*h])
    
    id1 = 4000
    id2 = aa1.shape[0] - 12000

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    ax.plot(aa1HatProj[id1:id2, 0], aa1HatProj[id1:id2, 1],
            c='r', lw=1)
    ax.scatter(aa1HatProj[id1, 0], aa1HatProj[id1, 1], s=30)
    ax.scatter([0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aa1HatProj[id1:id2, 0], aa1HatProj[id1:id2, 1],
            aa1HatProj[id1:id2, 2], c='r', lw=1)
    ax.scatter(aa1HatProj[id1, 0], aa1HatProj[id1, 1],
               aa1HatProj[id1, 2], s=30)
    ax.scatter([0], [0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 60:
    """
    In the case that we visualize Hopf limit cycle and the explosion in the
    same frame, we find that after explosion, the system almost lands
    in a closed cycle. This is a good candidate for search periodic orbits.
    So, we record it.
    """
    N = 1024
    d = 30
    h = 0.0002

    di = -0.0799
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, di, 4)

    aa1 = load('0799_v4.npz')['x']
    aa1Hat, th1, phi1 = cgl.orbit2slice(aa1)
    i1 = 0
    i2 = 1
    i3 = 2

    id1 = 19810
    id2 = 20000
    plot3dfig(aa1Hat[id1:id2+1, i1], aa1Hat[id1:id2+1, i2],
              aa1Hat[id1:id2+1, i3])

    x = aa1[id1]
    nstp = id2 - id1
    T = nstp * h
    th = th1[id1] - th1[id2]
    phi = phi1[id1] - phi1[id2]
    err = 1000.0
    print T, th, phi, err
    cqcglSaveRPO('rpo3.h5', '1', x, T, nstp, th, phi, err)

if case == 70:
    """
    construct the plane wave solutions and get their stability
    """
    N = 1024
    L = 30
    h = 0.0002
    b = 4.0
    c = 0.8
    dr = 0.01
    di = 0.07985

    cgl = pyCqcgl1d(N, L, h, True, 0, b, c, dr, di, 4)

    m = 14
    k = 2*np.pi / L * m
    a2 = 1/(2*dr) * (1 + np.sqrt(1-4*dr*(k**2+1)))
    w = b*k**2 - c*a2 + di*a2**2

    a0 = np.zeros(cgl.Ndim)
    a0[2*m] = sqrt(a2) * N
    nstp = 5000
    aa = cgl.intg(a0, nstp, 1)
    plotConfigSpaceFromFourier(cgl, aa, [0, L, 0, nstp*h])

    eigvalues, eigvectors = eigReq(cgl, a0, 0, w)
    eigvectors = Tcopy(realve(eigvectors))
    print eigvalues[:20]
    

if case == 80:
    """
    plot the figure show the stability of solition solutions for
    different di
    """
    N = 1024
    d = 30
    
    dis, reqs = cqcglReadReqEVAll('../../data/cgl/reqDi.h5', 1)
    ers = []
    for i in range(len(reqs)):
        a, wth, wphi, err, er, ei, vr, vi = reqs[i]
        k = 0
        while abs(er[k]) < 1e-8:
            k += 1
        ers.append(er[k])
        
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    ax.plot([min(dis), max(dis)], [0, 0], c='r', ls='--', lw=1.5)
    ax.scatter(dis, ers, s=10, marker='o', facecolor='b', edgecolors='none')
    ax.set_xlabel(r'$d_i$', fontsize=20)
    ax.set_ylabel(r'$\mu$', fontsize=20)
    # ax.set_yscale('log')
    ax.set_xlim(0, 1)
    # ax.set_ylim(-2, 18)
    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 90:
    """
    view the unstable manifold of the req in the symmetry reduced projected
    coordinates
    """
    N = 1024
    d = 30
    di = 0.06
   
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)
    cgl.changeOmega(-wphi0)
    cgl.rtol = 1e-10
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, 0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)

    T = 20
    
    a0Erg = a0 + eigvectors[0]*1e-3
    aaErg = cgl.aintg(a0Erg, 0.001, T, 1)
    aaErgHat, th, th = cgl.orbit2slice(aaErg)
    aaErgHat -= a0Hat
    
    a0E2 = a0 + eigvectors[2]*1e-3
    aaE2 = cgl.aintg(a0E2, 0.001, T, 1)
    aaE2H, th2, phi2 = cgl.orbit2slice(aaE2)
    aaE2H -= a0Hat

    # e1, e2 = orthAxes2(veHat[0], veHat[1])
    e1, e2, e3 = orthAxes(veHat[3], veHat[2], veHat[6])
    bases = np.vstack((e1, e2, e3)).T
    aaErgHatProj = np.dot(aaErgHat, bases)
    OProj = np.dot(-a0Hat, bases)
    
    aaE2HP = np.dot(aaE2H, bases)

    i1 = 32000
    i2 = 38000
    fig, ax = pl3d(labs=[r'$e_1$', r'$e_2$', r'$e_3$'],
                   axisLabelSize=25)
    #ax.plot(aaErgHatProj[i1:i2, 0], aaErgHatProj[i1:i2, 1],
    #        aaErgHatProj[i1:i2, 2], c='g', lw=1, alpha=0.4)
    ax.plot(aaE2HP[i1:i2, 0], aaE2HP[i1:i2, 1],
            aaE2HP[i1:i2, 2], c='r', lw=1, alpha=0.4)
    ax.plot(aaE2HP[:9000, 0], aaE2HP[:9000, 1],
            aaE2HP[:9000, 2], c='g', lw=1, alpha=0.4)
    ax.plot(aaErgHatProj[:11000, 0], aaErgHatProj[:11000, 1],
            aaErgHatProj[:11000, 2], c='y', lw=1, alpha=1)
    ax.scatter([0], [0], [0], s=80, marker='o', c='b',  edgecolors='none')
    ax.scatter(OProj[0], OProj[1], OProj[2], s=60, marker='o', c='c',
               edgecolors='none')
    ax3d(fig, ax)

    plotConfigSpaceFromFourier(cgl, aaErg[::10].copy(), [0, d, 0, T])
    plotConfigSpaceFromFourier(cgl, aaE2[::10].copy(), [0, d, 0, T])

    # vel = []
    # for i in range(-10000, -000):
    #     vel.append(norm(cgl.velocity(aaErg[i])))

if case == 100:
    """
    view the 2 soliton solution together, and
    see how the unstable manifold of the first
    hits the second solition.
    """
    N = 1024
    d = 30
    h = 0.0002

    di = 0.4227
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    a1, wth1, wphi1, err1 = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                           di, 2)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)
    a1Hat = cgl.orbit2slice(a1)[0]
    a1Hat -= a0Hat

    h3 = 0.0005
    cgl3 = pyCqcgl1d(N, d, h3, False, 0, 4.0, 0.8, 0.01, di, 4)
    nstp = 70000
    a0Erg = a0 + eigvectors[0]*1e-3
    aaErg = cgl3.intg(a0Erg, 50000, 50000)
    a0Erg = aaErg[-1]
    aaErg = cgl3.intg(a0Erg, nstp, 2)
    aaErgHat, th3, th3 = cgl3.orbit2slice(aaErg)
    aaErgHat -= a0Hat
    
    # e1, e2 = orthAxes2(veHat[0], veHat[1])
    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[6])
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2, e3)).T)
    OProj = np.dot(-a0Hat, np.vstack((e1, e2, e3)).T)
    a1HatProj = np.dot(a1Hat, np.vstack((e1, e2, e3)).T)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            aaErgHatProj[:, 2], c='g', lw=1, alpha=0.4)
    # i1 = -1000
    # ax.plot(aaErgHatProj[i1:, 0], aaErgHatProj[i1:, 1],
    #         aaErgHatProj[i1:, 2], c='k', lw=2)
    ax.scatter([0], [0], [0], s=80, marker='o', c='b',  edgecolors='none')
    ax.scatter(OProj[0], OProj[1], OProj[2], s=60, marker='o', c='c',
               edgecolors='none')
    ax.scatter(a1HatProj[0], a1HatProj[1], a1HatProj[2],
               s=60, marker='o', c='m', edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    # plotConfigSurfaceFourier(cgl, aa1, [0, d, 0, T1])

    # vel = []
    # for i in range(-10000, -000):
    #     vel.append(norm(cgl.velocity(aaErg[i])))

