from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 20

if case == 10:
    """
    calculate the stability exponents of req
    of different di.
    """
    N = 1024
    d = 30
    h = 0.0002
    
    di = -0.05
    a0, wth0, wphi0, err = cqcglReadReq('req2.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, di, 4)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    print eigvalues[:10]

if case == 11:
    """
    calculate the stability exponents and 10 leading vectors
    save them into the database
    """
    N = 1024
    d = 30
    h = 0.0002
    
    # dis = [0.54, 0.55, 0.56, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67,
    #        0.69, 0.71, 0.74, 0.77, 0.8, 0.83]
    # files = ['req54.h5', 'req55.h5', 'req56.h5', 'req57.h5', 'req59.h5',
    #          'req61.h5', 'req63.h5', 'req65.h5', 'req67.h5', 'req69.h5',
    #          'req71.h5', 'req74.h5', 'req77.h5', 'req8.h5', 'req83.h5']
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
    h = 0.0002
    di = 0.38
    a0, wth0, wphi0, err = cqcglReadReq('req38.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    nstp = 10000
    for i in range(3):
        aa = cgl.intg(a0, nstp, 1)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])


if case == 20:
    """
    Try to locate the Hopf bifurcation limit cycle.
    There is singularity for reducing the discrete symmetry
    """
    N = 1024
    d = 30
    h = 0.0005
    di = 0.36
    a0, wth0, wphi0, err = cqcglReadReq('req36.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    # veTilde = cgl.reflectVe(veHat, a0Hat)
    
    nstp = 10000
    a0Erg = a0 + eigvectors[0]*1e-2
    for i in range(1):
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
 
if case == 30:
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

if case == 40:
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

if case == 50:
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

if case == 60:
    """
    Ater we find the rpo, we want to visualize it.
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

    e1, e2 = orthAxes2(veHat[0], veHat[1])

    x1, T1, nstp1, th1, phi1, err1 = cqcglReadRPO(
        '../../data/cgl/rpo/rpo0799T2X1.h5', '1')
    h1 = T1 / nstp1
    cgl2 = pyCqcgl1d(N, d, h1, False, 0, 4.0, 0.8, -0.01, di, 4)
    aa1 = cgl2.intg(x1[0], nstp1, 1)
    aa1Hat, th2, phi2 = cgl2.orbit2slice(aa1)
    aa1Hat -= a0Hat
    aa1HatProj = np.dot(aa1Hat, np.vstack((e1, e2)).T)
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    ax.plot(aa1HatProj[:, 0], aa1HatProj[:, 1], c='r', lw=1)
    ax.scatter([0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

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
    
