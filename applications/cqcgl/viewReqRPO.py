from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 10

if case == 5:
    """
    View the req and rpo for the same di.
    2d version
    """
    N = 1024
    d = 30
    h = 0.0002

    di = 0.422
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)

    e1, e2 = orthAxes2(veHat[0], veHat[1])
    
    x1, T1, nstp1, th1, phi1, err1 = cqcglReadRPOdi(
        '../../data/cgl/rpoT2X1.h5', di, 1)
    h1 = T1 / nstp1
    nstp1 = np.int(nstp1)
    cgl2 = pyCqcgl1d(N, d, h1, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa1 = cgl2.intg(x1, nstp1, 1)
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

    # plotConfigSurfaceFourier(cgl, aa1, [0, d, 0, T1])

if case == 6:
    """
    View origin, req and rpo for the same di.
    3d version
    """
    N = 1024
    d = 30
    h = 0.0002

    di = 0.41
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)

    x1, T1, nstp1, th1, phi1, err1 = cqcglReadRPOdi(
        '../../data/cgl/rpoT2X1.h5', di, 1)
    h1 = T1 / nstp1
    nstp1 = np.int(nstp1)
    cgl2 = pyCqcgl1d(N, d, h1, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa1 = cgl2.intg(x1, nstp1, 1)
    aa1Hat, th2, phi2 = cgl2.orbit2slice(aa1)
    aa1Hat -= a0Hat
    
    # e1, e2 = orthAxes2(veHat[0], veHat[1])
    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[6])
    aa1HatProj = np.dot(aa1Hat, np.vstack((e1, e2, e3)).T)
    OProj = np.dot(-a0Hat, np.vstack((e1, e2, e3)).T)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aa1HatProj[:, 0], aa1HatProj[:, 1], aa1HatProj[:, 2], c='r', lw=2)
    ax.scatter([0], [0], [0], s=80, marker='o', c='b',  edgecolors='none')
    ax.scatter(OProj[0], OProj[1], OProj[2], s=60, marker='o', c='c',
               edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 10:
    """
    view rep and rpo together in the symmetry reduced
    prejection frame.
    Also view the unstable manifold of the req
    """
    N = 1024
    d = 30
    h = 0.0002

    di = 0.4226
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)

    x1, T1, nstp1, th1, phi1, err1 = cqcglReadRPOdi(
        '../../data/cgl/rpoT2X1.h5', di, 1)
    h1 = T1 / nstp1
    nstp1 = np.int(nstp1)
    cgl2 = pyCqcgl1d(N, d, h1, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa1 = cgl2.intg(x1, nstp1, 1)
    aa1Hat, th2, phi2 = cgl2.orbit2slice(aa1)
    aa1Hat -= a0Hat

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
    aa1HatProj = np.dot(aa1Hat, np.vstack((e1, e2, e3)).T)
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2, e3)).T)
    OProj = np.dot(-a0Hat, np.vstack((e1, e2, e3)).T)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aa1HatProj[:, 0], aa1HatProj[:, 1], aa1HatProj[:, 2], c='r', lw=2)
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            aaErgHatProj[:, 2], c='g', lw=1, alpha=0.4)
    i1 = 27500
    ax.plot(aaErgHatProj[i1:, 0], aaErgHatProj[i1:, 1],
            aaErgHatProj[i1:, 2], c='k', lw=1)
    ax.scatter([0], [0], [0], s=80, marker='o', c='b',  edgecolors='none')
    ax.scatter(OProj[0], OProj[1], OProj[2], s=60, marker='o', c='c',
               edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    # plotConfigSurfaceFourier(cgl, aa1, [0, d, 0, T1])