from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 20


if case == 10:
    """
    plot unstable manifold
    of the exploding soliton solution
    """
    N = 1024
    d = 30
    h = 0.0002
    di = 0.05

    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    veHat = cgl.ve2slice(eigvectors, a0)

    nstp = 85000
    a0Erg = a0 + eigvectors[0]*1e-4
    aaErg = cgl.intg(a0Erg, nstp, 1)
    aaErgHat, th, phi = cgl.orbit2slice(aaErg)

    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[4])
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2, e3)).T)

    # plot3dfig(aaErgHatProj[1000:, 0], aaErgHatProj[1000:, 1], aaErgHatProj[1000:, 2])
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ix1 = 43000
    ix2 = 47000
    ixs = ix1 + 1400
    ax.plot(aaErgHatProj[ix1:ix2, 0], aaErgHatProj[ix1:ix2, 1],
            aaErgHatProj[ix1:ix2, 2], c='r', lw=1)
    # ax.scatter([0], [0], [0], s=120)
    ax.scatter(aaErgHatProj[ixs, 0], aaErgHatProj[ixs, 1],
               aaErgHatProj[ixs, 2], s=120, c='k')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    plotConfigSpaceFromFourier(cgl, aaErg[ix1:ix2], [0, d, 0, nstp*h])
    plotOneConfigFromFourier(cgl, aaErg[ixs], d=d)

    cqcglSaveReqdi('../../data/cgl/reqTmp.h5', di, 1, aaErg[ixs], 0, 0, 100)
    # plot3dfig(aaErgTildeProj[:, 0], aaErgTildeProj[:, 1], aaErgTildeProj[:, 2],
    #           labs=[r'$e_1$', r'$e_2$', r'$e_3$'])
    # plot3dfig(aaErg[:, 0], aaErg[:, 2], aaErg[:, 4])
    # plot3dfig(aaErgHat[:, 0], aaErgHat[:, 2], aaErgHat[:, 4])

    # plot1dfig(aaErgHatProj[:, 0], marker='')
    # plot1dfig(aaErgHat[:, 0], marker='')

if case == 20:
    """
    plot unstable manifold
    of the exploding soliton solution from a
    line of states
    """
    N = 1024
    d = 30
    h = 0.0002
    di = 0.05

    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    veHat = cgl.ve2slice(eigvectors, a0)

    nstp = 3000
    Er = eigvalues[0].real
    Ei = eigvalues[0].imag
    Vr, Vi = orthAxes2(eigvectors[0], eigvectors[1])
    n = 40
    aaE = []
    aaEHat = []
    for i in range(n):
        print i
        # ang = 2 * i * np.pi / n
        e = np.exp(2*np.pi * Er / Ei / n * i*10)
        a0Erg = a0 + Vr * e * 1e-4
        aaErg = cgl.intg(a0Erg, nstp, 1)
        aaErgHat, th, phi = cgl.orbit2slice(aaErg)
        # aaE.append(aaErg)
        aaEHat.append(aaErgHat - a0Hat)

    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[10])
    aaEHatP = []
    for i in range(n):
        aaErgHatProj = np.dot(aaEHat[i], np.vstack((e1, e2, e3)).T)
        aaEHatP.append(aaErgHatProj)

    ix1 = 00000
    ix2 = 03000
    ixs = ix1 + 1400

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        ax.plot(aaEHatP[i][ix1:ix2, 0], aaEHatP[i][ix1:ix2, 1],
                aaEHatP[i][ix1:ix2, 2], c='r', lw=1)
    # ax.scatter([0], [0], [0], s=120)
    # ax.scatter(aaErgHatProj[ixs, 0], aaErgHatProj[ixs, 1],
    #            aaErgHatProj[ixs, 2], s=120, c='k')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    # plotConfigSpaceFromFourier(cgl, aaErg[ix1:ix2], [0, d, 0, nstp*h])
    # plotOneConfigFromFourier(cgl, aaErg[ixs], d=d)

    

if case == 30:
    """
    try to obtain the Poincare intersection points
    """
    N = 1024
    d = 50
    h = 0.0001
    
    cgl = pyCqcgl1d(N, d, h, True, 0,
                    -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
                    4)
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN1024.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    veTilde = cgl.reflectVe(veHat, a0Hat)
    e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])

    nstp = 60000

    M = 10
    a0Erg = np.empty((M, cgl.Ndim))
    for i in range(M):
        a0Erg[i] = a0 + (i+1) * eigvectors[0]*1e-4
    PointsProj = np.zeros((0, 2))
    PointsFull = np.zeros((0, cgl.Ndim))
    for i in range(30):
        for j in range(M):
            aaErg = cgl.intg(a0Erg[j], nstp, 1)
            aaErgHat, th, phi = cgl.orbit2slice(aaErg)
            aaErgTilde = cgl.reduceReflection(aaErgHat)
            aaErgTilde -= a0Tilde
            aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)

            # plotConfigSpace(cgl.Fourier2Config(aaErg),
            #                 [0, d, nstp*h*i, nstp*h*(i+1)])
            points, index, ratios = PoincareLinearInterp(aaErgTildeProj, getIndex=True)
            PointsProj = np.vstack((PointsProj, points))
            for i in range(len(index)):
                dif = aaErgTilde[index[i]+1] - aaErgTilde[index[i]]
                p = dif * ratios[i] + aaErgTilde[index[i]]
                PointsFull = np.vstack((PointsFull, p))
            a0Erg[j] = aaErg[-1]

    upTo = PointsProj.shape[0]
    scatter2dfig(PointsProj[:upTo, 0], PointsProj[:upTo, 1], s=10,
                 labs=[r'$e_2$', r'$e_3$'])
    dis = getCurveCoor(PointsFull)
    # np.savez_compressed('PoincarePoints', totalPoints=totalPoints)

if case == 40:
    """
    plot the Poincare intersection points
    """
    totalPoints = np.load('PoincarePoints.npz')['totalPoints']
    multiScatter2dfig([totalPoints[:280, 0], totalPoints[280:350, 0]],
                      [totalPoints[:280, 1], totalPoints[280:350, 1]],
                      s=[15, 15], marker=['o', 'o'], fc=['r', 'b'],
                      labs=[r'$e_2$', r'$e_3$'])
    
    scatter2dfig(totalPoints[:, 0], totalPoints[:, 1], s=10,
                 labs=[r'$e_2$', r'$e_3$'])

if case == 50:
    """
    use the new constructor of cqcgl
    try to obtain the Poincare intersection points
    """
    N = 1024
    d = 40
    h = 0.0005

    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, -0.04, 4)
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN1024.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    veTilde = cgl.reflectVe(veHat, a0Hat)
    e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])

    nstp = 60000

    M = 10
    a0Erg = np.empty((M, cgl.Ndim))
    for i in range(M):
        a0Erg[i] = a0 + (i+1) * eigvectors[0]*1e-4
    PointsProj = np.zeros((0, 2))
    PointsFull = np.zeros((0, cgl.Ndim))
    for i in range(30):
        for j in range(M):
            aaErg = cgl.intg(a0Erg[j], nstp, 1)
            aaErgHat, th, phi = cgl.orbit2slice(aaErg)
            aaErgTilde = cgl.reduceReflection(aaErgHat)
            aaErgTilde -= a0Tilde
            aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)

            # plotConfigSpace(cgl.Fourier2Config(aaErg),
            #                 [0, d, nstp*h*i, nstp*h*(i+1)])
            points, index, ratios = PoincareLinearInterp(aaErgTildeProj, getIndex=True)
            PointsProj = np.vstack((PointsProj, points))
            for i in range(len(index)):
                dif = aaErgTilde[index[i]+1] - aaErgTilde[index[i]]
                p = dif * ratios[i] + aaErgTilde[index[i]]
                PointsFull = np.vstack((PointsFull, p))
            a0Erg[j] = aaErg[-1]

    upTo = PointsProj.shape[0]
    scatter2dfig(PointsProj[:upTo, 0], PointsProj[:upTo, 1], s=10,
                 labs=[r'$e_2$', r'$e_3$'])
    dis = getCurveCoor(PointsFull)
    # np.savez_compressed('PoincarePoints', totalPoints=totalPoints)
