from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 120

if case == 1:
    """
    view the rpo I found
    view its color map, error, Fourier modes and symmetry reduced Fourier modes
    """
    N = 1024
    d = 30
    di = 0.4226
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    nstp = np.int(nstp)
    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa = cgl.intg(x, nstp, 1)
    aaHat, thAll, phiAll = cgl.orbit2slice(aa)
    
    # print the errors and plot the color map
    print norm(cgl.Rotate(aa[-1], th, phi) - aa[0])
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

    # Fourier trajectory in the full state space
    i1 = 0
    i2 = 1
    i3 = 2
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aa[:, i1], aa[:, i2], aa[:, i3], c='r', lw=1)
    ax.scatter(aa[0, i1], aa[0, i2], aa[0, i3], s=50, marker='o',
               facecolor='b', edgecolors='none')
    ax.scatter(aa[-1, i1], aa[-1, i2], aa[-1, i3], s=50, marker='o',
               facecolor='b', edgecolors='none')
    ax.set_xlabel('x', fontsize=25)
    ax.set_ylabel('y', fontsize=25)
    ax.set_zlabel('z', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

    #  Fourier trajectory in the continous symmetry reduced space
    i1 = 0
    i2 = 1
    i3 = 2
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aaHat[:, i1], aaHat[:, i2], aaHat[:, i3], c='r', lw=1)
    ax.scatter(aaHat[0, i1], aaHat[0, i2], aaHat[0, i3], s=50, marker='o',
               facecolor='b', edgecolors='none')
    ax.scatter(aaHat[-1, i1], aaHat[-1, i2], aaHat[-1, i3], s=50, marker='o',
               facecolor='b', edgecolors='none')
    ax.set_xlabel('x', fontsize=25)
    ax.set_ylabel('y', fontsize=25)
    ax.set_zlabel('z', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    
    # plot 4 periods
    M = 6
    aa2 = cgl.intg(x, nstp*M, 1)
    plotConfigSpaceFromFourier(cgl, aa2, [0, d, 0, nstp*h*M])

if case == 20:
    """
    view the rpo torus
    """
    N = 1024
    d = 30
    di = 0.39
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa0 = cgl.intg(x[0], nstp, 1)

    # Fourier trajectory in the full state space
    i1 = 4
    i2 = 7
    i3 = 2
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        ang = i / 10.0 * 2 * np.pi
        aa = cgl.Rotate(aa0, ang, 0)
        ax.plot(aa[:, i1], aa[:, i2], aa[:, i3], c='r', lw=1)
        ax.scatter(aa[0, i1], aa[0, i2], aa[0, i3], s=50, marker='o',
                   facecolor='b', edgecolors='none')
        ax.scatter(aa[-1, i1], aa[-1, i2], aa[-1, i3], s=50, marker='o',
                   facecolor='k', edgecolors='none')
    ax.set_xlabel('x', fontsize=25)
    ax.set_ylabel('y', fontsize=25)
    ax.set_zlabel('z', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 21:
    """
    check whether the velocity is conserved
    """
    N = 1024
    d = 30
    di = 0.39
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, True, 1, 4.0, 0.8, 0.01, di, 4)
    v0 = cgl.velocity(x[0])
    av = cgl.intgv(x[0], v0, nstp)
    v1 = av[1]
    
    print norm(v1)

if case == 30:
    """
    calculate the Floquet exponents of limit cycles
    by power iteration
    """
    N = 1024
    d = 30
    di = 0.4225
    M = 10
    
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, True, M, 4.0, 0.8, 0.01, di, 4)
    Q0 = rand(M, cgl.Ndim)
    # Q, R, D, C = cgl.powIt(x[0], th, phi, Q0, False, nstp,
    #                        nstp, 1000, 1e-10, True, 10)
    # print D
    e = cgl.powEigE(x[0], th, phi, Q0, nstp, nstp, 2000, 1e-12, True, 10)
    print e

if case == 40:
    """
    calculate the Floquet exponents of limit cycles directly
    """
    N = 1024
    d = 30
    di = 0.4225
    
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    aa, J = cgl.intgj(x[0], nstp, nstp, nstp)
    e, v = eig(cgl.Rotate(J, th, phi))
    
    idx = np.argsort(abs(e))
    idx = idx[::-1]
    e = e[idx]
    v = v[:, idx]

if case == 50:
    """
    plot all the rpos I have
    """
    N = 1024
    d = 30
    
    dis, rpos = cqcglReadRPOAll('../../data/cgl/rpoT2X1.h5', 1)
    for i in range(len(rpos)):
        x, T, nstp, th, phi, err = rpos[i]
        di = dis[i]
        h = T / nstp
        cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)
        
        M = 4
        aa = cgl.intg(x[0], nstp*M, 1)
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h*M],
                                   save=True,
                                   name='cqcglHopfCycle' + str(di) + '_4T.eps')

if case == 60:
    """
    test the correctness of the Floquet exponents/vectors
    obtained from the Krylov-Schur algorithm
    """
    N = 1024
    d = 30
    di = 0.36
    x, T, nstp, th, phi, err, es, vs = cqcglReadRPOEVdi(
        '../../data/cgl/rpoT2X1_v2.h5', di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)

    U = vs[0:3]
    # angle between velocity and marginal subspace
    v0 = cgl.velocity(x)
    ang1 = pAngle(v0, U.T)

    # angle between group tangent and marginal subspace
    tx_tau = cgl.transTangent(x)
    ang2 = pAngle(tx_tau, U.T)
    tx_rho = cgl.phaseTangent(x)
    ang3 = pAngle(tx_rho, U.T)

    print es
    print ang1, ang2, ang3

if case == 61:
    """
    Test whether the Krylov-Schur algorithm produces the correct
    number of marginal exponents.
    This is very import indicator for it to resolve degenercy.
    """
    for di in np.arange(0.36, 0.4211, 0.001).tolist() + np.arange(0.4211, 0.42201, 0.0001).tolist() + [0.4225, 0.4226]:
        x, T, nstp, th, phi, err, es, vs = cqcglReadRPOEVdi(
            '../../data/cgl/rpoT2X1EV30.h5', di, 1)
        print di, es[0][:4]

if case == 70:
    """
    move rpo with FE/FV
    """
    inFile = '../../data/cgl/rpoT2X1_v3.h5'
    outFile = '../../data/cgl/rpoT2X1EV30.h5'
    # for di in np.arange(0.36, 0.411, 0.001).tolist() + np.arange(0.414, 0.418, 0.001).tolist() + [0.4225, 0.4226]:
    for di in np.arange(0.361, 0.4191, 0.002).tolist():
        disp(di)
        cqcglMoveRPOEVdi(inFile, outFile, di, 1)

if case == 80:
    """
    plot the largest non-marginal Floquet exponent and periods of all
    RPO founded.
    """
    dis, xx = cqcglReadRPOAll('../../data/cgl/rpoT2X1EV30.h5', 1, True)
    fe = []
    T = []
    for i in range(len(dis)):
        e = xx[i][6][0]
        ep = removeMarginal(e, 3)
        fe.append(ep)
        T.append(xx[i][1])

    e1 = []
    e2 = []
    for i in range(len(dis)):
        e1.append(fe[i][0])
        e2.append(fe[i][1])

    # plot
    scale = 1.3
    fig = plt.figure(figsize=[6/scale, 4/scale])
    ax = fig.add_subplot(111)
    ax.plot(dis, e1,  c='b', lw=1, ls='-', marker='o', ms=5, mfc='r', mec='none')
    ax.plot(dis, np.zeros(len(dis)), c='g', ls='--', lw=2)
    ax.set_xlabel(r'$d_i$', fontsize=20)
    ax.set_ylabel(r'$\mu_1$', fontsize=20)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    
    scale = 1.3
    fig = plt.figure(figsize=[6/scale, 4/scale])
    ax = fig.add_subplot(111)
    ax.plot(dis, T,  c='b', lw=1, ls='-', marker='o', ms=5, mfc='r',
            mec='none')
    ax.set_xlabel(r'$d_i$', fontsize=20)
    ax.set_ylabel(r'$T_p$', fontsize=20)
    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 81:
    """
    For the Hopf bifurcation, we can actually estimate the parameters
    by experimental data. Here,
    we plot 1/T and the leading stable expoent of rpo and unstable expoent
    of req
    """
    dic = 0.36

    dis, xx = cqcglReadRPOAll('../../data/cgl/rpoT2X1EV30.h5', 1, True)
    T = []
    e1 = []
    for i in range(len(dis)):
        e = xx[i][6][0]
        ep = removeMarginal(e, 3)
        e1.append(ep[0])
        T.append(xx[i][1])

    ix = [i for i in range(len(dis)) if dis[i] <= dic+0.01 and dis[i] >= dic]

    dis2, xx2 = cqcglReadReqAll('../../data/cgl/reqDi.h5', 1, True)
    e1p = []
    for i in range(len(dis2)):
        e = xx2[i][4]
        ep = removeMarginal(e, 2)
        e1p.append(ep[0])
        
    ix2 = [i for i in range(len(dis2)) if dis2[i] <= dic+0.01 and dis2[i] >= dic]

    # plot
    scale = 1.3
    fig = plt.figure(figsize=[6/scale, 4/scale])
    ax = fig.add_subplot(111)
    ax.scatter(np.array(dis)[ix] - dic, -np.array(e1)[ix]/2,
               s=25, marker='o', facecolor='r', edgecolor='none')
    ax.scatter(np.array(dis2)[ix2] - dic, np.array(e1p)[ix2],
               s=25, marker='s', facecolor='b', edgecolor='none')
    # ax.plot(dis, np.zeros(len(dis)), c='g', ls='--', lw=2)
    ax.set_xlabel(r'$d_i - d_{ic}$', fontsize=20)
    # ax.set_ylabel(r'$\mu_1$', fontsize=20)
    ax.set_xlim([-0.001, 0.011])
    ax.set_ylim([0, 0.1])
    fig.tight_layout(pad=0)
    plt.show(block=False)
    
    scale = 1.3
    fig = plt.figure(figsize=[6/scale, 4/scale])
    ax = fig.add_subplot(111)
    ax.scatter(np.array(dis)[ix] - dic, 1 / np.array(T)[ix],
               s=25, marker='o', facecolor='r', edgecolor='none')
    ax.set_xlim([-0.001, 0.011])
    ax.set_xlabel(r'$d_i - d_{ic}$', fontsize=20)
    ax.set_ylabel(r'$1/T_p$', fontsize=20)
    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 90:
    """
    calculate the distance of rpo to the zero state
    because we guess the reason that rpo does not exit
    after a certain di is that it goes close to zero state
    which is contracting.
    """
    N = 1024
    d = 30
    dis = np.arange(0.36, 0.421, 0.002).tolist() + np.arange(0.421, 0.42201, 0.0001).tolist() + [0.4225, 0.4226]
    minNors = []
    for di in dis:
        print di
        x, T, nstp, th, phi, err = cqcglReadRPOdi(
            '../../data/cgl/rpoT2X1.h5', di, 1)
        h = T / nstp
        nstp = np.int(nstp)
        cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
        aa = cgl.intg(x, nstp, 1)
        aaHat, th2, phi2 = cgl.orbit2slice(aa)
        nors = []
        for i in range(aaHat.shape[0]):
            nor = norm(aaHat[i])
            nors.append(nor)

        minNors.append(min(nors))

    scale = 1.3
    fig = plt.figure(figsize=[6/scale, 4/scale])
    ax = fig.add_subplot(111)
    ax.plot(dis, minNors,  c='b', lw=1, ls='-', marker='o', ms=5, mfc='r',
            mec='none')
    ax.set_xlabel(r'$d_i$', fontsize=20)
    ax.set_ylabel(r'$\min|A|$', fontsize=20)
    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 100:
    """
    view the unstable manifold for some unstable Hopf cycles
    """
    N = 1024
    d = 30
    di = 0.4226

    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCqcgl1d(N, d, 0.0002, True, 0, 4.0, 0.8, 0.01, di, 4)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = Tcopy(realve(eigvectors))
    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)

    x, T, nstp, th, phi, err, e, v = cqcglReadRPOEVdi(
        '../../data/cgl/rpoT2X1EV31.h5', di, 1)
    h = T / nstp
    nstp = np.int(nstp)
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    aa = cgl.intg(x, nstp, 1)
    aaHat, th, phi = cgl.orbit2slice(aa)
    aaHat -= a0Hat

    h3 = 0.0005
    cgl3 = pyCqcgl1d(N, d, h3, False, 0, 4.0, 0.8, 0.01, di, 4)
    a0Erg = x + v[0] * 1e-3
    nstp = 70000
    aaErg = cgl3.intg(a0Erg, 10000, 10000)
    a0Erg = aaErg[-1]
    aaErg = cgl3.intg(a0Erg, nstp, 2)
    aaErgHat, th, th = cgl3.orbit2slice(aaErg)
    aaErgHat -= a0Hat
    
    # e1, e2 = orthAxes2(veHat[0], veHat[1])
    e1, e2, e3 = orthAxes(veHat[0], veHat[1], veHat[6])
    aaHatProj = np.dot(aaHat, np.vstack((e1, e2, e3)).T)
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2, e3)).T)
    OProj = np.dot(-a0Hat, np.vstack((e1, e2, e3)).T)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aaHatProj[:, 0], aaHatProj[:, 1], aaHatProj[:, 2], c='r', lw=2)
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            aaErgHatProj[:, 2], c='g', lw=1, alpha=0.4)
    ax.scatter([0], [0], [0], s=80, marker='o', c='b',  edgecolors='none')
    ax.scatter(OProj[0], OProj[1], OProj[2], s=60, marker='o', c='c',
               edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    # plotConfigSurfaceFourier(cgl, aa1, [0, d, 0, T1])

if case == 110:
    """
    view the not converged rpo
    """
    N = 1024
    d = 30
    di = 0.04
    
    x, T, nstp, th, phi, err = cqcglReadRPO('rpo2.h5', '2')
    M = x.shape[0]
    h = T / nstp / M
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    es, ev = eigReq(cgl, a0, wth0, wphi0)
    ev = Tcopy(realve(ev))
    a0H = cgl.orbit2slice(a0)[0]
    veH = cgl.ve2slice(ev, a0)

    nsp = 20
    aas = np.empty([0, cgl.Ndim])
    aaHs = []
    for i in range(M):
        aa = cgl.intg(x[i], nstp, nsp)
        aaH = cgl.orbit2slice(aa)[0]
        aas = np.vstack((aas, aa))
        aaHs.append(aaH)
    
    plotConfigSpaceFromFourier(cgl, aas, [0, d, 0, T])

    e1, e2, e3 = orthAxes(veH[0], veH[1], veH[10])
    bases = np.vstack((e1, e2, e3))
    aaHPs = []
    for i in range(M):
        aaHP = np.dot(aaHs[i]-a0H, bases.T)
        aaHPs.append(aaHP)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    for i in range(M):
        ax.plot(aaHPs[i][:, 0], aaHPs[i][:, 1],
                aaHPs[i][:, 2], c='g', lw=1)
        if i == 0:
            c = 'r'
        elif i == M-1:
            c = 'k'
        else:
            c = 'b'
        # ax.scatter(aaHPs[i][0, 0], aaHPs[i][0, 1], aaHPs[i][0, 2],
        #           s=50, marker='o', c=c,  edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

    for i in range(M-1):
        tmp = ax.scatter(aaHPs[i][-1, 0], aaHPs[i][-1, 1], aaHPs[i][-1, 2],
                         s=50, marker='o', c='k',  edgecolors='none')
        tmp2 = ax.scatter(aaHPs[i+1][0, 0], aaHPs[i+1][0, 1], aaHPs[i+1][0, 2],
                          s=50, marker='o', c='r',  edgecolors='none')
        t = raw_input("input: ")
        tmp.remove()
        tmp2.remove()


if case == 120:
    """
    view the not converged rpo
    the full multistep method
    """
    N = 512
    d = 30
    di = 0.04
    
    x, T, nstp, th, phi, err = cqcglReadRPO('rpo4.h5', '3')
    M = x.shape[0]
    Ts = x[:, -3]
    ths = x[:, -2]
    phis = x[:, -1]
    x = x[:, :-3]

    h = T / nstp / M
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    
    nsp = 2
    aas = np.empty([0, cgl.Ndim])
    aas2 = []
    aaHs = []
    for i in range(M):
        newh = Ts[i] / nstp
        cgl.changeh(newh)
        aa = cgl.intg(x[i], nstp, nsp)
        aaH = cgl.orbit2slice(aa)[0]
        aas = np.vstack((aas, aa))
        aas2.append(aa)
        aaHs.append(aaH)

    ers = np.zeros(M)
    ers2 = np.zeros(M)
    for i in range(M):
        j = (i+1) % M
        ers[i] = norm(aaHs[i][-1] - aaHs[j][0])
        ers2[i] = norm(cgl.Rotate(aas2[i][-1], ths[i], phis[i]) - aas2[j][0])

    plotConfigSpaceFromFourier(cgl, aas, [0, d, 0, T])

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111, projection='3d')
    for i in range(M):
        ax.plot(aaHs[i][:, 0], aaHs[i][:, 1],
                aaHs[i][:, 2], c='g', lw=1)
        if i == 0:
            c = 'r'
        elif i == M-1:
            c = 'k'
        else:
            c = 'b'
        # ax.scatter(aaHPs[i][0, 0], aaHPs[i][0, 1], aaHPs[i][0, 2],
        #           s=50, marker='o', c=c,  edgecolors='none')
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)

    for i in range(M-1):
        tmp = ax.scatter(aaHPs[i][-1, 0], aaHPs[i][-1, 1], aaHPs[i][-1, 2],
                         s=50, marker='o', c='k',  edgecolors='none')
        tmp2 = ax.scatter(aaHPs[i+1][0, 0], aaHPs[i+1][0, 1], aaHPs[i+1][0, 2],
                          s=50, marker='o', c='r',  edgecolors='none')
        t = raw_input("input: ")
        tmp.remove()
        tmp2.remove()
