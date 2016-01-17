from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 80

if case == 1:
    """
    view the rpo I found
    view its color map, error, Fourier modes and symmetry reduced Fourier modes
    """
    N = 1024
    d = 30
    di = 0.4225
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)
    aa = cgl.intg(x[0], nstp, 1)
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
    aa2 = cgl.intg(x[0], nstp*M, 1)
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
    x, T, nstp, th, phi, err, es, vs = cqcglReadRPOEVdi('../../data/cgl/rpoT2X1_v2.h5', di, 1)
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

if case == 70:
    """
    move rpo with FE/FV
    """
    inFile = '../../data/cgl/rpoT2X1EV31.h5'
    outFile = '../../data/cgl/rpoT2X1_v2.h5'
    for di in np.arange(0.36, 0.421, 0.002).tolist() + np.arange(0.421, 0.42201, 0.0001).tolist() + [0.4225, 0.4226]:
    # for di in [0.368]:
        disp(di)
        cqcglMoveRPOEVonlydi(inFile, outFile, di, 1)

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
    ax.plot(dis, T,  c='b', lw=1, ls='-', marker='o', ms=5, mfc='r', mec='none')
    ax.set_xlabel(r'$d_i$', fontsize=20)
    ax.set_ylabel(r'$T_p$', fontsize=20)
    fig.tight_layout(pad=0)
    plt.show(block=False)
