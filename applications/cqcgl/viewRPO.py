from py_CQCGL1d import *
from personalFunctions import *
from scipy.integrate import odeint

case = 411

if case == 1:
    """
    view the rpo I found
    view its color map, error, Fourier modes and symmetry reduced Fourier modes
    """
    N = 1024
    d = 30
    di = 0.36
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    nstp = np.int(nstp)
    cgl = pyCQCGL1d(N, d, 4.0, 0.8, 0.01, di, -1)
    aa = cgl.intg(x, h, nstp, 1)
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
    aa2 = cgl.intg(x, h, nstp*M, 1)
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

if case == 40:
    """
    use L = 50 to obtain the guess initial conditon
    for rpo
    """
    N = 1024
    d = 50
    Bi = 0.8
    Gi = -3.6

    rpo = CQCGLrpo()
    x, T, nstp, th, phi, err = rpo.readRpodi('../../data/cgl/rpoT2X1.h5',
                                             0.36, 1)
    x = x * 0.1**0.5
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    aa = cgl.intg(x, 1e-3, 40000, 50)
    aa = cgl.intg(aa[-1], 1e-3, 10000, 10)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, 10])

if case == 41:
    """
    use L = 50 to view the rpo
    """
    N, d = 1024, 50
    Bi, Gi = 2, -5.6

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)
    x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                                     Bi, Gi, 1, flag=2)
    a0 = x[:cgl.Ndim]
    print e[:10]

    a0 += 0.1*norm(a0)*v[0]
    numT = 10
    for i in range(1):
        aa = cgl.intg(a0, T/nstp, numT*nstp, 10)
        a0 = aa[-1]
        cp.config(aa, [0, d, 0, T*numT])
        
    
if case == 411:
    """
    use L = 50 to view the rpo and the difference plot
    """
    N = 1024
    d = 50
    Bi, Gi = 4.8, -4.5

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)
    x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                                     Bi, Gi, 1, flag=2)
    a0 = x[:cgl.Ndim]
    print e[:10]
    
    aa0 = cgl.intg(a0, T/nstp, nstp, 10)[:-1]

    a0 += 0.1*norm(a0)*v[0]
    numT = 7
    skipRate = 10

    aaBase = aa0
    for i in range(1, numT):
        aaBase = np.vstack((aaBase, cgl.Rotate(aa0, -i*th, -i*phi)))
    
    for i in range(1):
        aa = cgl.intg(a0, T/nstp, numT*nstp, skipRate)
        a0 = aa[-1]
        cp.config(aa, [0, d, 0, T*numT])
        dif = aa[:-1] - aaBase
        cellSize = nstp / skipRate
        cp.config(dif[0*cellSize:6*cellSize], [0, d, 0*T, 6*T])

if case == 42:
    """
    L = 50 check the po is not req
    """
    N = 1024
    d = 50

    fileName = '../../data/cgl/rpoBiGi2.h5'
    angs = np.zeros([39, 55])
    for i in range(39):
        Bi = 1.9 + i * 0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
            rpo = CQCGLrpo(cgl)
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1)):
                x, T, nstp, th, phi, err = rpo.readRpoBiGi(fileName, Bi, Gi, 1)
                a0 = x[:cgl.Ndim]
                v0 = cgl.velocity(a0)
                t1 = cgl.transTangent(a0)
                t2 = cgl.phaseTangent(a0)
                ang = pAngle(v0, np.vstack((t1, t2)).T)
                angs[i, j] = ang
                if ang < 1e-3:
                    print Bi, Gi, ang

if case == 43:
    """
    L = 50
    plot the stability table of the rpos
    """
    N = 1024
    d = 50 

    fileName = '../../data/cgl/rpoBiGiEV.h5'
    
    cs = {0: 'g'}#, 10: 'm', 2: 'c', 4: 'r', 6: 'b', 8: 'y'}#, 56: 'r', 14: 'grey'}
    ms = {0: 's'}#, 10: '^', 2: '+', 4: 'o', 6: 'D', 8: 'v'}#, 56: '4', 14: 'v'}

    es = {}
    e1 = {}
    unconverge = {}
    Ts = np.zeros((39, 55))
    ns = set()

    fig, ax = pl2d(size=[8, 6], labs=[r'$G_i$', r'$B_i$'], axisLabelSize=25, tickSize=20,
                   xlim=[-5.7, -3.95], ylim=[1.8, 6])
    for i in range(39):
        Bi = 1.9 + i*0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            rpo = CQCGLrpo()
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1) + '/er'):
                x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi(fileName, Bi, Gi, 1, flag=2)
                es[(Bi, Gi)] = e
                m, ep, accu = numStab(e, nmarg=3, tol=1e-4, flag=1)
                if accu == False:
                    unconverge[(Bi, Gi)] = 1
                    print Bi, Gi, e[m:m+3]
                e1[(Bi, Gi)] = ep[0]
                Ts[i, j] = T
                ns.add(m)
                # print index, Bi, Gi, m
                ax.scatter(Gi, Bi, s=60, edgecolors='none',
                           marker=ms.get(m, 'o'), c=cs.get(m, 'r'))

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.grid(which='major', linewidth=2)
    #ax.grid(which='minor', linewidth=1)
    ax2d(fig, ax)
    plotIm(Ts, [-5.6, 0, -4, 6], size=[8, 6], labs=[r'$G_i$', r'$B_i$'])
    fig, ax = pl2d(size=[8, 6], labs=[r'$G_i$', r'$B_i$'], axisLabelSize=20, tickSize=15,
                   xlim=[-5.7, -3.95], ylim=[1.8, 6])
    keys = unconverge.keys()
    for i in keys:
        ax.scatter(i[1], i[0], s=60, edgecolors='none', marker='o', c='k')
    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.grid(which='major', linewidth=2)
    #ax.grid(which='minor', linewidth=1)
    ax2d(fig, ax)
    

if case == 44:
    """
    L = 50
    Plot the rpo existence in the Bi-Gi plane
    """
    N = 1024
    d = 50 

    fileName = '../../data/cgl/rpoBiGi2.h5'
    fig, ax = pl2d(size=[8, 6], labs=[r'$G_i$', r'$B_i$'], axisLabelSize=25,
                   xlim=[-5.8, -3.5], ylim=[1.8, 6])
    for i in range(39):
        Bi = 1.9 + i*0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            rpo = CQCGLrpo()
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1)):
                ax.scatter(Gi, Bi, s=15, edgecolors='none', marker='o', c='r')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(which='major', linewidth=2)
    ax.grid(which='minor', linewidth=1)
    ax2d(fig, ax)
    
if case == 45:
    """
    use L = 50 to view the rpo
    """
    N, d = 1024, 50
    Bi, Gi = 1.9, -5.6
    
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo()
    cp = CQCGLplot(cgl)
    x, T, nstp, th, phi, err = rpo.readRpoBiGi('../../data/cgl/rpoBiGi2.h5',
                                               Bi, Gi, 1, flag=0)
    a0 = x[:cgl.Ndim]

    for i in range(2):
        aa = cgl.intg(a0, T/nstp, 15*nstp, 10)
        a0 = aa[-1]
        cp.config(aa, [0, d, 0, 2*T])

if case == 46:
    """
    use L = 50 to view the rpo int the symmetry reduced state space
    """
    N = 1024
    d = 50
    Bi = 4.8
    Gi = -4.5
    sysFlag = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)

    x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                               Bi, Gi, 1, flag=2)
    a0 = x[:cgl.Ndim]
    po = cgl.intg(a0, T/nstp, nstp, 10)
    poH = cgl.orbit2slice(po, sysFlag)[0]

    aE = a0 + 0.001 * norm(a0) * v[0];

    aa = cgl.intg(aE, T/nstp, 12*nstp, 10)
    cp.config(aa, [0, d, 0, 12*T])
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    cp. config(aaH, [0, d, 0, 12*T])

    # req = CQCGLreq(cgl)
    # reqe, reqaH, reqvH = req.getAxes('../../data/cgl/reqBiGiEV.h5', Bi, Gi, 1, sysFlag)
    # e1, e2, e3 = orthAxes(reqvH[0].real, reqvH[1].real, reqvH[2].real)
    # bases = np.vstack((e1, e2, e3))
    # poHP, aaHP = poH.dot(bases.T), aaH.dot(bases.T)
    
    vrH = cgl.ve2slice(v.real.copy(), a0, sysFlag)
    viH = cgl.ve2slice(v.imag.copy(), a0, sysFlag)
    e1, e2, e3 = orthAxes(vrH[0], viH[0], vrH[1])
    bases = np.vstack((e1, e2, e3))
    poHP, aaHP = poH.dot(bases.T), aaH.dot(bases.T)
    
    fig, ax = pl3d(size=[8, 6])
    ax.plot(poH[:, -1], poH[:, 4], poH[:, 5], c='r', lw=2)
    ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    #ax.plot(poHP[:, 0], poHP[:, 1], poHP[:, 2], c='r', lw=2)
    #ax.plot(aaHP[:, 0], aaHP[:, 1], aaHP[:, 2], c='b', lw=1)
    ax3d(fig, ax)
    
if case == 47:
    """
    L = 50 find all moving rpo 
    """
    N = 1024
    d = 50

    fileName = '../../data/cgl/rpoBiGi2.h5'
    mv = {}
    nmv = {}
    for i in range(39):
        Bi = 1.9 + i * 0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            rpo = CQCGLrpo()
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1)):
                x, T, nstp, th, phi, err = rpo.readRpoBiGi(fileName, Bi, Gi, 1)
                if abs(th) > 1e-4:
                    mv[(Bi, Gi)] = th
                else :
                    nmv[(Bi, Gi)] = th
                

    fig, ax = pl2d(size=[8, 6], labs=[r'$G_i$', r'$B_i$'], axisLabelSize=20, tickSize=15,
                   xlim=[-5.7, -3.95], ylim=[1.8, 6])
    mvkeys = mv.keys()
    for i in mvkeys:
        ax.scatter(i[1], i[0], s=60, edgecolors='none', marker='o', c='r')
    nmvkeys = nmv.keys()
    for i in nmvkeys:
        ax.scatter(i[1], i[0], s=60, edgecolors='none', marker='s', c='g')
    ax2d(fig, ax)

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


if case == 130:
    """
    use odeint to integrate in the slice directly
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

    def vel(x, t):
        return cgl.velocity(x)
    
    pH = np.empty([0, cgl.Ndim])
    pHs = []
    for k in range(M):
        x0H = aaHs[k][0]
        xH, infodict = odeint(vel, x0H, np.linspace(0, Ts[k], nstp),
                              full_output=True)
        print np.min(infodict['hu']), np.max(infodict['hu'])
        pH = np.vstack((pH, xH))
        pHs.append(xH)

if case == 140:
    cgl.changeh(cgl.h/10)
    J = cgl.intgj(x[0], 10, 10, 10)[1].T
    J2 = np.eye(cgl.Ndim) + cgl.stab(x[0]).T * cgl.h * 10
    print np.max(J), np.max(J2), np.min(J), np.min(J2)
    print J[:3, :3]
    print J2[:3, :3]
    
    def velJ(xj, t):
        x = xj[:cgl.Ndim]
        J = xj[cgl.Ndim:].reshape(cgl.Ndim, cgl.Ndim)
        v = cgl.velocity(x)
        A = cgl.stab(x).T
        AJ = A.dot(J)
        return np.concatenate((v, AJ.ravel()))
        
    def calJ(x, n):
        j = np.eye(cgl.Ndim)
        xj = np.concatenate((x, j.ravel()))
        y = odeint(velJ, xj, np.linspace(0, cgl.h*n, n+1))
        
        return y[:, :cgl.Ndim].copy(), y[-1, cgl.Ndim:].copy().reshape((cgl.Ndim, cgl.Ndim))

    def calJ2(x, n):
        y = odeint(velJ, x, np.linspace(0, cgl.h*n, n+1))
        return y[-1]

    def calJ3(x, n):
        j = np.eye(cgl.Ndim)
        xj = np.concatenate((x, j.ravel()))
        y = odeint(velJ, xj, np.linspace(0, cgl.h*n, n+1))
  
        return y[-1]
        
    aa, J = cgl.intgj(x[0], 10, 10, 10)
    aa2, J2 = calJ(x[0], 10)

    cgl.changeh(cgl.h / 10)
    aa3, J3 = cgl.intgj(x[0], 100, 100, 100)
    aa4, J4 = calJ(x[0], 100)

    print norm(aa3[-1] - aa[-1]), norm(aa4[-1] - aa2[-1])
    print np.max(np.max(np.abs(J3-J))), np.max(np.max(np.abs(J4-J2)))
    
    y = calJ3(x[0], 10)
    for i in range(2):
        print i
        y = calJ2(y, 100)
