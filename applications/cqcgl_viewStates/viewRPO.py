from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 21

if case == 1:
    """
    view the rpo I found
    view its color map, error, Fourier modes and symmetry reduced Fourier modes
    """
    N = 1024
    d = 30
    di = 0.39
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
    """
    N = 1024
    d = 30
    di = 0.39
    M = 4
    
    x, T, nstp, th, phi, err = cqcglReadRPOdi('../../data/cgl/rpoT2X1.h5',
                                              di, 1)
    h = T / nstp
    cgl = pyCqcgl1d(N, d, h, True, M, 4.0, 0.8, 0.01, di, 4)
    # aa = cgl.intg(x[0], nstp, 1)
    # aaHat, thAll, phiAll = cgl.orbit2slice(aa)
    Q0 = rand(M, cgl.Ndim)
    # Q, R, D, C = cgl.powIt(x[0], th, phi, Q0, False, nstp,
    #                        nstp, 1000, 1e-10, True, 10)
    # print D
    e = cgl.powEigE(x[0], th, phi, Q0, nstp, nstp, 1000, 1e-10, True, 10)
    print e
