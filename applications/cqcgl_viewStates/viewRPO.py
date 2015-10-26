from py_cqcgl1d import pyCqcgl1d
from personalFunctions import *

case = 1

if case == 1:
    """
    view the rpo I found
    view its color map, error, Fourier modes and symmetry reduced Fourier modes
    """
    N = 1024
    d = 30
    M = 2
    x, T, nstp, th, phi, err = cqcglReadRPO('../../data/cgl/rpoT2x2.h5', '2')
    h = T / nstp / M
    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, -0.01, -0.04, 4)
    aa0 = cgl.intg(x[0], nstp*M, 1)
    aa1 = cgl.intg(x[0], nstp, 1)
    aa2 = cgl.intg(x[1], nstp, 1)
    aa = np.vstack((aa1, aa2))
    aaHat, thAll, phiAll = cgl.orbit2slice(aa)
    aaTilde = cgl.reduceReflection(aaHat)
    
    # print the errors and plot the color map
    print norm(cgl.Rotate(aa0[-1], th, phi) - aa0[0])
    print norm(aa1[-1]-x[1]), norm(cgl.Rotate(aa2[-1], th, phi) - x[0])
    plotConfigSpaceFromFourier(cgl, aa0, [0, d, 0, nstp*h*M])
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h*M])

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

    # see the profile fo exposion part
    plotOneConfigFromFourier(cgl, aa[1500], d)

