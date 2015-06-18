import h5py
from pylab import *
import numpy as np
from numpy.linalg import norm, eig
from time import time
from py_cqcgl1d import pyCqcgl1d
from personalFunctions import *

case = 1


if case == 1:
    """
    plot unstable manifold
    """
    N = 512
    d = 50
    h = 0.001

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN512.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    veTilde = cgl.reflectVe(veHat, a0Hat)

    a0Reflected = cgl.reflect(a0)
    a0ReflectedHat = cgl.orbit2slice(a0Reflected)[0].squeeze()

    nstp = 85000
    a0Erg = a0 + eigvectors[0]*1e-4
    aaErg = cgl.intg(a0Erg, nstp, 5)
    aaErgHat, th, phi = cgl.orbit2slice(aaErg)
    aaErgTilde = cgl.reduceReflection(aaErgHat)
    aaErgTilde -= a0Tilde

    nstp2 = 90000
    a0Erg2 = a0 + eigvectors[2]*1e-4
    aaErg2 = cgl.intg(a0Erg2, nstp2, 5)
    aaErgHat2, th2, phi2 = cgl.orbit2slice(aaErg2)
    aaErgTilde2 = cgl.reduceReflection(aaErgHat2)
    aaErgTilde2 -= a0Tilde

    nstp3 = 85000
    a0Erg3 = cgl.Config2Fourier(centerRand(2*N, 0.25)).squeeze()
    aaErg3 = cgl.intg(a0Erg3, 10000, 10000)
    a0Erg3 = aaErg3[-1]
    aaErg3 = cgl.intg(a0Erg3, nstp3, 2)
    aaErgHat3, th3, phi3 = cgl.orbit2slice(aaErg3)
    aaErgTilde3 = cgl.reduceReflection(aaErgHat3)
    aaErgTilde3 -= a0Tilde

    e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])
    aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)
    aaErgTildeProj2 = np.dot(aaErgTilde2, np.vstack((e1, e2, e3)).T)
    aaErgTildeProj3 = np.dot(aaErgTilde3, np.vstack((e1, e2, e3)).T)

    # plot3dfig(aaErgHatProj[1000:, 0], aaErgHatProj[1000:, 1], aaErgHatProj[1000:, 2])
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ix1 = 37000/2
    ix2 = 43000/2
    ax.plot(aaErgTildeProj[:, 0], aaErgTildeProj[:, 1],
            aaErgTildeProj[:, 2], c='r', lw=1)
    ax.plot(aaErgTildeProj2[:, 0], aaErgTildeProj2[:, 1],
            aaErgTildeProj2[:, 2], c='m', lw=1)
    ax.plot(aaErgTildeProj3[ix1:ix2, 0], aaErgTildeProj3[ix1:ix2, 1],
            aaErgTildeProj3[ix1:ix2, 2], c='g', lw=1)
    ax.scatter([0], [0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    plotConfigSpace(cgl.Fourier2Config(aaErg3), [0, d, 0, nstp*h])

    # plot3dfig(aaErg[:, 0], aaErg[:, 2], aaErg[:, 4])
    # plot3dfig(aaErgHat[:, 0], aaErgHat[:, 2], aaErgHat[:, 4])

    # plot1dfig(aaErgHatProj[:, 0], marker='')
    # plot1dfig(aaErgHat[:, 0], marker='')

