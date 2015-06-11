import h5py
from pylab import *
import numpy as np
from numpy.linalg import norm, eig
from numpy.random import rand
from numpy.fft import fft, ifft
from time import time
from py_cqcgl1d import pyCqcgl1d
from personalFunctions import *


case = 1


if case == 1:
    """
    view an ergodic instance
    full state space => reduce continuous symmetry =>
    reduce reflection symmetry
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    Ndim = cgl.Ndim

    A0 = centerRand(2*N, 0.2)
    a0 = cgl.Config2Fourier(A0)
    nstp = 15000
    aa = cgl.intg(a0, nstp, 1)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

    aaHat, th, phi = cgl.orbit2sliceWrap(aa)
    aaHat2, th2, phi2 = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(aaHat), [0, d, 0, nstp*h])
    plotConfigSpace(cgl.Fourier2Config(aaHat2), [0, d, 0, nstp*h])
    aaTilde = cgl.reduceReflection(aaHat)
    plotConfigSpace(cgl.Fourier2Config(aaTilde), [0, d, 0, nstp*h])

    # rely on numpy's unwrap function
    th3 = unwrap(th*2.0)/2.0
    phi3 = unwrap(phi*2.0)/2.0
    aaHat3 = cgl.rotateOrbit(aa, -th3, -phi3)
    plotConfigSpace(cgl.Fourier2Config(aaHat3), [0, d, 0, nstp*h])

    # rotate by g(pi, pi)
    aaHat4 = cgl.Rotate(aaHat2, pi, pi)
    plotConfigSpace(cgl.Fourier2Config(aaHat4), [0, d, 0, nstp*h])
    aaTilde4 = cgl.reduceReflection(aaHat4)
    plotConfigSpace(cgl.Fourier2Config(aaTilde4), [0, d, 0, nstp*h])

    # reflection
    aa2 = cgl.reflect(aa)
    plotConfigSpace(cgl.Fourier2Config(aa2), [0, d, 0, nstp*h])
    aaHat5, th5, phi5 = cgl.orbit2slice(aa2)
    plotConfigSpaceFromFourier(cgl, aaHat5, [0, d, 0, nstp*h])
    aaTilde5 = cgl.reduceReflection(aaHat5)
    plotConfigSpace(cgl.Fourier2Config(aaTilde5), [0, d, 0, nstp*h])

# view relative equlibria
if case == 2:
    N = 256
    d = 50
    h = 0.01

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/req.h5', 'r')
    req = '/1/'
    a0 = f[req+'a'].value
    th0 = f[req+'wth'].value
    phi0 = f[req+'wphi'].value
    err = f[req+'err'].value
    f.close()

    vReq = cgl.velocityReq(a0, th0, phi0)
    print norm(vReq)

    # check the reflected state
    a0Reflected = cgl.reflect(a0)
    vReqReflected = cgl.velocityReq(a0Reflected, -th0, phi0)
    print norm(vReqReflected)
    plotOneConfigFromFourier(cgl, a0)
    plotOneConfigFromFourier(cgl, a0Reflected)

    # obtain the stability exponents/vectors
    stabMat = cgl.stabReq(a0, th0, phi0).T
    # stabMat = cgl.stabReq(a0Reflected, -th0, phi0).T  # transpose is needed
    eigvalues, eigvectors = eig(stabMat)
    eigvalues, eigvectors = sortByReal(eigvalues, eigvectors)
    print eigvalues[:10]

    # make sure you make a copy because Fourier2Config takes contigous memory
    tmpr = eigvectors[:, 0].real.copy()
    tmprc = cgl.Fourier2Config(tmpr).squeeze()
    tmpi = eigvectors[:, 0].imag.copy()
    tmpic = cgl.Fourier2Config(tmpi).squeeze()
    plotOneConfig(tmprc, size=[6, 4])
    plotOneConfig(tmpic, size=[6, 4])


    nstp = 2000
    aa = cgl.intg(a0Reflected, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h], [0, 2])
    raa, th, phi = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(raa), [0, d, 0, nstp*h], [0, 2])
    plotOneFourier(aa[-1])

# test the ve2slice function
if case == 3:
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/req.h5', 'r')
    req = '/req1/'
    a0 = f[req+'a0'].value
    th0 = f[req+'w1'].value[0]
    phi0 = f[req+'w2'].value[0]

    stabMat = cgl.stabReq(a0, th0, phi0).T
    eigvalues, eigvectors = eig(stabMat)
    eigvalues, eigvectors = sortByReal(eigvalues, eigvectors)

    # plot the last vector to see whether Tcopy works
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    plotOneConfigFromFourier(cgl, eigvectors[-1].real)

    veHat = cgl.ve2slice(eigvectors, a0)
    print veHat[:4, :8]
    plotOneConfigFromFourier(cgl, eigvectors[0].real)
    # print out the norm of two marginal vectors. They should valish
    print norm(veHat[4]), norm(veHat[5])

    # test the angle between each eigenvector
    v1 = veHat[1]
    v2 = veHat[3]
    print np.dot(v1, v2) / norm(v1) / norm(v2)
    plotOneFourier(v1)
    plotOneFourier(v2)

# unstable manifold
if case == 4:
    N = 256
    d = 50
    h = 0.001
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/req.h5', 'r')
    req = '/1/'
    a0 = f[req+'a'].value
    wth0 = f[req+'wth'].value
    wphi0 = f[req+'wphi'].value
    err = f[req+'err'].value
    f.close()

    stabMat = cgl.stabReq(a0, wth0, wphi0).T
    eigvalues, eigvectors = eig(stabMat)
    eigvalues, eigvectors = sortByReal(eigvalues, eigvectors)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    veHat = cgl.ve2slice(eigvectors, a0)

    a0Hat = cgl.orbit2slice(a0)[0].squeeze()
    a0Reflected = cgl.reflect(a0)
    a0ReflectedHat = cgl.orbit2slice(a0Reflected)[0].squeeze()

    nstp = 85000
    a0Erg = a0 + eigvectors[0]*1e-4
    aaErg = cgl.intg(a0Erg, nstp, 5)
    aaErgHat, th, phi = cgl.orbit2slice(aaErg)
    aaErgHat -= a0Hat

    nstp2 = 85000
    a0Erg2 = a0 + eigvectors[2]*1e-4
    aaErg2 = cgl.intg(a0Erg2, nstp2, 5)
    aaErgHat2, th2, phi2 = cgl.orbit2slice(aaErg2)
    aaErgHat2 -= a0Hat

    nstp3 = 85000
    a0Erg3 = cgl.Config2Fourier(centerRand(2*N, 0.25)).squeeze()
    aaErg3 = cgl.intg(a0Erg3, nstp3, 5)
    a0Erg3 = aaErg3[-1]
    aaErg3 = cgl.intg(a0Erg3, nstp3, 5)
    aaErgHat3, th3, phi3 = cgl.orbit2slice(aaErg3)
    aaErgHat3 -= a0Hat

    e1, e2, e3 = orthAxes(veHat[0], veHat[2], veHat[3])
    a0ReflectedHatProj = np.dot(a0ReflectedHat, np.vstack((e1, e2, e3)).T)
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2, e3)).T)
    aaErgHatProj2 = np.dot(aaErgHat2, np.vstack((e1, e2, e3)).T)
    aaErgHatProj3 = np.dot(aaErgHat3, np.vstack((e1, e2, e3)).T)

    # plot3dfig(aaErgHatProj[1000:, 0], aaErgHatProj[1000:, 1], aaErgHatProj[1000:, 2])
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ix1 = 00000/5
    ix2 = 14000/5
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            aaErgHatProj[:, 2], c='r', lw=1)
    # ax.plot(aaErgHatProj2[:, 0], aaErgHatProj2[:, 1],
    #         aaErgHatProj2[:, 2], c='m', lw=1)
    ax.plot(aaErgHatProj3[ix1:ix2, 0], aaErgHatProj3[ix1:ix2, 1],
            aaErgHatProj3[ix1:ix2, 2], c='g', lw=1)
    ax.scatter([0], [0], [0], s=80)
    # ax.scatter(a0ReflectedHatProj[0], a0ReflectedHatProj[1], a0ReflectedHatProj[2], s=80, c='k')
    fig.tight_layout(pad=0)
    plt.show(block=False)
    plotConfigSpace(cgl.Fourier2Config(aaErg3), [0, d, 0, nstp*h])

    # plot3dfig(aaErg[:, 0], aaErg[:, 2], aaErg[:, 4])
    # plot3dfig(aaErgHat[:, 0], aaErgHat[:, 2], aaErgHat[:, 4])

    # plot1dfig(aaErgHatProj[:, 0], marker='')
    # plot1dfig(aaErgHat[:, 0], marker='')
