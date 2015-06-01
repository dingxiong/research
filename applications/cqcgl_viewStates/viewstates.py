import h5py
from pylab import *
import numpy as np
from numpy.linalg import norm, eig
from numpy.random import rand
from numpy.fft import fft, ifft
from time import time
from py_cqcgl1d import pyCqcgl1d
from cqcgl1d import *

N = 256
d = 50
h = 0.01

case = 4

# view an ergodic instance
if case == 1:
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    A0 = rand(2*N)
    A0[:N/2] = 0
    A0[-1:-N/2:-1] = 0
    a0 = cgl.Config2Fourier(A0)
    nstp = 10000
    aa = cgl.intg(a0, nstp, 1)
    AA = cgl.Fourier2Config(aa)
    plotConfigSpace(AA, [0, d, 0, nstp*h])
    plotOneConfig(AA[5200])
    raa, th, phi = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(raa), [0, d, 0, nstp*h])


# view relative equlibria
if case == 2:
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/req.h5', 'r')
    req = '/req1/'
    a0 = f[req+'a0'].value
    th0 = f[req+'w1'].value[0]
    phi0 = f[req+'w2'].value[0]

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
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/req.h5', 'r')
    req = '/req1/'
    a0 = f[req+'a0'].value
    th0 = f[req+'w1'].value[0]
    phi0 = f[req+'w2'].value[0]

    stabMat = cgl.stabReq(a0, th0, phi0).T
    eigvalues, eigvectors = eig(stabMat)
    eigvalues, eigvectors = sortByReal(eigvalues, eigvectors)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    veHat = cgl.ve2slice(eigvectors, a0)

    e1, e2, e3 = orthAxes(veHat[0], veHat[2], veHat[6])
    nstp = 3000
    # a0Erg = a0 + e1*1e-5
    A0 = rand(2*N)
    A0[:N/2] = 0
    A0[-1:-N/2:-1] = 0
    a0Erg = cgl.Config2Fourier(A0)
    aaErg = cgl.intg(a0Erg, nstp, 1)
    raaErg, th, phi = cgl.orbit2slice(aaErg - a0)
    raaErgProj = np.dot(raaErg, np.vstack((e1, e2, e3)).T)

    # plot3dfig(raaErgProj[1000:, 0], raaErgProj[1000:, 1], raaErgProj[1000:, 2])
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(raaErgProj[1000:, 0], raaErgProj[1000:, 1], raaErgProj[1000:, 2], c='r', lw=1)
    ax.scatter([0], [0], [0], s=80)
    fig.tight_layout(pad=0)
    plt.show(block=False)
    plotConfigSpace(cgl.Fourier2Config(aaErg), [0, d, 0, nstp*h])
    
