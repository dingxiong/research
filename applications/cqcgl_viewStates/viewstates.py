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

case = 2

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

    stabMat = cgl.stabReq(a0, th0, phi0).T  # transpose is needed
    eigvalues, eigvectors = eig(stabMat)
    eigvalues, eigvectors = sortByReal(eigvalues, eigvectors)
    print eigvalues[:10]

    # make sure you make a copy because Fourier2Config takes contigous memory
    tmpr = eigvectors[:, 0].real.copy()
    tmpr = cgl.Fourier2Config(tmpr).squeeze()
    tmpi = eigvectors[:, 0].imag.copy()
    tmpi = cgl.Fourier2Config(tmpi).squeeze()
    plotOneConfig(tmpi)
 

    
    nstp = 2000
    aa = cgl.intg(a0, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])
    raa, th, phi = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(raa), [0, d, 0, nstp*h])
    plotOneFourier(aa[-1])
