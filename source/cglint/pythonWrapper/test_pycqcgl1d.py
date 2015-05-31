import numpy as np
from numpy.random import rand
from numpy.fft import fft, ifft
from time import time
from py_cqcgl1d import pyCqcgl1d
from cqcgl1d import *

case = 4

if case == 1:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(512)
    aa = cgl.intg(a0, 10, 1)
    aa, daa = cgl.intgj(a0, 10, 1, 1)

# compare fft with Fourier2Config
if case == 2:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(512)
    aa = cgl.intg(a0, 1000, 1)
    t_init = time()
    for i in range(100):
        AA = cgl.Fourier2Config(aa)
    print time() - t_init

    t_init = time()
    for i in range(100):
        AA2 = ifft(aa[:, ::2]+1j*aa[:, 1::2], axis=1)
    print time() - t_init

    AA3 = np.zeros((AA2.shape[0], AA2.shape[1]*2))
    AA3[:, ::2] = AA2.real
    AA3[:, 1::2] = AA2.imag

    print np.amax(np.abs(AA - AA3))

# test plotting fiugres
if case == 3:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    A0 = rand(512)
    a0 = cgl.Config2Fourier(A0)
    aa = cgl.intg(a0, 1000, 1)
    AA = cgl.Fourier2Config(aa)
    plotConfigSpace(AA, [0, 50, 0, 1000*0.01])

# test reflection
if case == 4:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(512)
    aa = cgl.intg(a0, 1000, 1)
    raa = cgl.reflect(aa)
    print aa[:2, :10]
    print raa[:2, :4]
    print raa[:2, -10:]
