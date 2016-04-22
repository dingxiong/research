import numpy as np
from numpy.random import rand
from numpy.fft import fft, ifft
from time import time
from py_CQCGL_threads import *
from personalFunctions import *

case = 30

if case == 10:
    """
    test fourier to physical states and vice verse
    """
    N = 1024
    d = 30
    di = 0.05

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, -1, 4)

    Ndim = cgl.Ndim
    aa = rand(100, Ndim)
    AA = cgl.Fourier2Config(aa)
    aAA = cgl.Fourier2ConfigMag(aa)
    phase = cgl.Fourier2Phase(aa)
    aa2 = cgl.Config2Fourier(AA)
    print norm(aa-aa2)
    print norm(np.abs(AA)-aAA)
    
if case == 20:
    """
    test the const time step integration routines
    """
    N = 1024
    d = 30
    di = 0.06

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, -1, 4)

    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)

    t = time()
    # aa, daa = cgl.intgj(a0, 1000, 1, 1000)
    aa = cgl.intg(a0, 0.0001, 40000, 1)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, 10])
    print time() - t

if case == 30:
    """
    test time step adaptive integration routines
    """
    N = 1024
    d = 30
    di = 0.06

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, -1, 4)

    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)

    t = time()
    cgl.changeOmega(-176.67504941219335)
    tt, aa = cgl.aintg(a0, 0.001, 4,  1)
    tt, aa = cgl.aintg(aa[-1], 0.001, 4,  1)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, 10])
    plot1dfig(cgl.hs(), yscale='log')
    plot1dfig(cgl.lte(), yscale='log')
    print time() - t

# compare fft with Fourier2Config
if case == 2:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(510)
    aa = cgl.intg(a0, 1000, 1)
    t_init = time()
    for i in range(100):
        AA = cgl.Fourier2Config(aa)
    print time() - t_init

    t_init = time()
    for i in range(100):
        aa2 = aa[:, ::2]+1j*aa[:, 1::2]
        aa2 = np.hstack((aa2[:, :128], 1j*np.zeros([1001, 1]),
                         aa2[:, -127:]))
        AA2 = ifft(aa2, axis=1)
    print time() - t_init

    AA3 = np.zeros((AA2.shape[0], AA2.shape[1]*2))
    AA3[:, ::2] = AA2.real
    AA3[:, 1::2] = AA2.imag

    print np.amax(np.abs(AA - AA3))

# test plotting fiugres
if case == 3:
    cgl = pyCqcgl1d(512, 50, 0.01, False, 0,
                    -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
                    4)
    A0 = rand(512*2)
    a0 = cgl.Config2Fourier(A0)
    aa = cgl.intg(a0, 1000, 1)
    AA = cgl.Fourier2Config(aa)
    plotConfigSpace(AA, [0, 50, 0, 1000*0.01])

# test reflection
if case == 4:
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(510)
    aa = cgl.intg(a0, 1000, 1)
    raa = cgl.reflect(aa)
    print aa[:2, :10]
    print raa[:2, :10]
    print raa[:2, -10:]
    print aa[:2, -10:]

if case == 5:
    # test continous symmetry reduction
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0 = rand(510)
    aa = cgl.intg(a0, 10, 1)
    aaHat, th, phi = cgl.orbit2slice(aa)
    print aaHat.shape
    print aaHat[:, 3]
    print aaHat[:, -1]

if case == 6:
    # test the reduceReflection function
    cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    A0 = centerRand(512, 0.2)
    a0 = cgl.Config2Fourier(A0)

    nstp = 10000
    aa = cgl.intg(a0, nstp, 1)
    plotConfigSpaceFromFourier(cgl, aa, [0, 50, 0, nstp*0.01])

    aaHat, th, phi = cgl.orbit2slice(aa)
    plotConfigSpaceFromFourier(cgl, aaHat, [0, 50, 0, nstp*0.01])
    aaTilde = cgl.reduceReflection(aaHat)
    plotConfigSpaceFromFourier(cgl, aaTilde, [0, 50, 0, nstp*0.01])

    raa = cgl.reflect(aa)
    aaHat2, th2, phi2 = cgl.orbit2slice(raa)
    plotConfigSpaceFromFourier(cgl, aaHat2, [0, 50, 0, nstp*0.01])
    aaTilde2 = cgl.reduceReflection(aaHat2)
    plotConfigSpaceFromFourier(cgl, aaTilde2, [0, 50, 0, nstp*0.01])

    aaHat3 = cgl.Rotate(aaHat, pi, pi)
    plotConfigSpaceFromFourier(cgl, aaHat3, [0, 50, 0, nstp*0.01])
    aaTilde3 = cgl.reduceReflection(aaHat3)
    plotConfigSpaceFromFourier(cgl, aaTilde3, [0, 50, 0, nstp*0.01])

    print np.allclose(aaTilde, aaTilde2)
    print np.allclose(aaTilde, aaTilde3)
    print np.amax(np.abs(aaTilde - aaTilde2))
    print np.argmax(np.abs(aaTilde - aaTilde2))
    print np.amax(np.abs(aaTilde - aaTilde3))
    print np.argmax(np.abs(aaTilde - aaTilde3))
