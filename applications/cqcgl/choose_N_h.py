import h5py
from pylab import *
import numpy as np
from numpy.linalg import norm, eig
from time import time
from py_cqcgl1d_threads import pyCqcgl1d, pyCqcglRPO
from personalFunctions import *


case = 1


if case == 1:
    """
    use different N and h to view the explosion process.
    Try to find the optimal values.
    """
    N = 512
    d = 50
    h = 0.0001

    cgl = pyCqcgl1d(N, d, h, True, 0,
                    -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
                    4)
    Ndim = cgl.Ndim

    nstp3 = 200000
    a0Erg3 = cgl.Config2Fourier(centerRand(2*N, 0.25))
    aaErg3 = cgl.intg(a0Erg3, 20000, 20000)
    a0Erg3 = aaErg3[-1]
    aaErg3 = cgl.intg(a0Erg3, nstp3, 1)
    aaErgHat3, th3, phi3 = cgl.orbit2slice(aaErg3)
    aaErgTilde3 = cgl.reduceReflection(aaErgHat3)

    # plot3dfig(aaErgTildeProj3[:, 0], aaErgTildeProj3[:, 1], aaErgTildeProj3[:, 2])
    # plot the variance => indicator of the integration error.
    plot1dfig(aaErg3[:, 0])
    plot1dfig(aaErg3[:-1, 0]-aaErg3[1:, 0])
    plot1dfig(abs(aaErg3[:, Ndim/2]), yscale='log')
    # plotConfigSpace(cgl.Fourier2Config(aaErg3), [0, d, 0, nstp3*h])
