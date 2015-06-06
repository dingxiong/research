import h5py
from pylab import *
import numpy as np
from time import time
from py_cqcgl1d import pyCqcgl1d
from personalFunctions import *


def cqcglFindReq(fileName, frac=0.3, MaxIter=300, Ntrial=1000):
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)

    ReqNum = 1
    for i in range(Ntrial):
        A0 = 3*centerRand(2*N, frac)
        a0 = cgl.Config2Fourier(A0).squeeze()
        wth0 = rand()
        wphi0 = rand()
        a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, MaxIter,
                                        1e-12, True, False)
        if err < 1e-12:
            print "found : " + str(ReqNum)
            cqcglSaveReq(fileName, str(ReqNum), a, wth, wphi, err)
            ReqNum += 1


case = 3

if case == 1:
    """
    refine the old relative equilibria
    """
    N = 256
    d = 50
    h = 0.005

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/reqOld.h5', 'r')
    req = '/req1/'
    a0 = f[req+'a0'].value
    wth0 = f[req+'w1'].value[0]
    wphi0 = f[req+'w2'].value[0]

    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 100, 1e-12, True, True)
    cqcglSaveReq("req.hdf5", '1', a, wth, wphi, err)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])

if case == 2:
    """
    try different guesses to find relative equilibria
    Just for test purpose
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    
    A0 = 3*centerRand(2*N, 0.25)
    a0 = cgl.Config2Fourier(A0).squeeze()
    wth0 = rand()
    wphi0 = rand()
    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 500, 1e-12, True, True)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])
    plot1dfig(a)
    plotOneConfigFromFourier(cgl, a)
    print wth, wphi
    # cqcglSaveReq("req2.hdf5", '16', a, wth, wphi, err)

if case == 3:
    """
    run a long time to collect relative equilibria
    """
    cqcglFindReq("req3.hdf5", 0.3, 300, 1000)
