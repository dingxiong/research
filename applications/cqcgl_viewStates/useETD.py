from personalFunctions import *
from py_cqcgl1d import pyCqcgl1d, pyCqcglETD

case = 1

if case == 1:
    N = 1024
    d = 30
    h = 0.001

    W = 0
    di = 0.05
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)

    cgl = pyCqcgl1d(N, d, h, False, 0, 4.0, 0.8, 0.01, di, 4)
    cgletd = pyCqcglETD(N, d, W, 4.0, 0.8, 0.01, di)
    cgletd.setRtol(1e-8)
    tt, aa = cgletd.etd(a0, 1, 0.001, 1, 2, True)
    hs = cgletd.hs()[1:]
    duu = cgletd.duu()[1:]
    plot1dfig(duu, yscale='log')
    plot1dfig(hs)
    print cgletd.etdParam()
