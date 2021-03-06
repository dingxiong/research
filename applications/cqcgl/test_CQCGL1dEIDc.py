from cglHelp import *
from py_CQCGL1d import *
from py_CQCGL1dEIDc import *
import time

case = 20

if case == 10:
    """
    Test the integration result by EID method
    """
    N, d = 1024 , 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi = 2, -2

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=2)
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50

    t = time.time()
    aa = cgl.intg(aE, h, np.int(T/h), 10)
    print time.time() - t 
    cp.config(aa, [0, d, 0, T])

    t = time.time()
    aa2 = EI.intgC(aE, h, T, 10)
    print time.time() - t

    t = time.time()
    aa3  = EI.intg(aE, h, T, 10)
    print time.time() -t 
    cp.config(aa3, [0, d, 0, T])

if case == 20:
    """
    same test as case = 10, but with rpo intial condition
    """
    N, d = 1024 , 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi =  4.8, -4.5

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)

    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5',
                                              rpo.toStr(Bi, Gi, 1), flag=2)
    a0 = x[:cgl.Ndim]
    aE = a0 + 0.001 * norm(a0) * v[0].real

    t = time.time()
    aa = cgl.intg(aE, T/nstp/2, 40*nstp, 10)
    print time.time() - t 
    cp.config(aa, [0, d, 0, 20*T])

    t = time.time()
    aa2 = EI.intgC(aE, T/nstp/2, 20*T, 10)
    print time.time() - t

    t = time.time()
    aa3  = EI.intg(aE, T/nstp, 20*T, 10)
    print time.time() -t 
    cp.config(aa3, [0, d, 0, 20*T])
