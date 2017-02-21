from py_CQCGL2d import *
from py_CQCGL2dReq import *
from personalFunctions import *

case = 40

if case == 10:
    """
    test find req
    """
    N = 1024
    d = 30
    di = 0.05
    
    cgl = pyCQCGL2dReq(N, d, 4.0, 0.8, 0.01, di, 4)
    c2dp = CQCGL2dPlot(d, d)
    a0 = c2dp.load('ex.h5', 721)
    # c2dp.plotOneState(cgl, 'ex.h5', 400)
    cgl.GmresRestart = 300
    cgl.GmresRtol = 1e-6
    th = -cgl.optReqTh(a0)
    print th
    x, wthx, wthy, wphi, err = cgl.findReq_hook(a0, th[0], th[1], th[2])
    
if case == 20:
    """
    find the best candidate for req
    """
    N = 1024
    d = 30
    di = 0.05
    
    cgl = pyCQCGL2dReq(N, d, 4.0, 0.8, 0.01, di, 4)
    c2dp = CQCGL2dPlot(d, d)
    es = []
    for i in range(750):
        a0 = c2dp.load('ex.h5', i)
        th = -cgl.optReqTh(a0)
        e = norm(cgl.velocityReq(a0, th[0], th[1], th[2]))
        es.append(e)
    es = np.array(es)

if case == 30:
    """
    find possible different localized invariant structures.
    """
    N = 1024
    d = 30
    di = 0.05
    
    cgl = pyCQCGL2dReq(N, d, 4.0, 0.8, 0.01, di, 4)
    c2dp = CQCGL2dPlot(d, d)
    A0 = 4*centerRand2d(N, N, 0.2, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    cgl.GmresRestart = 300
    cgl.GmresRtol = 1e-6
    th = -cgl.optReqTh(a0)
    print th
    x, wthx, wthy, wphi, err = cgl.findReq_hook(a0, th[0], th[1], th[2])
    print "norm", norm(x)

if case == 40:
    """
    check the accuracy of the req
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6
    reqFile = "../../data/cgl/req2dBiGi.h5"

    cgl = pyCQCGL2dReq(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4)
    c2dp = CQCGL2dPlot(d, d)
    a0, wthx0, wthy0, wphi0, err = c2dp.loadReq(reqFile, c2dp.toStr(Bi, Gi, 1))
    
    v = cgl.velocityReq(a0, wthx0, wthy0, wphi0)
    
    print norm(v)

if case == 60:
    """
    find possible different localized invariant structures.
    for L = 50
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6
    
    cgl = pyCQCGL2dReq(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4)
    c2dp = CQCGL2dPlot(d, d)
    a0 = c2dp.load('ex.h5', 350)
    # A0 = 1*centerRand2d(N, N, 0.2, 0.2, True)
    # a0 = cgl.Config2Fourier(A0)
    cgl.GmresRestart = 300
    cgl.GmresRtol = 1e-6
    th = -cgl.optReqTh(a0)
    print th
    x, wthx, wthy, wphi, err = cgl.findReq_hook(a0, th[0], th[1], th[2])
    print "norm", norm(x)
