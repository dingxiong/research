from py_CQCGL2d import *
from py_CQCGL2dReq import *
from personalFunctions import *

case = 10

if case == 10:
    """
    test find req
    """
    N = 1024
    d = 30
    di = 0.05
    
    cgl = pyCQCGL2dReq(N, d, 4.0, 0.8, 0.01, di, 4)
    c2dp = CQCGL2dPlot(d, d)
    a0 = c2dp.load('ex.h5', 500)
    x, wthx, wthy, wphi = cgl.findReq_hook(a0, rand(), rand(), rand())
    
if case == 30:
    """
    test plotting a sequence
    """
    N = 1024
    d = 30
    di = 0.05

    cgl = pyCQCGL2d(N, d, 4.0, 0.8, 0.01, di, 4)
    fileName = 'ex2.h5'
    c2dp = CQCGL2dPlot(d, d)
    x = c2dp.loadSeq(fileName, 0, 0, range(1000))
    plot1dfig(x.real)
    plot1dfig(x.imag)


