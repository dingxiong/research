from py_CQCGL2d import *
from personalFunctions import *

case = 40

if case == 10:
    """
    test the plot function
    """
    N = 1024
    d = 30
    di = 0.05

    cgl = pyCQCGL2d(N, d, 4.0, 0.8, 0.01, di, 4)
    Ns = 15000
    skip = 20
    cgl.constETDPrint = 200
    fileName = 'ex.h5'

    A0 = 3*centerRand2d(N, N, 0.2, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    # aa = cgl.intg(a0, 0.001, Ns, skip, True, fileName)
    c2dp = CQCGL2dPlot(d, d)
    # c2dp.plotOneState(cgl, fileName, 100)
    c2dp.savePlots(cgl, fileName, range(Ns/skip+1), 'fig3',
                   plotType=2, onlyMovie=True, size=[12, 6])

if case == 20:
    """
    integrate using current data
    """
    N = 1024
    d = 30
    di = 0.05
    
    cgl = pyCQCGL2d(N, d, 4.0, 0.8, 0.01, di, 4)
    Ns = 2000
    skip = 2
    cgl.constETDPrint = 100
    fileName = 'ex2.h5'

    c2dp = CQCGL2dPlot(d, d)
    a0 = c2dp.load('ex.h5', 750)
    aa = cgl.intg(a0, 0.001, Ns, skip, True, fileName)
    
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

if case == 40:
    """
    test stabReq
    """
    N = 1024
    d = 30
    di = 0.05

    cgl = pyCQCGL2d(N, d, 4.0, 0.8, 0.01, di, 4)
    fileName = 'ex2.h5'
    c2dp = CQCGL2dPlot(d, d)
    x = c2dp.load('ex.h5', 500)
    v = rand(cgl.Me, cgl.Ne)
    ax = cgl.stab(x, v)
