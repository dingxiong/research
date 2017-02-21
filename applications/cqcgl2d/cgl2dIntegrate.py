from py_CQCGL2d import *
from personalFunctions import *

case = 10

if case == 10:
    """
    test the plot function
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6

    cgl = pyCQCGL2d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4)
    Ns = 10000
    skip = 20
    cgl.constETDPrint = 200
    fileName = 'ex.h5'

    A0 = 3*centerRand2d(N, N, 0.2, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    aa = cgl.intg(a0, 0.005, Ns, skip, True, fileName)
    c2dp = CQCGL2dPlot(d, d)
    # c2dp.oneState(cgl, fileName, 100)
    c2dp.savePlots(cgl, fileName, range(Ns/skip+1), 'fig3',
                   plotType=2, size=[12, 6])

