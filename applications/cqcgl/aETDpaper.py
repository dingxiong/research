from py_CQCGL_threads import *
from personalFunctions import *

case = 20

if case == 10:
    """
    save the same initial condition
    """
    N = 1024
    d = 30
    di = 0.06
    T = 8
    
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)

    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)

    aa = cgl.aintg(a0, 0.001, T, 100000)
    print aa.shape
    np.savetxt('init.dat', aa[-1])

if case == 20:
    """
    constant time step result
    """
    N = 1024
    d = 30
    di = 0.06
    T = 8
    h = 1e-4
    nstp = np.int(T/h)
    
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)

    a0 = np.loadtxt('init.dat')
    aa = cgl.intg(a0, h, nstp, 10)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])
    plotConfigWireFourier(cgl, aa[::10].copy(), [0, d, 0, T])
    plotOneConfigFromFourier(cgl, aa[0])
    # t = 2
    plotOneConfigFromFourier(cgl, aa[2000])
    
    lte = cgl.lte()
    fig, ax = pl2d(size=[8, 6], labs=[r'$t$', r'$LTE$'],
                   axisLabelSize=25, tickSize=15,
                   yscale='log')
    ax.plot(np.linspace(0, T, len(lte)), lte, lw=1)
    ax2d(fig, ax)

if case == 30:
    """
    time step adaption in static frame
    """
    N = 1024
    d = 30
    di = 0.06
    T = 8
    
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)

    a0 = np.loadtxt('init.dat')
    aa = cgl.aintg(a0, 1e-3, T, 10)
    Ts = cgl.Ts()
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T], tt=Ts, yls=range(0, T+1))
    plotOneConfigFromFourier(cgl, aa[0])
    
