from py_CQCGL_threads import *
from personalFunctions import *

case = 60

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
    cgl.Method = 1

    a0 = np.loadtxt('init.dat')
    aa = cgl.intg(a0, h, nstp, 10)
    lte1 = cgl.lte()

    # plot 3d and heat profiles
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T],
                               tickSize=15, axisLabelSize=25)
    plotConfigWireFourier(cgl, aa[::40].copy(), [0, d, 0, T])

    # plot single soliton profile
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$x$', r'$|A|$'],
                   axisLabelSize=25, tickSize=15)
    A1 = cgl.Fourier2Config(aa[0])
    A2 = cgl.Fourier2Config(aa[2100])    # t = 2.1
    ax.plot(np.linspace(0, d, len(A1)), np.abs(A1), lw=1.5, ls='-', c='b')
    ax.plot(np.linspace(0, d, len(A2)), np.abs(A2), lw=1.5, ls='-', c='r')
    ax2d(fig, ax)

    cgl.Method = 2
    aa2 = cgl.intg(a0, h, nstp, 10)
    lte2 = cgl.lte()
    
    # plot lte
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$t$', r'$LTE$'],
                   axisLabelSize=25, tickSize=15,
                   yscale='log')
    ax.plot(np.linspace(0, T, len(lte1)), lte1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(lte2)), lte2, lw=2, ls='--', c='b',
            label='Krogstad')
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
    cgl.changeOmega(-176.67504941219335)
    
    cgl.Method = 1
    aa = cgl.aintg(a0, 1e-3, T, 40)
    Ts1 = cgl.Ts()
    lte1 = cgl.lte()
    hs1 = cgl.hs()
    print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe

    cgl.Method = 2
    aa2 = cgl.aintg(a0, 1e-3, T, 40)
    Ts2 = cgl.Ts()
    lte2 = cgl.lte()
    hs2 = cgl.hs()
    print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe

    # plot heat map
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T],
                               tt=Ts1, yls=range(0, T+1),
                               tickSize=15, axisLabelSize=25)
    
    # plot the accumlated time.
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$t$'],
                   axisLabelSize=25, tickSize=15)
    ax.plot(np.linspace(0, T, len(Ts1)), Ts1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(Ts2)), Ts2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot lte
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$LTE$'],
                   axisLabelSize=25, tickSize=15,
                   ylim=[1e-11, 1e-10],
                   yscale='log')
    ax.plot(np.linspace(0, T, len(lte1)), lte1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(lte2)), lte2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot hs
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$h$'],
                   axisLabelSize=25, tickSize=18,
                   ylim=[8e-6, 1e-4],
                   yscale='log')
    ax.plot(np.linspace(0, T, len(hs1)), hs1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(hs2)), hs2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot the real part of the zeros mode
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$Re(a_0)$'],
                   axisLabelSize=25, tickSize=18)
    n = aa.shape[0]
    ids = np.arange(3*n/4, n)
    ax.plot(np.linspace(3*T/4, T, len(ids)), aa[ids, 0], lw=1, ls='-', c='r')
    ax2d(fig, ax)

if case == 40:
    """
    time step adaption in co-moving frame
    """
    N = 1024
    d = 30
    di = 0.06
    T = 8
    
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)
    a0 = np.loadtxt('init.dat')
    cgl.changeOmega(-176.67504941219335)
    
    cgl.Method = 1
    aa = cgl.aintg(a0, 1e-3, T, 40)
    Ts1 = cgl.Ts()
    lte1 = cgl.lte()
    hs1 = cgl.hs()
    print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe

    cgl.Method = 2
    aa2 = cgl.aintg(a0, 1e-3, T, 40)
    Ts2 = cgl.Ts()
    lte2 = cgl.lte()
    hs2 = cgl.hs()
    print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe

    # plot heat map
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T],
                               tt=Ts1, yls=range(0, T+1),
                               tickSize=15, axisLabelSize=25)
    
    # plot the accumlated time.
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$t$'],
                   axisLabelSize=25, tickSize=15)
    ax.plot(np.linspace(0, T, len(Ts1)), Ts1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(Ts2)), Ts2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot lte
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$LTE$'],
                   axisLabelSize=25, tickSize=15,
                   ylim=[1e-11, 1e-10],
                   yscale='log')
    ax.plot(np.linspace(0, T, len(lte1)), lte1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(lte2)), lte2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot hs
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$h$'],
                   axisLabelSize=25, tickSize=18,
                   ylim=[8e-6, 1e-3],
                   yscale='log')
    ax.plot(np.linspace(0, T, len(hs1)), hs1, lw=1.5, ls='-', c='r',
            label='Cox-Matthews')
    ax.plot(np.linspace(0, T, len(hs2)), hs2, lw=2, ls='--', c='b',
            label='Krogstad')
    ax2d(fig, ax)

    # plot the real part of the zeros mode
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\tau$', r'$Re(a_0)$'],
                   axisLabelSize=25, tickSize=18)
    n = aa.shape[0]
    ids = np.arange(3*n/4, n)
    ax.set_ylim([-1000, 1000])
    ax.plot(np.linspace(3*T/4, T, len(ids)), aa[ids, 0], lw=1, ls='-', c='r')
    ax2d(fig, ax)


if case == 50:
    """
    plot the relative equilibrium
    """
    N = 1024
    d = 30
    di = 0.06
   
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          di, 1)
    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)

    # plot the solition profile
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$x$', r'$|A(x)|$'],
                   axisLabelSize=25, tickSize=18)
    A1 = cgl.Fourier2Config(a0)
    ax.plot(np.linspace(0, d, len(A1)), np.abs(A1), lw=1.5, ls='-', c='r')
    ax2d(fig, ax)

if case == 60:
    """
    collect the performance of static and co-moving frames
    """
    N = 1024
    d = 30
    di = 0.06
    T = 8

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, 0, 4)
    a0 = np.loadtxt('init.dat')
    rtols = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    
    R1 = np.zeros((2, len(rtols)))
    N1 = np.zeros((2, len(rtols)))
    for i in range(len(rtols)):
        cgl.rtol = rtols[i]
        print i, cgl.rtol

        cgl.Method = 1
        aa = cgl.aintg(a0, 1e-3, T, 1000000)
        print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe
        R1[0, i] = cgl.NReject
        N1[0, i] = cgl.NSteps
        
        cgl.Method = 2
        aa2 = cgl.aintg(a0, 1e-3, T, 1000000)
        print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe
        R1[1, i] = cgl.NReject
        N1[1, i] = cgl.NSteps
       
    cgl.changeOmega(-176.67504941219335)
    R2 = np.zeros((2, len(rtols)))
    N2 = np.zeros((2, len(rtols)))
    for i in range(len(rtols)):
        cgl.rtol = rtols[i]
        print i, cgl.rtol

        cgl.Method = 1
        aa = cgl.aintg(a0, 1e-3, T, 1000000)
        print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe
        R2[0, i] = cgl.NReject
        N2[0, i] = cgl.NSteps
        
        cgl.Method = 2
        aa2 = cgl.aintg(a0, 1e-3, T, 1000000)
        print cgl.NSteps, cgl.NReject, cgl.NCallF, cgl.NCalCoe
        R2[1, i] = cgl.NReject
        N2[1, i] = cgl.NSteps
        
    # save result
    # np.savez_compressed('perf', R1=R1, N1=N1, R2=R2, N2=N2)

    # plot the number of integration steps
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\epsilon$', r'$N_s$'],
                   axisLabelSize=25, tickSize=18, yscale='log')
    ax.set_xscale('log')
    ax.plot(rtols, N1[0], lw=1.5, ls='-', marker='o', ms=7,
            mew=2, mfc='none', mec='r', c='r', label='Static Cox-Matthews')
    ax.plot(rtols, N1[1], lw=1.5, ls='-', marker='s', ms=10,
            mew=2, mfc='none', mec='g', c='g', label='Static Krogstad')
    ax.plot(rtols, N2[0], lw=1.5, ls='--', marker='o', ms=7,
            mew=2, mfc='none', mec='c', c='c', label='Co-moving Cox-Matthews')
    ax.plot(rtols, N2[1], lw=1.5, ls='--', marker='s', ms=10,
            mew=2, mfc='none', mec='k', c='k', label='Co-moving Krogstad')
    ax.grid(True, which='both')
    ax2d(fig, ax)

    # plot the number of rejections
    fig, ax = pl2d(size=[6, 4.5], labs=[r'$\epsilon$', r'$N_r$'],
                   axisLabelSize=25, tickSize=18)
    ax.set_xscale('log')
    ax.plot(rtols, R1[0], lw=1.5, ls='-', marker='o', ms=7,
            mew=2, mfc='none', mec='r', c='r', label='Static Cox-Matthews')
    ax.plot(rtols, R1[1], lw=1.5, ls='-', marker='s', ms=10,
            mew=2, mfc='none', mec='g', c='g', label='Static Krogstad')
    ax.plot(rtols, R2[0], lw=1.5, ls='--', marker='o', ms=7,
            mew=2, mfc='none', mec='c', c='c', label='Co-moving Cox-Matthews')
    ax.plot(rtols, R2[1], lw=1.5, ls='--', marker='s', ms=10,
            mew=2, mfc='none', mec='k', c='k', label='Co-moving Krogstad')
    ax.grid(True, which='both')
    ax2d(fig, ax)
