from personalFunctions import *
from py_CQCGL_threads import *
from py_CQCGL2d import *

case = 60

labels = ["Cox-Matthews", "Krogstad", "Hochbruck-Ostermann",
          "Luan-Ostermann", "IFRK43", "IFRK54", "SSPP43"]
mks = ['o', 's', '+', '^', 'x', 'v', 'p']
Nscheme = len(labels)
lss = ['--', '-.', ':', '-', '-', '-', '-']

if case == 10:
    lte = np.loadtxt('data/N10_lte.dat')
    T = 10.25
    n = lte.shape[0]
    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    for i in range(6):
        ax.plot(np.linspace(0, 2*T, n), lte[:, i], lw=2, label=labels[i])
    ax2d(fig, ax)
        
if case == 20:
    ltes = []
    for i in range(3):
        k = 10**i
        lte = np.loadtxt('data/KS_N20_lte' + str(k) + '.dat')
        ltes.append(lte)
    T = 10.25
    
    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, T])
    k = 5
    for i in range(3):
        n = len(ltes[i][:, k])
        ax.plot(np.linspace(0, T, n), ltes[i][:, k], lw=2)
    ax2d(fig, ax)

    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    k = 1
    for i in range(Nscheme):
        n = len(ltes[k][:, i])
        ax.plot(np.linspace(0, 2*T, n), ltes[k][:, i], lw=2, label=labels[i])
    ax2d(fig, ax)

if case == 30:
    err = np.loadtxt('data/KS_N30_err.dat')
    h = err[:, 0]
    fig, ax = pl2d(size=[6, 5], labs=[r'$h$', 'relative error'],
                   axisLabelSize=20, tickSize=15,
                   xlim=[5e-5, 4e-1],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(h, err[:, i+1], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.plot([1e-2, 1e-1], [1e-10, 1e-6], lw=2, c='k', ls='--')
    ax.plot([4e-2, 4e-1], [1e-11, 1e-6], lw=2, c='k', ls='--')
    # ax.grid(True, which='both')
    ax.locator_params(axis='y', numticks=4)
    ax2d(fig, ax)

###############################################################################
# 1d cqcgl
if case == 40:
    """
    plot 1d cqcgl
    """
    N = 1024
    d = 30
    di = 0.06

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, -1, 4)
    aa = np.loadtxt('aa.dat')
    aa2 = np.loadtxt('aa2.dat')
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, 4])
    plotConfigSpaceFromFourier(cgl, aa2, [0, d, 0, 4])


if case == 50:
    """
    plot the relative errors of 1d cgcgl by constant schemes
    """
    err = np.loadtxt('data/cqcgl1d_N20_err.dat')
    h = err[:, 0]
    fig, ax = pl2d(size=[6, 5], labs=[r'$h$', 'relative error'],
                   axisLabelSize=20, tickSize=15,
                   xlim=[1e-8, 5e-3],
                   ylim=[1e-9, 1e1],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(h, err[:, i+1], lw=1.5, marker=mks[i], mfc='none',
                label=labels[i])
    ax.plot([1e-5, 1e-3], [1e-6, 1e2], lw=2, c='k', ls='--')
    ax.plot([1e-4, 1e-3], [1e-8, 1e-3], lw=2, c='k', ls='--')
    # ax.grid(True, which='both')
    ax.locator_params(axis='y', numticks=4)
    ax2d(fig, ax, loc='upper left')

if case == 60:
    """
    plot the estimated local error of 1d cgcgl by constant schemes
    """
    err = np.loadtxt('data/cqcgl1d_N30_lte.dat')
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    n = err.shape[0]
    T = 4.0
    x = np.arange(1, n+1) * T/n
    
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'estimated local error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[1e-22, 1e-8],
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(x, err[:, i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    ax2d(fig, ax, loc='lower right')

    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'estimated local error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[1e-30, 1e-8],
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(x, err[:, Nscheme+i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=4)
    ax.locator_params(axis='x', nbins=5)
    ax2d(fig, ax, loc='lower right')

if case == 65:
    """
    plot the time steps used in the process
    """
    hs = []
    for i in range(Nscheme):
        x = np.loadtxt('data/cqcgl1d_N50_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 4.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[5e-6, 1e-2],
                   yscale='log')
    for i in range(Nscheme):
        n = len(hs[i])
        x = np.arange(1, n+1) * T/n
        ax.plot(x, hs[i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=4)
    ax.locator_params(axis='x', nbins=5)
    ax2d(fig, ax, loc='upper left')
    
if case == 66:
    """
    same as case 65 but in comoving frame
    plot the time steps used in the process
    """
    hs = []
    for i in range(Nscheme):
        x = np.loadtxt('data/cqcgl1d_N60_comoving_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 4.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[5e-6, 2e-1],
                   yscale='log')
    for i in range(Nscheme):
        n = len(hs[i])
        x = np.arange(1, n+1) * T/n
        ax.plot(x, hs[i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    ax2d(fig, ax, loc='upper left')

if case == 67:
    """
    static frame
    plot the relative error, Nab, Nn vs rtol
    """
    err = np.loadtxt('data/cqcgl1d_N70_stat.dat')
    rtol = err[:, 0]
    
    # plot relative error
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', 'relative error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[1e-9, 3e-2],
                   xscale='log',
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+1], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='bottem right')
    
    # plot Nab
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Nab$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[4e2, 2400],
                   # yscale='log',
                   xscale='log')
    for i in range(4):
        ax.plot(rtol, err[:, 4*i+2], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper left')

    # plot Nn
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Nn$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[5e4, 1e9],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+3], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper right')

    # plot Wt
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Wt$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[2e0, 1e4],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+4], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper right')

if case == 68:
    """
    comoving frame
    plot the relative error, Nab, Nn vs rtol
    """
    err = np.loadtxt('data/cqcgl1d_N70_comoving.dat')
    rtol = err[:, 0]
    
    # plot relative error
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', 'relative error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[1e-9, 4e-0],
                   xscale='log',
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+1], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='lower right')
    
    # plot Nab
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Nab$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[500, 4000],
                   # yscale='log',
                   xscale='log')
    for i in range(4):
        ax.plot(rtol, err[:, 4*i+2], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper left')

    # plot Nn
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Nn$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[4e4, 1e9],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+3], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper right')

    # plot Wt
    fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', r'$Wt$'],
                   axisLabelSize=20, tickSize=15,
                   ylim=[2e0, 1e4],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 4*i+4], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax2d(fig, ax, loc='upper right')

if case == 69:
    """
    Compare the accuracy of static and comoving frames
    """
    err = np.loadtxt('data/cqcgl1d_N70_stat.dat')
    err2 = np.loadtxt('data/cqcgl1d_N70_comoving.dat')
    rtol = err[:, 0]

    for i in range(4):
        fig, ax = pl2d(size=[6, 5], labs=[r'$rtol$', 'relative error'],
                       axisLabelSize=20, tickSize=15,
                       # xlim=[1e-8, 5e-3],
                       ylim=[1e-9, 4e-0],
                       xscale='log',
                       yscale='log')
        ax.plot(rtol, err[:, 4*i+1], lw=1.5, marker=mks[0], mfc='none',
                ms=8, label='static')
        ax.plot(rtol, err2[:, 4*i+1], lw=1.5, marker=mks[1], mfc='none',
                ms=8, label='comoving')
        ax.locator_params(axis='y', numticks=5)
        ax.locator_params(axis='x', numticks=5)
        ax.text(0.3, 0.9, labels[i], fontsize=15, horizontalalignment='center',
                verticalalignment='center',  bbox=dict(ec='black', fc='none'),
                transform=ax.transAxes)
        ax2d(fig, ax, loc='upper right')

###############################################################################
# 2d cqcgl
if case == 70:
    """
    check the integration process is correct.
    Save the figure and compare
    """
    N = 1024
    d = 30
    di = 0.05
    cgl = pyCQCGL2d(N, d, 4.0, 0.8, 0.01, di, 4)
    c2dp = CQCGL2dPlot(d, d)

    a1 = c2dp.load('aa.h5', 399)
    a2 = c2dp.load('aa2.h5', 400)
    print np.max(np.abs(a1-a2))
    c2dp.savePlots(cgl, "aa.h5", range(400), 'fig1')
    c2dp.savePlots(cgl, "aa2.h5", range(401), 'fig2')
