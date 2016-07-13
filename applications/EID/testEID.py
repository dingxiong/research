from personalFunctions import *
from py_CQCGL_threads import *

case = 40

labels = ["Cox-Matthews", "Krogstad", "Hochbruck-Ostermann",
          "Luan-Ostermann", "IFRK43", "IFRK54"]

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
        lte = np.loadtxt('data/N20_lte' + str(k) + '.dat')
        ltes.append(lte)
    T = 10.25
    
    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    k = 5
    for i in range(3):
        n = len(ltes[i][:, k])
        ax.plot(np.linspace(0, 2*T, n), ltes[i][:, k], lw=2)
    ax2d(fig, ax)

    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    k = 0
    for i in range(6):
        n = len(ltes[k][:, i])
        ax.plot(np.linspace(0, 2*T, n), ltes[k][:, i], lw=2, label=labels[i])
    ax2d(fig, ax)

if case == 30:
    err = np.loadtxt('data/N30_err.dat')
    mks = ['o', 's', '+', '^', 'x', 'v']
    h = err[:, 0]
    fig, ax = pl2d(size=[6, 5], labs=[r'$h$', 'relative error'],
                   axisLabelSize=15, xlim=[2e-4, 4e-1],
                   xscale='log', yscale='log')
    for i in range(6):
        ax.plot(h, err[:, i+1], lw=1.5, marker=mks[i], label=labels[i])
    ax.plot([1e-2, 1e-1], [1e-10, 1e-6], lw=2, c='k', ls='--')
    ax.plot([4e-2, 4e-1], [1e-11, 1e-6], lw=2, c='k', ls='--')
    # ax.grid(True, which='both')
    ax.locator_params(axis='y', numticks=4)
    ax2d(fig, ax)

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
