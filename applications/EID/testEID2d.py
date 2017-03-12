from personalFunctions import *
# from py_CQCGL1d import *
from py_CQCGL2d import *


case = 80

labels = ["IFRK4(3)", "IFRK5(4)",
          "ERK4(3)2(2)", "ERK4(3)3(3)", "ERK4(3)4(3)", "ERK5(4)5(4)",
          "SSPP4(3)"]
mks = ['o', 's', '+', '^', 'x', 'v', 'p']
Nscheme = len(labels)
lss = ['--', '-.', ':', '-', '-', '-', '-']


if case == 60:
    """
    plot the estimated local error of 2d cgcgl by constant schemes
    """
    err = np.loadtxt('data/cqcgl2d_N30_lte.dat')
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    n = err.shape[0]
    T = 20.0
    x = np.arange(1, n+1) * T/n
    
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'estimated local error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   # ylim=[1e-17, 1e-7],
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(x, err[:, i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    fig.tight_layout(pad=0)
    ax.legend(loc='lower right', ncol=2)
    plt.show(block=False)
    # ax2d(fig, ax, loc='lower right')

    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'estimated local error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   # ylim=[1e-30, 1e-8],
                   yscale='log')
    for i in range(Nscheme):
        ax.plot(x, err[:, Nscheme+i], lw=2, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=4)
    ax.locator_params(axis='x', nbins=5)
    ax2d(fig, ax, loc='lower right')

if case == 70:
    """
    plot the time steps used in the process
    """
    hs = []
    for i in range(Nscheme):
        x = np.loadtxt('data/cqcgl2d_N50_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 20.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   # ylim=[2e-5, 2e-3],
                   yscale='log')
    for i in range(Nscheme):
        n = len(hs[i])
        x = np.arange(1, n+1) * T/n
        ax.plot(x, hs[i], lw=2, ls=lss[i], label=labels[i])
    # ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    fig.tight_layout(pad=0)
    ax.legend(loc='lower right', ncol=2)
    plt.show(block=False)
    # ax2d(fig, ax, loc='upper left')
    
if case == 80:
    """
    same as case 70 but in comoving frame
    plot the time steps used in the process
    """
    hs = []
    for i in range(Nscheme):
        x = np.loadtxt('data/cqcgl2d_N60_comoving_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 20.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   # ylim=[8e-6, 2e-2],
                   yscale='log')
    for i in range(Nscheme):
        n = len(hs[i])
        x = np.arange(1, n+1) * T/n
        ax.plot(x, hs[i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    fig.tight_layout(pad=0)
    ax.legend(loc='lower right', ncol=2)
    plt.show(block=False)
    # ax2d(fig, ax, loc='upper left')
