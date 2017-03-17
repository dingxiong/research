from cglHelp import *
from py_CQCGL2d import *


case = 90

labels = ["IF4(3)", "IF5(4)",
          "ERK4(3)2(2)", "ERK4(3)3(3)", "ERK4(3)4(3)", "ERK5(4)5(4)",
          "SS4(3)"]
mks = ['o', 's', '+', '^', 'x', 'v', 'p']
Nscheme = len(labels)
lss = ['--', '-.', ':', '-', '-', '-', '-']

if case == 40:
    """
    plot one instance of explosion in heat map.
    Also plot the req in heat map
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6

    cgl = pyCQCGL2d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4)
    cglp = CQCGL2dPlot(d, d)

    fig = plt.figure(figsize=[12, 15])

    states = []
    # load explosion data
    ts = [3, 5, 8, 10, 12]
    for i in range(len(ts)):
        t = ts[i]
        index = np.int(t / 20.0 * 256)
        a = cglp.load('aaCox.h5', index)
        states.append(a)
    # load req data
    a, wthx, wthy, wphi, err = cglp.loadReq('../../data/cgl/req2dBiGi.h5', cglp.toStr(Bi, Gi, 1))
    states.append(a)
    
    for i in range(6):
        ax = fig.add_subplot('32' + str(i+1))
        ax.set_axis_off()
        # ax.tick_params(axis='both', which='major', labelsize=18)
        ax.text(0.1, 0.9, '('+ chr(ord('a') + i) +')', horizontalalignment='center',
                transform=ax.transAxes, fontsize=18, color='white')

        A = cgl.Fourier2Config(states[i])
        aA = np.abs(A).T

        im = ax.imshow(aA, cmap=plt.get_cmap('jet'), aspect='equal',
                       origin='lower', extent=[0, d, 0, d])
        dr = make_axes_locatable(ax)
        cax = dr.append_axes('right', size=0.08, pad=0.05)
        cax.tick_params(labelsize=18)
        plt.colorbar(im, cax=cax, ticks=[1, 2, 3])
        
    ####
    fig.tight_layout(pad=1)
    plt.show(block=False)


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

if case == 90:
    """
    static & comoving frame
    plot the relative error, Nab, Nn vs rtol
    """
    loadFlag = 1

    err = np.loadtxt('data/new_cqcgl2d_N70_stat' + ('0' if loadFlag == 0 else '1') + '.dat')
    rtol = err[:, 0]

    hs = []
    for i in range(Nscheme):
        s = '50' if loadFlag == 0 else '60_comoving'
        x = np.loadtxt('data/cqcgl2d_N' + s + '_hs_' + str(i) + '.dat')
        hs.append(x)
    
    fig = plt.figure(figsize=[12, 15])

    #plot time step
    T = 20.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    ax = fig.add_subplot('321')
    ax.text(0.5, 0.9, '(a)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'$h$', fontsize=20)
    ax.set_ylim([3e-5, 2e-2] if loadFlag == 0 else [2e-5, 4e-2])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        n = len(hs[i])
        x = np.arange(1, n+1) * T/n
        ax.plot(x, hs[i], lw=1.5, ls=lss[i], label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', nbins=5)
    ax.legend(loc='lower right', ncol=2, fontsize=12)

        
    # plot Nab
    ax = fig.add_subplot('322')
    ax.text(0.5, 0.9, '(b)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'$Nab$', fontsize=20)
    ax.set_ylim([0, 5000] if loadFlag == 0 else [0, 7000])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(2, 6):
        ax.plot(rtol, err[:, 5*i+1], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='right', fontsize=12)

    # plot Nn
    ax = fig.add_subplot('323')
    ax.text(0.1, 0.9, '(c)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'$Nn$', fontsize=20)
    ax.set_ylim([8e3, 1e7])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 5*i+2], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='upper right', fontsize=12)

    # plot NReject
    ax = fig.add_subplot('324')
    ax.text(0.9, 0.9, '(d)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'number of rejections', fontsize=15)
    ax.set_ylim([0, 180] if loadFlag == 0 else [0, 250])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 5*i+3], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='upper left', ncol=2, fontsize=12)

    # plot Wt
    ax = fig.add_subplot('325')
    ax.text(0.1, 0.9, '(e)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'$Wt$', fontsize=20)
    ax.set_ylim([1e3, 1e6])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 5*i+4], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='upper right', fontsize=12)

    # plot time of calculating coefficients / Wt
    ax = fig.add_subplot('326')
    ax.text(0.07, 0.9, '(f)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'time ratio', fontsize=15)
    ax.set_ylim([1e-6, 2e0])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 5*i+5]/err[:, 5*i+4], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=4)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='lower right', ncol=2, fontsize=12)

    ## 
    fig.tight_layout(pad=1)
    plt.show(block=False)
