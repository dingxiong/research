from cglHelp import *
from py_CQCGL1d import *

case = 44

labels = ["IF4(3)", "IF5(4)",
          "ERK4(3)2(2)", "ERK4(3)3(3)", "ERK4(3)4(3)", "ERK5(4)5(4)",
          "SS4(3)"]
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
    plot 1d cqcgl of heat map / configuration figure for the const time step,
    time step adaption and comoving frame methods
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    cp = CQCGLplot(cgl)

    ###################
    # the case with constant time step
    ################### 
    aa = np.loadtxt('data/aa.dat')

    # plot the heat map
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    fig, ax = pl2d(size=[4, 5], labs=[r'$x$', r'$t$'], axisLabelSize=25, tickSize=18)
    im = ax.imshow(Aamp, cmap=plt.get_cmap('jet'), extent=[0, d, 0, 20],
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    ax2d(fig, ax)

    # plot a single state
    numOfState = aa.shape[0]
    fig, ax = pl2d(size=[6, 5], labs=[r'$x$', r'$|A|$'], 
                   ylim=[0, 3.5],
                   axisLabelSize=25, tickSize=18)
    A = cgl.Fourier2Config(aa[int(10./20 * numOfState)])
    Aamp = np.abs(A)
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=3, ls='--', c='r')
    A = cgl.Fourier2Config(aa[int(7.0/20 * numOfState)])
    Aamp = np.abs(A)
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2, ls='-', c='b')
    ax.locator_params(axis='y', nbins=4)
    ax2d(fig, ax)

    ###################
    # the case with static frame time step adaption
    ################### 
    aa = np.loadtxt('data/aaAdapt_Cox_Matthews.dat')
    Ts = np.loadtxt('data/TsAdapt_Cox_Matthews.dat')
    
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    fig, ax = pl2d(size=[4, 5], labs=[r'$x$', r'$t$'], axisLabelSize=25, tickSize=18)
    im = ax.imshow(Aamp, cmap=plt.get_cmap('jet'), extent=[0, d, 0, 20],
                   aspect='auto', origin='lower')
    yls = [0, 5, 10, 15, 20]
    ids = [bisect_left(Ts, yl) for yl in yls]
    yts = [x / float(Ts.size) * 20 for x in ids]
    ax.set_yticks(yts)
    ax.set_yticklabels(yls)
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    ax2d(fig, ax)

    aa = np.loadtxt('data/aaAdapt_SSPP43.dat')
    Ts = np.loadtxt('data/TsAdapt_SSPP43.dat')
    
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    fig, ax = pl2d(size=[4, 5], labs=[r'$x$', r'$t$'], axisLabelSize=25, tickSize=18)
    im = ax.imshow(Aamp, cmap=plt.get_cmap('jet'), extent=[0, d, 0, 20],
                   aspect='auto', origin='lower')
    yls = [0, 5, 10, 15, 20]
    ids = [bisect_left(Ts, yl) for yl in yls]
    yts = [x / float(Ts.size) * 20 for x in ids]
    ax.set_yticks(yts)
    ax.set_yticklabels(yls)
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    ax2d(fig, ax)

    ###################
    # the case with comoving frame time step adaption
    ################### 
    aa = np.loadtxt('data/aaCom.dat')
    Ts = np.loadtxt('data/TsCom.dat')
    
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    fig, ax = pl2d(size=[4, 5], labs=[r'$x$', r'$t$'], axisLabelSize=25, tickSize=18)
    im = ax.imshow(Aamp, cmap=plt.get_cmap('jet'), extent=[0, d, 0, 20],
                   aspect='auto', origin='lower')
    yls = [0, 5, 10, 15, 20]
    ids = [bisect_left(Ts, yl) for yl in yls]
    yts = [x / float(Ts.size) * 20 for x in ids]
    ax.set_yticks(yts)
    ax.set_yticklabels(yls)
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    ax2d(fig, ax)
    
    ####################
    # plot the high frequency phase rotation
    ####################
    aa = np.loadtxt('data/a0NoAdapt.dat')
    fig = plt.figure(figsize=[6, 5])
    ax = fig.add_subplot('111')
    ax.set_xlabel(r'$t$', fontsize=25)
    ax.set_ylabel(r'$Re(a_0)$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.plot(np.linspace(0, 20, aa.size), aa, lw=1, c='r')
    plt.tight_layout(pad=0)
    plt.show(block=False)

    ####################
    # plot the phase rotation and the profile of req
    ####################
    aa = np.loadtxt('data/a0ComNoAdapt.dat')
    req = CQCGLreq(cgl)
    a0, wth0, wphi0, err0 = req.readReqBiGi('../../data/cgl/reqBiGi.h5',
                                            Bi, Gi, 1)
    AA = cgl.Fourier2Config(a0)
    Aamp = np.abs(AA)    
    
    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot('121')
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xlabel(r'$x$', fontsize=25)
    ax.set_ylabel(r'$|A|$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2)

    ax = fig.add_subplot('122')
    ax.text(0.1, 0.9, '(b)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xlabel(r'$t$', fontsize=25)
    ax.set_ylabel(r'$Re(\tilde{a}_0)$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.plot(np.linspace(0, 20, aa.size), aa, lw=1, c='r')

    plt.tight_layout(pad=0)
    plt.show(block=False)

if case == 44:
    """
    transform  data in case 40 
    """
    N, d = 1024, 50
    
    # 1st
    cgl = pyCQCGL1d(N, d, -0.1, 0.08, 0.5, 0.782, 1, -0.1, -0.08, -1)
    pulsate = np.load('data/pulsatingSoliton.npz')
    aa, aa2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    AA2 = cgl.Fourier2Config(aa2)
    Aamp2 = np.abs(AA2)
    np.savez_compressed('data/pulsatingSolitonAmp.npz', states=Aamp, statesAdapt=Aamp2, Ts=Ts, T=T)

    # 2nd
    Bi, Gi = 3.5, -5.0
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    pulsate = np.load('data/extremeSoliton.npz')
    aa, aa2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    AA2 = cgl.Fourier2Config(aa2)
    Aamp2 = np.abs(AA2)
    np.savez_compressed('data/extremeSolitonAmp.npz', states=Aamp, statesAdapt=Aamp2, Ts=Ts, T=T)

    # 3rd
    Bi, Gi = 0.8, -0.6
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    aa = np.loadtxt('data/aa.dat')
    aa2 = np.loadtxt('data/aaAdapt_Cox_Matthews.dat')
    Ts = np.loadtxt('data/TsAdapt_Cox_Matthews.dat')
    T = 20
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    AA2 = cgl.Fourier2Config(aa2)
    Aamp2 = np.abs(AA2)
    np.savez_compressed('data/explodingSolitonAmp.npz', states=Aamp, statesAdapt=Aamp2, Ts=Ts, T=T)

if case == 45:
    """
    plot fence instead of heat
    """
    N, d = 1024, 50
    fig = plt.figure(figsize=[15, 4])
    
    # 1st
    ax = ax3dinit(fig, num=131, labs=[r'$x$', r'$t$', r'$|A|$'], axisLabelSize=20, tickSize=12)
    ax.text2D(0.3, 0.9, '(a)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=18)
    cgl = pyCQCGL1d(N, d, -0.1, 0.08, 0.5, 0.782, 1, -0.1, -0.08, -1)
    skipRate = 1
    pulsate = np.load('data/pulsatingSoliton.npz')
    aa, aa2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    Aamp = Aamp[::skipRate, :]
    rows, cols = Aamp.shape
    X = np.linspace(0, d, cols)
    Y = np.linspace(0, T, rows)
    for i in range(Aamp.shape[0]):
        ax.plot(X, np.ones(cols) * Y[i], Aamp[i], c='k', alpha=1)
    
    ax.set_yticks([0, 30, 60])
    ax.set_zticks([0, 3])
    ax.view_init(75, -50)
    
    # 2nd
    ax = ax3dinit(fig, num=132, labs=[r'$x$', r'$t$', r'$|A|$'], axisLabelSize=20, tickSize=12)
    ax.text2D(0.3, 0.9, '(b)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=18)
    Bi, Gi = 3.5, -5.0
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    skipRate = 1
    pulsate = np.load('data/extremeSoliton.npz')
    aa, aa2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    Aamp = Aamp[::skipRate, :]
    rows, cols = Aamp.shape
    X = np.linspace(0, d, cols)
    Y = np.linspace(0, T, rows)
    for i in range(Aamp.shape[0]):
        ax.plot(X, np.ones(cols) * Y[i], Aamp[i], c='k', alpha=1)
    
    ax.set_ylim([0, 13])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([0, 1])
    ax.view_init(60, -40)

    # 3rd
    ax = ax3dinit(fig, num=133, labs=[r'$x$', r'$t$', r'$|A|$'], axisLabelSize=20, tickSize=12)
    ax.text2D(0.3, 0.9, '(c)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=18)
    Bi, Gi = 0.8, -0.6
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    skipRate = 5
    aa = np.loadtxt('data/aa.dat')
    T = 20
    AA = cgl.Fourier2Config(aa)
    Aamp = np.abs(AA)
    Aamp = Aamp[::skipRate, :]
    rows, cols = Aamp.shape
    X = np.linspace(0, d, cols)
    Y = np.linspace(0, T, rows)
    for i in range(Aamp.shape[0]):
        ax.plot(X, np.ones(cols) * Y[i], Aamp[i], c='k', alpha=1)
    
    ax.set_ylim([0, 13])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([0, 3])
    ax.view_init(75, -80)
    
    ############
    ax3d(fig, ax)
    

if case == 50:
    """
    plot the relative errors of 1d cgcgl by constant schemes
    """
    err = np.loadtxt('data/cqcgl1d_N20_err.dat')
    h = err[:, 0]
    fig, ax = pl2d(size=[6, 5], labs=[r'$h$', None],
                   axisLabelSize=20, tickSize=15,
                   xlim=[3e-5, 3e-2],
                   ylim=[1e-11, 1e1],
                   xscale='log', yscale='log')
    for i in range(Nscheme):
        ax.plot(h, err[:, i+1], lw=1.5, marker=mks[i], mfc='none',
                label=labels[i])
    ax.plot([1e-4, 1e-2], [1e-8, 1e0], lw=2, c='k', ls='--')
    ax.plot([2e-3, 2e-2], [1e-9, 1e-4], lw=2, c='k', ls='--')
    # ax.grid(True, which='both')
    ax.locator_params(axis='y', numticks=5)
    ax2d(fig, ax, loc='upper left')

if case == 60:
    """
    plot the estimated local error of 1d cgcgl by constant schemes
    """
    err = np.loadtxt('data/cqcgl1d_N30_lte.dat')
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    n = err.shape[0]
    T = 20.0
    x = np.arange(1, n+1) * T/n
    
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'estimated local error'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[1e-17, 1e-7],
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
                   ylim=[1e-30, 1e-8],
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
        x = np.loadtxt('data/cqcgl1d_N50_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 20.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[2e-5, 2e-3],
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
        x = np.loadtxt('data/cqcgl1d_N60_comoving_hs_' + str(i) + '.dat')
        hs.append(x)
    
    T = 20.0
    lss = ['--', '-.', ':', '-', '-', '-', '-']
    fig, ax = pl2d(size=[6, 5], labs=[r'$t$', r'$h$'],
                   axisLabelSize=20, tickSize=15,
                   # xlim=[1e-8, 5e-3],
                   ylim=[8e-6, 2e-2],
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
    loadFlag = 0

    err = np.loadtxt('data/cqcgl1d_N70_stat' + ('0' if loadFlag == 0 else '1') + '.dat')
    rtol = err[:, 0]
    
    fig = plt.figure(figsize=[12, 15])
    
    # plot relative error
    ax = fig.add_subplot('321')
    ax.text(0.5, 0.9, '(a)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel('relative error', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 6*i+1], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='bottem right', fontsize=12)
    
    # plot Nab
    ax = fig.add_subplot('322')
    ax.text(0.5, 0.9, '(b)', horizontalalignment='center',
         transform=ax.transAxes, fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_xlabel(r'$rtol$', fontsize=20)
    ax.set_ylabel(r'$Nab$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(2, 6):
        ax.plot(rtol, err[:, 6*i+2], lw=1.5, marker=mks[i], mfc='none',
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
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 6*i+3], lw=1.5, marker=mks[i], mfc='none',
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
    ax.set_ylim([0, 80] if loadFlag == 0 else [0, 250])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 6*i+4], lw=1.5, marker=mks[i], mfc='none',
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
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i in range(Nscheme):
        ax.plot(rtol, err[:, 6*i+5], lw=1.5, marker=mks[i], mfc='none',
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
        ax.plot(rtol, err[:, 6*i+6]/err[:, 6*i+5], lw=1.5, marker=mks[i], mfc='none',
                ms=8, label=labels[i])
    ax.locator_params(axis='y', numticks=5)
    ax.locator_params(axis='x', numticks=5)
    ax.legend(loc='lower right', fontsize=12)

    ## 
    fig.tight_layout(pad=1)
    plt.show(block=False)

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

if case == 120:
    """
    Compare the accuracy of static and comoving frames
    """
    err = np.loadtxt('data/cqcgl1d_N70_stat0.dat')
    err2 = np.loadtxt('data/cqcgl1d_N70_stat1.dat')
    rtol = err[:, 0]

    for i in range(2, 6):
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

