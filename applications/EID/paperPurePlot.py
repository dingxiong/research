from cglHelp import *
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})  # I love Times
rc('text', usetex=True)


case = 45

labels = ["IF4(3)", "IF5(4)",
          "ERK4(3)2(2)", "ERK4(3)3(3)", "ERK4(3)4(3)", "ERK5(4)5(4)",
          "SS4(3)"]
mks = ['o', 's', '+', '^', 'x', 'v', 'p']
Nscheme = len(labels)
lss = ['--', '-.', ':', '-', '-', '-', '-']

if case == 45:
    """
    plot fence instead of heat
    """
    N, d = 1024, 50
    fig = plt.figure(figsize=[15, 4])
    
    # 1st
    ax = ax3dinit(fig, num=131, labs=[r'$x$', r'$t$', None], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(a)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    # cgl = pyCQCGL1d(N, d, -0.1, 0.08, 0.5, 0.782, 1, -0.1, -0.08, -1)
    skipRate = 1
    pulsate = np.load('data/pulsatingSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp = Aamp[::skipRate, :]
    Aamp2 = Aamp2[::skipRate, :]
    rows, cols = Aamp.shape
    X = np.linspace(0, d, cols)
    Y = np.linspace(0, T, rows)
    for i in range(Aamp.shape[0]):
        ax.plot(X, np.ones(cols) * Y[i], Aamp[i], c='k', alpha=1)
    
    ax.set_yticks([0, 30, 60])
    ax.set_zticks([0, 3])
    ax.view_init(75, -50)
    
    # 2nd
    ax = ax3dinit(fig, num=132, labs=[r'$x$', r'$t$', None], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(b)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    # Bi, Gi = 3.5, -5.0
    # cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    skipRate = 1
    pulsate = np.load('data/extremeSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp = Aamp[::skipRate, :]
    Aamp2 = Aamp2[::skipRate, :]
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
    ax = ax3dinit(fig, num=133, labs=[r'$x$', r'$t$', r'$|A|$'], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(c)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    # Bi, Gi = 0.8, -0.6
    # cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    skipRate = 5
    pulsate = np.load('data/explodingSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp = Aamp[::skipRate, :]
    Aamp2 = Aamp2[::skipRate, :]
    rows, cols = Aamp.shape
    X = np.linspace(0, d, cols)
    Y = np.linspace(0, T, rows)
    for i in range(Aamp.shape[0]):
        ax.plot(X, np.ones(cols) * Y[i], Aamp[i], c='k', alpha=1)
    
    ax.set_ylim([0, 20])
    ax.set_yticks([0, 10, 20])
    ax.set_zticks([0, 3])
    ax.view_init(75, -80)
    
    ############
    ax3d(fig, ax)


if case == 46:
    """
    plot fence instead of heat same as case 45
    but for time adaptive in constant frame
    """
    N, d = 1024, 50
    fig = plt.figure(figsize=[15, 4])
    
    # 1st
    ax = ax3dinit(fig, num=131, labs=[r'$x$', r'$t$', None], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(a)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    skipRate = 3
    pulsate = np.load('data/pulsatingSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp2 = Aamp2[::skipRate, :]
    Ts = Ts[::skipRate]
    rows, cols = Aamp2.shape
    X = np.linspace(0, d, cols)
    for i in range(rows):
        ax.plot(X, np.ones(cols) * Ts[i], Aamp2[i], c='k', alpha=1)
    
    ax.set_yticks([0, 30, 60])
    ax.set_zticks([0, 3])
    ax.view_init(75, -50)
    
    # 2nd
    ax = ax3dinit(fig, num=132, labs=[r'$x$', r'$t$', None], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(b)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    # Bi, Gi = 3.5, -5.0
    # cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    skipRate = 2
    pulsate = np.load('data/extremeSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp2 = Aamp2[::skipRate, :]
    Ts = Ts[::skipRate]
    rows, cols = Aamp2.shape
    X = np.linspace(0, d, cols)
    for i in range(rows):
        ax.plot(X, np.ones(cols) * Ts[i], Aamp2[i], c='k', alpha=1)
    
    # ax.set_ylim([0, 13])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([0, 1])
    ax.view_init(60, -40)

    # 3rd
    ax = ax3dinit(fig, num=133, labs=[r'$x$', r'$t$', r'$|A|$'], axisLabelSize=20, tickSize=15)
    ax.text2D(0.3, 0.9, '(c)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=22)
    skipRate = 5
    pulsate = np.load('data/explodingSolitonAmp.npz')
    Aamp, Aamp2, Ts, T = pulsate['states'], pulsate['statesAdapt'], pulsate['Ts'], pulsate['T']
    Aamp2 = Aamp2[::skipRate, :]
    Ts = Ts[::skipRate]
    rows, cols = Aamp2.shape
    X = np.linspace(0, d, cols)
    for i in range(rows):
        ax.plot(X, np.ones(cols) * Ts[i], Aamp2[i], c='k', alpha=1)
    
    # ax.set_ylim([0, 20])
    ax.set_yticks([0, 10, 20])
    ax.set_zticks([0, 3])
    ax.view_init(75, -80)
    
    ############
    ax3d(fig, ax)
