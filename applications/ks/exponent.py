import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from personalFunctions import *


case = 1

if case == 1:
    """
    plot Floquet exponents for N = 64 with a insert plot
    """
    FE = KSreadFE('../../data/ks22h001t120x64E.h5', 'ppo', 1)[0]

    N = 32
    d = 22
    x = np.arange(1, N, 0.01)
    qk = 2 * np.pi / d * x
    L = qk**2 - qk**4

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.plot(x, L, 'g--', lw=1.8)
    ax.scatter(np.arange(1, N), FE[::2], c='r',
               marker='o', s=22, edgecolors='none')
    ax.scatter(np.arange(1, N), FE[1::2], c='r',
               marker='s', s=30, facecolors='none',
               edgecolors='k')
    yfmt = ScalarFormatter()
    yfmt.set_powerlimits((0, 1))
    ax.yaxis.set_major_formatter(yfmt)

    ax.set_yticks((-7e3, -3e3, 0, 1e3))
    ax.set_xlim((0, 35))
    ax.set_ylim((-7e3, 1e3))
    ax.grid('on')

    axin = inset_axes(ax, width="45%", height="50%", loc=3)
    axin.scatter(np.arange(1, N), FE[::2], c='r',
                 marker='o', s=22, edgecolors='none')
    axin.scatter(np.arange(1, N), FE[1::2], c='r',
                 marker='s', s=30, facecolors='none',
                 edgecolors='k')
    axin.set_xlim(0.5, 4.5)
    axin.set_ylim(-0.4, 0.1)
    axin.yaxis.set_ticks_position('right')
    axin.xaxis.set_ticks_position('top')
    axin.set_xticks((1, 2, 3, 4))
    axin.set_yticks((-0.4, -0.2, 0))
    axin.grid('on')

    mark_inset(ax, axin, loc1=1, loc2=2, fc="none")

    fig.tight_layout(pad=0)
    plt.show()
    # plt.savefig('pprpfigure.eps',format='eps', dpi=30)
