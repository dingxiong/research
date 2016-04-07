from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *


def plotcgl(Aamp, ext, barTicks=[0, 3], colortype='jet',
            percent='5%', size=[4, 5],
            axisLabelSize=20,
            save=False, name='out.png'):
    """
    plot the color map of the states
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x', fontsize=axisLabelSize)
    ax.set_ylabel('t', fontsize=axisLabelSize)
    im = ax.imshow(Aamp, cmap=plt.get_cmap(colortype), extent=ext,
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size=percent, pad=0.05)
    plt.colorbar(im, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)


case = 1

if case == 1:
    N = 1024
    d = 30
    h = 0.0002
    di = 0.06
    
    Y = loadtxt('Y.dat')
    plotcgl(Y, [0, 30, 0, 3])
