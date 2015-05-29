##################################################
# Functions used for cqcgl1d system
##################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotConfigSpace(AA, ext, barTicks=[0, 3], colortype='jet', size=[4, 5]):
    Ar = AA[:, 0::2]
    Ai = AA[:, 1::2]
    Aamp = abs(Ar + 1j*Ai)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    im = ax.imshow(Aamp, cmap=plt.get_cmap(colortype), extent=ext,
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size='5%', pad=0.05)
    bar = plt.colorbar(im, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    plt.show(block='false')

