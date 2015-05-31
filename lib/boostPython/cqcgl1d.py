##################################################
# Functions used for cqcgl1d system
##################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotConfigSpace(AA, ext, barTicks=[0, 3], colortype='jet',
                    percent='5%', size=[4, 5]):
    """
    plot the color map of the states
    """
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
    cax = dr.append_axes('right', size=percent, pad=0.05)
    bar = plt.colorbar(im, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plotOneConfig(A, d=50, size=[8, 6]):
    """
    plot the configuration at one point
    """
    Aamp = abs(A[0::2]+1j*A[1::2])
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, d, Aamp.size), Aamp)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$|A|$')
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plotOneConfigFromFourier(cgl, a0, d=50, size=[8, 6]):
    """
    plot the configuration at one point from Fourier mode
    """
    plotOneConfig(cgl.Fourier2Config(a0).squeeze(), d, size)


def plotOneFourier(a, color='r', size=[8, 6]):
    """
    plot Fourier modes at one point
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(a, c=color)
    ax.set_xlabel('k')
    ax.set_ylabel('Fourier modes')
    fig.tight_layout(pad=0)
    plt.show(block=False)


def sortByReal(eigvalue, eigvector=None):
    indx = np.argsort(eigvalue.real)
    indx = indx[::-1]
    if eigvector is None:
        return eigvalue[indx]
    else:
        return eigvalue[indx], eigvector[:, indx]
