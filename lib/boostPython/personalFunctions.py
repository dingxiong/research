##################################################
# Functions tool set 
##################################################
from time import time
import h5py
import matplotlib
matplotlib.use('Qt4Agg')
from pylab import *
import numpy as np
from numpy.random import rand
from scipy.spatial.distance import pdist, cdist, squareform
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import subprocess as sps

from numpy.linalg import norm
from matplotlib.ticker import AutoMinorLocator

from bisect import bisect_left

##################################################
#               Plot related                     #
##################################################


class Arrow3D(FancyArrowPatch):
    """
    The 3d arrow class
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def setvalue(self, x, y, z):
        self._verts3d = x, y, z


def plinit(size=[8, 6]):
    fig = plt.figure(figsize=size)
    return fig


def ax3dinit(fig, num=111, labs=[r'$x$', r'$y$', r'$z$'], axisLabelSize=25,
             tickSize=None,
             xlim=None, ylim=None, zlim=None, isBlack=False):
    ax = fig.add_subplot(num, projection='3d')

    if labs[0] is not None:
        ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    if labs[1] is not None:
        ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    if labs[2] is not None:
        ax.set_zlabel(labs[2], fontsize=axisLabelSize)

    if tickSize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tickSize)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    if isBlack:
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')

    return ax


def pl3d(size=[8, 6], labs=[r'$x$', r'$y$', r'$z$'], axisLabelSize=25,
         tickSize=None,
         xlim=None, ylim=None, zlim=None, isBlack=False):
    fig = plinit(size=size)
    ax = ax3dinit(fig, labs=labs, axisLabelSize=axisLabelSize,
                  tickSize=tickSize,
                  xlim=xlim, ylim=ylim, zlim=zlim, isBlack=isBlack)
    return fig, ax


def ax3d(fig, ax, doBlock=False, save=False, name='output.png',
         angle=None, title=None,
         loc='best', alpha=0.2, picForm='png'):
    fig.tight_layout(pad=0)

    if angle is not None:
        ax.view_init(angle[0], angle[1])
    
    if title is not None:
        ax.set_title(title)

    ax.legend(loc=loc, framealpha=alpha)
    if save:
        plt.savefig(name, format=picForm)
        plt.close()
    else:
        plt.show(block=doBlock)


def add3d(fig, ax, x, y, z, maxShow=5, c='r', s=70):
    i = 0
    pts = []
    while raw_input(i) == '':
        if i >= len(x):
            for p in pts:
                p.remove()
            break
        
        if len(pts) > maxShow:
            p = pts.pop(0)
            p.remove()
           
        if len(pts) > 0:
            pts[-1].set_sizes([2*s])
        if len(pts) > 1:
            pts[-2].set_sizes([s])

        tmp = ax.scatter(x[i], y[i], z[i], c=c, s=s,
                         edgecolors='none')
        pts.append(tmp)
        fig.canvas.draw()
        # plt.show(block=False)
        i += 1
    

def ax2dinit(fig, num=111, labs=[r'$x$', r'$y$'], axisLabelSize=25,
             xscale=None,
             yscale=None, xlim=None, ylim=None, tickSize=None,
             isBlack=False, ratio='auto'):
    ax = fig.add_subplot(num)
    
    if labs[0] is not None:
        ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    if labs[1] is not None:
        ax.set_ylabel(labs[1], fontsize=axisLabelSize)

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)

    if tickSize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tickSize)
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if isBlack:
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')

    ax.set_aspect(ratio)
    
    return ax


def pl2d(size=[8, 6], num=111, labs=[r'$x$', r'$y$'], axisLabelSize=25,
         xscale=None, yscale=None, xlim=None, ylim=None, tickSize=None,
         isBlack=False, ratio='auto'):
    fig = plt.figure(figsize=size)
    ax = ax2dinit(fig, num=111, labs=labs, axisLabelSize=axisLabelSize, xscale=xscale,
                  yscale=yscale, xlim=xlim, ylim=ylim, tickSize=tickSize,
                  isBlack=isBlack, ratio=ratio)
    return fig, ax


def ax2d(fig, ax, doBlock=False, save=False, name='output',
         title=None, loc='best', alpha=0.2):
    fig.tight_layout(pad=0)
    if title is not None:
        print "aaa"
        ax.set_title(title)

    ax.legend(loc=loc, framealpha=alpha)
    if save:
        plt.savefig(name + '.png', format='png')
        # plt.savefig(name + '.eps', format='eps')
        plt.close()
    else:
        plt.show(block=doBlock)


def makeMovie(data):
    fig, ax = pl3d()
    frame, = ax.plot([], [], [], c='red', ls='-', lw=1, alpha=0.5)
    pts, = ax.plot([], [], [], 'co', lw=3)

    def anim(i):
        j = min(i, data.shape[0])
        frame.set_data(data[:j, 0], data[:j, 1])
        frame.set_3d_properties(data[:j, 2])
        pts.set_data(data[j, 0], data[j, 1])
        pts.set_3d_properties(data[j, 2])
        
        ax.view_init(30, 0.5 * i)
        return frame, pts

    ani = animation.FuncAnimation(fig, anim, frames=data.shape[0],
                                  interval=0, blit=False, repeat=False)
    # ax3d(fig, ax)
    ax.legend()
    fig.tight_layout(pad=0)
    # ani.save('ani.mp4', dpi=200, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def plot1dfig(x, c='r', lw=1, ls='-', marker=None, labs=[r'$x$', r'$y$'],
              size=[8, 6], axisLabelSize=25, yscale=None):
    """
    plot 2d figures in genereal
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(x, c=c, lw=lw, ls=ls, marker=marker)
    if labs[0] is not None:
        ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    if labs[1] is not None:
        ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    if yscale is not None:
        ax.set_yscale(yscale)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plot2dfig(x, y, c='r', lw=1, ls='-', labs=[r'$x$', r'$y$'],
              size=[8, 6], axisLabelSize=25):
    """
    plot 2d figures in genereal
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(x, y, c=c, lw=lw, ls=ls)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def scatter2dfig(x, y, s=20, marker='o', fc='r', ec='none',
                 labs=[r'$x$', r'$y$'],
                 size=[8, 6], axisLabelSize=25, ratio='auto'):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=s, marker=marker,
               facecolor=fc, edgecolors=ec)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    ax.set_aspect(ratio)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plot3dfig(x, y, z, c='r', lw=1, labs=[r'$x$', r'$y$', r'$z$'],
              size=[8, 6], axisLabelSize=25):
    """
    plot 3d figures in genereal
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c=c, lw=lw)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    ax.set_zlabel(labs[2], fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plotMat(y, colortype='jet', percent='5%', colorBar=True,
            save=False, name='out.png'):
    """
    plot a matrix
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(y, cmap=plt.get_cmap(colortype))
    ax.grid('on')
    if colorBar:
        dr = make_axes_locatable(ax)
        cax = dr.append_axes('right', size=percent, pad=0.05)
        plt.colorbar(im, cax=cax)
    if save:
        plt.savefig(name)
        plt.close()
    else:
        plt.show(block=False)


def plotIm(y, ext, size=[4, 5], labs=[r'$x$', r'$y$'], colortype='jet', percent='5%',
           axisLabelSize=25, barTicks=None, tickSize=None, save=False, name='out.png'):
    """
    plot the imag color map of a matrix. It has more control compared with plotMat
    """
    fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize)
    im = ax.imshow(y, cmap=plt.get_cmap(colortype), extent=ext,
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size=percent, pad=0.05)

    if barTicks is not None:
        plt.colorbar(im, cax=cax, ticks=barTicks)
    else:
        plt.colorbar(im, cax=cax)

    if tickSize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tickSize)
    
    ax2d(fig, ax, save=save, name=name)


def plotContour(z, x=None, y=None, size=[8, 6], labs=[r'$x$', r'$y$'],
                axisLabelSize=25,
                save=False, name='output.png', title=None, loc='best'):
    fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize)
    if x is None or y is None:
        m, n = z.shape
        x = np.arange(n)
        y = np.arange(m)
    X, Y = np.meshgrid(x, y)
    CS = ax.contour(X, Y, z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax2d(fig, ax, save=save, name=name, title=title, loc=loc)

############################################################
#                        Common functions                  #
############################################################

def sortByReal(eigvalue, eigvector=None):
    indx = np.argsort(eigvalue.real)
    indx = indx[::-1]
    if eigvector is None:
        return eigvalue[indx]
    else:
        return eigvalue[indx], eigvector[:, indx]


def pAngleVQ(V, Q):
    """
    compute the angle between a vector V and set of vectors Q
    Q is an orthonormal matrix
    """
    assert V.ndim == 1
    V = V / LA.norm(V)
    c = LA.norm(np.dot(Q.T, V))
    if (c > 1):
        c = 1
    return np.arccos(c)


def pAngle(V, U):
    """
    compute the principle angle between a vector V and a subspace U
    """
    q = LA.qr(U)[0]
    return pAngleVQ(V, q)


def seqAng(seqDifv, U):
    """
    seqDifv is a sequence of row vector
    """
    q = LA.qr(U)[0]
    M, N = seqDifv.shape
    ang = np.zeros(M)
    for i in range(M):
        ang[i] = pAngle(seqDifv[i, :], q)
    return ang


def findMarginal(e, k):
    """
    find the position of marginal exponents
    """
    ix = argsort(abs(e))
    return ix[:k]


def removeMarginal(e, k):
    ix = findMarginal(e, k)
    ep = [i for i in e if i not in e[ix]]
    return ep


def centerRand(N, frac, isComplex=True):
    """
    generate a localized random vector.
    frac: the fraction of the nonzero center in the totol size
    """
    if isComplex:
        a = rand(N) + 1j*rand(N)
    else:
        a = rand(N)
        
    N2 = np.int((1-frac)*0.5*N)
    a[:N2] = 0
    a[-N2:] = 0
    return a


def centerRand2d(M, N, f1, f2, isComplex=True):
    """
    generatate a localized random MxN matrix
    """
    if isComplex:
        a = rand(M, N) + 1j*rand(M, N) + 0.5
    else:
        a = rand(M, N)
    
    M2 = np.int((1-f1)*0.5*M)
    N2 = np.int((1-f2)*0.5*N)
    a[:M2, :] = 0
    a[-M2:, :] = 0
    a[:, :N2] = 0
    a[:, -N2:] = 0
    return a


def centerOne(N, frac):
    """
    generate a localized random vector.
    frac: the fraction of the nonzero center in the totol size
    """
    a = ones(N)
    N2 = np.int((1-frac)*0.5*N)
    a[:N2] = 0
    a[-N2:] = 0
    return a


def Tcopy(x):
    """
    transpose and change the memory layout
    """
    m, n = x.shape
    y = np.zeros((n, m))
    for i in range(n):
        y[i] = x[:, i]

    return y


def realve(ve):
    """
    transform complex vectors to real format
    """
    n, m = ve.shape
    rve = np.zeros((n, m))
    i = 0
    while i < m:
        if np.sum(np.iscomplex(ve[:, i])) > 0:
            rve[:, i] = ve[:, i].real
            rve[:, i+1] = ve[:, i].imag
            i = i+2
        else:
            rve[:, i] = ve[:, i].real
            i = i+1

    return rve


def orthAxes2(e1, e2):
    """
    construct an orthogonal two vectors from 2 general vectors.
    """
    n = e1.shape[0]
    x = np.zeros((n, 2))
    x[:, 0] = e1
    x[:, 1] = e2
    q, r = LA.qr(x)

    return q[:, 0], q[:, 1]


def orthAxes(e1, e2, e3):
    n = e1.shape[0]
    x = np.zeros((n, 3))
    x[:, 0] = e1
    x[:, 1] = e2
    x[:, 2] = e3
    q, r = LA.qr(x)

    return q[:, 0], q[:, 1], q[:, 2]


def mag2vec(v1, v2):
    return np.sqrt(v1**2 + v2**2)


def normR(x):
    """
    the norm of each row of a matrix
    """
    return norm(x, axis=1)


def int2p(x, y, k=0):
    """
    interpolate two points
    """
    c1 = y[k] / (y[k] - x[k])
    c2 = -x[k] / (y[k] - x[k])
    return c1 * x + c2 * y, c1


def rotz(x, y, th):
    """
    rotate by z axis
    """
    c = np.cos(th)
    s = np.sin(th)
    return c * x - s * y, s * x + c * y


def difMap(x, farSize, size=[6, 6], percent='5%', colortype='jet'):
    m, n = x.shape
    y = np.zeros((m, farSize))
    for i in range(m-farSize):
        y[i] = norm(x[i]-x[i:i+farSize], axis=1)
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    # ax.set_xlabel('x', fontsize=axisLabelSize)
    # ax.set_ylabel('t', fontsize=axisLabelSize)
    im = ax.imshow(y, cmap=plt.get_cmap(colortype),
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size=percent, pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def difMap2(x1, x2, size=[6, 4], percent='5%', colortype='jet'):
    """
    try to locate the possible po
    """
    y = cdist(x1, x2)
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    # ax.set_xlabel('x', fontsize=axisLabelSize)
    # ax.set_ylabel('t', fontsize=axisLabelSize)
    im = ax.imshow(y, cmap=plt.get_cmap(colortype),
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size=percent, pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.tight_layout(pad=0)
    plt.show(block=False)

    
def getJumpPts(x):
    """
    for a sequence of integers x, try to get the
    locations when it jumps
    """
    n = x.shape[0]
    d = x[1:] - x[:-1]
    y = np.arange(n-1)
    y = y[d != 0]
    return y

def numStab(e, nmarg=2, tol=1e-8, flag=0):
    """
    get the number of unstable exponents. Assume e is sorted by its
    real part. 

    Parameters
    ======
    nmarg : number of marginal eigenvalues of expolents
    flag :  flag = 0 => exponents; flag = 1 => eigenvalues
    tol :  tolerance to judge marginal exponents

    Return
    ======
    m : number of unstable exponents or starting index for marginal
        exponent
    ep : exponents without marginal ones
    """
    accu = True
    if flag == 0:
        x = e.real
    else:
        x = np.log(np.abs(e))

    n = len(x)
    m = n-1
    for i in range(n):
        if abs(x[i]) < tol:
            m = i
            break

    if m >= n-1:
        accu = False
    else:
        for i in range(m+1, min(m+nmarg, n)):
            if abs(x[i]) > tol:
                accu = False
                break

    ep = np.delete(e, range(m, m+nmarg))
    return m, ep, accu

