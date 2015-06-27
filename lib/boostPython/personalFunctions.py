##################################################
# Functions used for cqcgl1d system
##################################################
import h5py
from pylab import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

##################################################
#               Plot related                     #
##################################################


def plot1dfig(x, c='r', lw=1, ls='-', marker=None, labs=['x', 'y'],
              size=[8, 6], axisLabelSize=25, yscale=None):
    """
    plot 2d figures in genereal
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(x, c=c, lw=lw, ls=ls, marker=marker)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    if yscale is not None:
        ax.set_yscale(yscale)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plot2dfig(x, y, c='r', lw=1, ls='-', labs=['x', 'y'],
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


def scatter2dfig(x, y, s=20, marker='o', fc='r', ec='none', labs=['x', 'y'],
                 size=[8, 6], axisLabelSize=25):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=s, marker=marker,
               facecolor=fc, edgecolors=ec)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def multiScatter2dfig(x, y, s=[20], marker=['o'], fc=['r'],
                      ec='none', labs=['x', 'y'],
                      size=[8, 6], axisLabelSize=25):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    M = len(x)
    for i in range(M):
        ax.scatter(x[i], y[i], s=s[i], marker=marker[i],
                   facecolor=fc[i], edgecolors=ec)
    ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plot3dfig(x, y, z, c='r', lw=1, labs=['x', 'y', 'z'],
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


def plot3dfig2lines(x1, y1, z1, x2, y2, z2, c1='r', lw1=1,
                    c2='b', lw2=1, labs=['x', 'y', 'z'],
                    lab1='line 1', lab2='line 2',
                    size=[8, 6]):
    """
    plot 3d figures in genereal
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x1, y1, z1, c=c1, lw=lw1, label=lab1)
    ax.plot(x2, y2, z2, c=c2, lw=lw2, label=lab2)
    ax.set_xlabel(labs[0])
    ax.set_ylabel(labs[1])
    ax.set_zlabel(labs[2])
    ax.legend(loc='best')
    fig.tight_layout(pad=0)
    plt.show(block=False)


def plotConfigSpace(AA, ext, barTicks=[0, 3], colortype='jet',
                    percent='5%', size=[4, 5],
                    save=False, name='out.png'):
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
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)


def plotConfigSpaceFromFourier(cgl, aa, ext, barTicks=[0, 3],
                               colortype='jet',
                               percent='5%', size=[4, 5],
                               save=False, name='out.png'):
    """
    plot the configuration from Fourier mode
    """
    plotConfigSpace(cgl.Fourier2Config(aa), ext, barTicks,
                    colortype, percent, size, save, name)


def plotOneConfig(A, d=50, size=[8, 6], save=False, name='out.png'):
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
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)


def plotOneConfigFromFourier(cgl, a0, d=50, size=[8, 6],
                             save=False, name='out.png'):
    """
    plot the configuration at one point from Fourier mode
    """
    plotOneConfig(cgl.Fourier2Config(a0).squeeze(), d, size, save, name)


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

        
def sortByReal(eigvalue, eigvector=None):
    indx = np.argsort(eigvalue.real)
    indx = indx[::-1]
    if eigvector is None:
        return eigvalue[indx]
    else:
        return eigvalue[indx], eigvector[:, indx]


def eigReq(cgl, a0, wth0, wphi0):
    stabMat = cgl.stabReq(a0, wth0, wphi0).T
    eigvalue, eigvector = eig(stabMat)
    eigvalue, eigvector = sortByReal(eigvalue, eigvector)
    return eigvalue, eigvector


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
    n, m = ve.shape
    rve = np.zeros((n, m))
    i = 0
    while i < m:
        if np.sum(np.iscomplex(ve[:, i])) > 0:
            rve[:, i] = ve[:, i].real
            rve[:, i+1] = ve[:, i].imag
            i = i+2
        else:
            rve[:, i] = ve[:, i]
            i = i+1

    return rve


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


def cqcglSaveReq(fileName, groupName, a, wth, wphi, err):
    f = h5py.File(fileName, 'a')
    req = f.create_group(groupName)
    req.create_dataset("a", data=a)
    req.create_dataset("wth", data=wth)
    req.create_dataset('wphi', data=wphi)
    req.create_dataset('err', data=err)
    f.close()


def cqcglReadReq(fileName, groupName):
    f = h5py.File(fileName, 'r')
    req = '/' + groupName + '/'
    a = f[req+'a'].value
    wth = f[req+'wth'].value
    wphi = f[req+'wphi'].value
    err = f[req+'err'].value
    f.close()
    return a, wth, wphi, err


def cqcglRemoveReq(inputFile, outputFile, Num, groups):
    """
    remove some groups in relative equilibria file
    Num: the total number of groups in original file
         The group names are: 
         1, 2, 3, 4, ..., Num
    groups: the group names that need to be removed
    """
    ix = 1
    for i in range(1, Num+1):
        if i not in groups:
            a, wth, wphi, err = cqcglReadReq(inputFile, str(i))
            cqcglSaveReq(outputFile, str(ix), a, wth, wphi, err)
            ix += 1


def cqcglExtractReq(inputFile, outputFile, groups, startId=1):
    """
    Extract a subset of relative equibiria from input file
    The inverse of cacglRemoveReq
    groups: the indice of gropus that are going to be extracted from
            input file
    startId : the start group index in the output file
    """
    ix = startId
    n = np.size(groups)
    for i in range(n):
        a, wth, wphi, err = cqcglReadReq(inputFile, str(groups[i]))
        cqcglSaveReq(outputFile, str(ix), a, wth, wphi, err)
        ix += 1


def centerRand(N, frac):
    """
    generate a localized random vector.
    frac: the fraction of the nonzero center in the totol size
    """
    a = rand(N)
    N2 = np.int((1-frac)*0.5*N)
    a[:N2] = 0
    a[-N2:] = 0
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


def PoincareLinearInterp(x, getIndex=False):
    """
    x has dimension [N x 3]. The Poincare section is defined
    as x[i, 0] = 0. Linear interpolation is used to obtain
    the intersection points.

    (y - y1) / (0 - x1) = (y2 - y1) / (x2 - x1)
    (z - z1) / (0 - x1) = (z2 - z1) / (x2 - x1)
    """
    N, M = x.shape
    points = np.zeros((0, M-1))
    index = np.zeros((0,), dtype=np.int)
    for i in range(N-1):
        if x[i, 0] < 0 and x[i+1, 0] >= 0:
            ratio = -x[i, 0] / (x[i+1, 0] - x[i, 0])
            y = (x[i+1, 1] - x[i, 1]) * ratio + x[i, 1]
            z = (x[i+1, 2] - x[i, 2]) * ratio + x[i, 2]
            points = np.vstack((points, np.array([y, z])))
            index = np.append(index, i)
    if getIndex:
        return points, index
    else:
        return points
