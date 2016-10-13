##################################################
# Functions used for cqcgl1d system
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
             xlim=None, ylim=None, zlim=None, isBlack=False):
    ax = fig.add_subplot(num, projection='3d')

    if labs[0] is not None:
        ax.set_xlabel(labs[0], fontsize=axisLabelSize)
    if labs[1] is not None:
        ax.set_ylabel(labs[1], fontsize=axisLabelSize)
    if labs[2] is not None:
        ax.set_zlabel(labs[2], fontsize=axisLabelSize)

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
         xlim=None, ylim=None, zlim=None, isBlack=False):
    fig = plinit(size=size)
    ax = ax3dinit(fig, labs=labs, axisLabelSize=axisLabelSize,
                  xlim=xlim, ylim=ylim, zlim=zlim, isBlack=isBlack)
    return fig, ax


def ax3d(fig, ax, doBlock=False, save=False, name='output.png',
         angle=None, title=None,
         loc='best', alpha=0.2):
    fig.tight_layout(pad=0)

    if angle is not None:
        ax.view_init(angle[0], angle[1])
    
    if title is not None:
        ax.set_title(title)

    ax.legend(loc=loc, framealpha=alpha)
    if save:
        plt.savefig(name)
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


def pl2d(size=[8, 6], labs=[r'$x$', r'$y$'], axisLabelSize=25,
         xscale=None, yscale=None, xlim=None, ylim=None, tickSize=None,
         isBlack=False, ratio='auto'):
    fig = plt.figure(figsize=size)
    ax = ax2dinit(fig, labs=labs, axisLabelSize=axisLabelSize, xscale=xscale,
                  yscale=yscale, xlim=xlim, ylim=ylim, tickSize=tickSize,
                  isBlack=isBlack, ratio=ratio)
    return fig, ax


def ax2d(fig, ax, doBlock=False, save=False, name='output.png',
         title=None, loc='best', alpha=0.2):
    fig.tight_layout(pad=0)
    if title is not None:
        print "aaa"
        ax.set_title(title)

    ax.legend(loc=loc, framealpha=alpha)
    if save:
        plt.savefig(name)
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


##################################################
#            1d CQCGL related                    #
##################################################
class CQCGLplot():
    def __init__(self, cgl=None):
        self.cgl = cgl

    def oneConfig(self, A, d=30, isFourier=True, 
                  size=[6, 5], labs=[r'$x$', r'$|A|$'],
                  axisLabelSize=20, tickSize=None,
                  save=False, name='out.png'):
        """
        plot the configuration at one point 
        or plot one eigen vector.
        
        Parameter
        ======
        isFourier : is the input the Fourier modes of A. If it is, then we need back FFT.
        """
        fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize, tickSize=tickSize)
        if isFourier:
            A = self.cgl.Fourier2Config(A)
        Aamp = np.abs(A)
        ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=1.5)
        ax2d(fig, ax, save=save, name=name)
    
    def config(self, AA, ext, isFourier=True, tt=None, yls=None,
               barTicks=[2, 7], colortype='jet',
               percent='5%', size=[4, 5], labs=[r'$x$', r'$t$'],
               axisLabelSize=20, tickSize=None,
               save=False, name='out.png'):
        """
        plot the color map of the states
        """
        if isFourier:
            AA = self.cgl.Fourier2Config(AA)
        Aamp = np.abs(AA)
        fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize, tickSize=tickSize)
        im = ax.imshow(Aamp, cmap=plt.get_cmap(colortype), extent=ext,
                       aspect='auto', origin='lower')
        if tt is not None:
            n = len(tt)
            ids = findTimeSpots(tt, yls)
            yts = ids / float(n) * ext[3]
            ax.set_yticks(yts)
            ax.set_yticklabels(yls)

        ax.grid('on')
        dr = make_axes_locatable(ax)
        cax = dr.append_axes('right', size=percent, pad=0.05)
        plt.colorbar(im, cax=cax, ticks=barTicks)

        ax2d(fig, ax, save=save, name=name)


class CQCGLreq():
    def __init__(self, cgl=None):
        self.cgl = cgl
    
    def toStr(self, Bi, Gi, index):
        if abs(Bi) < 1e-6:
            Bi = 0
        if abs(Gi) < 1e-6:
            Gi = 0
        return (format(Bi, '013.6f') + '/' + format(Gi, '013.6f') +
                '/' + str(index))

    def checkExist(self, fileName, groupName):
        f = h5py.File(fileName, 'r')
        x = groupName in f
        f.close()
        return x
        
    def readReq(self, fileName, groupName, flag=0):
        f = h5py.File(fileName, 'r')
        req = '/' + groupName + '/'
        a = f[req+'a'].value
        wth = f[req+'wth'].value
        wphi = f[req+'wphi'].value
        err = f[req+'err'].value
        if flag == 1:
            e = f[req+'er'].value + 1j*f[req+'ei'].value
        if flag == 2:
            e = f[req+'er'].value + 1j*f[req+'ei'].value
            v = f[req+'vr'].value + 1j*f[req+'vi'].value
        f.close()
        
        if flag == 0:
            return a, wth, wphi, err
        if flag == 1:
            return a, wth, wphi, err, e
        if flag == 2:
            return a, wth, wphi, err, e, v

    def readReqdi(self, fileName, di, index, flag=0):
        groupName = format(di, '.6f') + '/' + str(index)
        return self.readReq(fileName, groupName, flag)

    def readReqBiGi(self, fileName, Bi, Gi, index, flag=0):
        return self.readReq(fileName, self.toStr(Bi, Gi, index), flag)

    def eigReq(self, a0, wth0, wphi0):
        stabMat = self.cgl.stabReq(a0, wth0, wphi0).T
        e, v = LA.eig(stabMat)
        e, v = sortByReal(e, v)
        v = v.T.copy()

        return e, v

    def getAxes(self, fileName, Bi, Gi, index, flag):
        a, wth, wphi, err, e, v = self.readReqBiGi(fileName, Bi, Gi, index, flag=2)
        aH = self.cgl.orbit2slice(a, flag)[0]
        vrH = self.cgl.ve2slice(v.real.copy(), a, flag)
        viH = self.cgl.ve2slice(v.imag.copy(), a, flag)
        
        return e, aH, vrH + 1j*viH
    

class CQCGLrpo():

    def __init__(self, cgl=None):
        self.cgl = cgl
        
    def toStr(self, Bi, Gi, index):
        if abs(Bi) < 1e-6:
            Bi = 0
        if abs(Gi) < 1e-6:
            Gi = 0
        return (format(Bi, '013.6f') + '/' + format(Gi, '013.6f') +
                '/' + str(index))

    def checkExist(self, fileName, groupName):
        f = h5py.File(fileName, 'r')
        x = groupName in f
        f.close()
        return x

    def readRpo(self, fileName, groupName, flag=0):
        f = h5py.File(fileName, 'r')
        req = '/' + groupName + '/'
        x = f[req+'x'].value
        T = f[req+'T'].value
        nstp = f[req+'nstp'].value
        th = f[req+'th'].value
        phi = f[req+'phi'].value
        err = f[req+'err'].value
        if flag == 1:
            e = f[req+'er'].value + 1j*f[req+'ei'].value
        if flag == 2:
            e = f[req+'er'].value + 1j*f[req+'ei'].value
            v = f[req+'v'].value
        f.close()

        if flag == 0:
            return x, T, nstp, th, phi, err
        if flag == 1:
            return x, T, nstp, th, phi, err, e
        if flag == 2:
            return x, T, nstp, th, phi, err, e, v

    def readRpodi(self, fileName, di, index, flag=0):
        groupName = format(di, '.6f') + '/' + str(index)
        return self.readRpo(fileName, groupName, flag)

    def readRpoBiGi(self, fileName, Bi, Gi, index, flag=0):
        return self.readRpo(fileName, self.toStr(Bi, Gi, index), flag)
    
    def readRPOAll(self, fileName, index, hasEV):
        f = h5py.File(fileName, 'r')
        gs = f.keys()
        f.close()
        xx = []                     # all rpo
        dis = []                    # all di
        for i in gs:
            di = float(i)
            dis.append(di)
            if hasEV:
                x = cqcglReadRPOEVdi(fileName, di, index)
            else:
                x = cqcglReadRPOdi(fileName, di, index)
            xx.append(x)
        return dis, xx

    def saveRpo(self, fileName, groupName, x, T, nstp, th, phi, err):
        f = h5py.File(fileName, 'a')
        rpo = f.create_group(groupName)
        rpo.create_dataset("x", data=x)
        rpo.create_dataset("T", data=T)
        rpo.create_dataset("nstp", data=nstp)
        rpo.create_dataset("th", data=th)
        rpo.create_dataset('phi', data=phi)
        rpo.create_dataset('err', data=err)
        f.close()

    def saveRpodi(self, fileName, di, index, x, T, nstp, th, phi, err):
        groupName = format(di, '.6f') + '/' + str(index)
        return self.saveRPO(fileName, groupName, x, T, nstp, th, phi, err)

    def saveRpoBiGi(self, fileName, Bi, Gi, index, x, T, nstp, th, phi, err):
        groupName = (format(Bi, '013.6f') + '/' + format(Gi, '013.6f') +
                     '/' + str(index))
        return self.saveRpo(fileName, groupName, x, T, nstp, th, phi, err)

    
        
#===================================================



        
def findTimeSpots(tt, spots):
    n = len(spots)
    ids = np.zeros(n, dtype=np.int)
    for i in range(n):
        t = spots[i]
        L = 0
        R = size(tt) - 1
        while R-L > 1:
            d = (R-L)/2
            if tt[L+d] <= t:
                L += d
            elif tt[R-d] > t:
                R -= d
            else:
                print "error"
        ids[i] = L
    return ids



def plotConfigSurface(AA, ext, barTicks=[2, 4], colortype='jet',
                      percent='5%', size=[7, 5], axisLabelSize=25,
                      save=False, name='out.png'):
    """
    plot the color map of the states
    """
    Aamp = np.abs(AA)
    
    X = np.linspace(ext[0], ext[1], Aamp.shape[1])
    Y = np.linspace(ext[2], ext[3], Aamp.shape[0])
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Aamp, rstride=10, cstride=10,
                           cmap=plt.get_cmap(colortype),
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, fraction=0.05, shrink=0.6, aspect=20, ticks=barTicks)
    
    ax.set_xlabel(r'$x$', fontsize=axisLabelSize)
    ax.set_ylabel(r'$t$', fontsize=axisLabelSize)
    ax.set_zlabel(r'$|A|$', fontsize=axisLabelSize)

    # dr = make_axes_locatable(ax)
    # cax = dr.append_axes('right', size=percent, pad=0.05)
    # plt.colorbar(surf, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
        plt.close()
    else:
        plt.show(block=False)


def plotConfigSurfaceFourier(cgl, aa, ext, barTicks=[2, 7],
                             colortype='jet',
                             percent='5%', size=[7, 5],
                             axisLabelSize=25,
                             save=False, name='out.png'):
    plotConfigSurface(cgl.Fourier2Config(aa), ext, barTicks,
                      colortype, percent, size,
                      axisLabelSize,
                      save, name)


def plotConfigWire(AA, ext, barTicks=[2, 7], size=[7, 6], axisLabelSize=25,
                   tickSize=15, c='r',
                   save=False, name='out.png'):
    """
    plot the meshframe plot
    """
    Aamp = abs(AA)
    
    X = np.linspace(ext[0], ext[1], Aamp.shape[1])
    Y = np.linspace(ext[2], ext[3], Aamp.shape[0])
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(X, Y, Aamp)
    for i in range(Aamp.shape[0]):
        plot(X[i], Y[i], Aamp[i], c=c, alpha=0.4)
    ax.set_xlabel(r'$x$', fontsize=axisLabelSize)
    ax.set_ylabel(r'$t$', fontsize=axisLabelSize)
    ax.set_zlabel(r'$|A|$', fontsize=axisLabelSize)

    if tickSize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tickSize)

    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
        plt.close()
    else:
        plt.show(block=False)
    

def plotConfigWireFourier(cgl, aa, ext, barTicks=[2, 7], size=[7, 6],
                          axisLabelSize=25, tickSize=15, c='r',
                          save=False, name='out.png'):
    plotConfigWire(cgl.Fourier2Config(aa), ext, barTicks, size,
                   axisLabelSize, tickSize, c, save, name)
    


def plotPhase(cgl, aa, ext, barTicks=[-3, 0, 3],
              colortype='jet',
              percent='5%', size=[4, 5],
              save=False, name='out.png'):
    """
    plot phase of filed A in cqcgl
    """
    phi = cgl.Fourier2Phase(aa)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    im = ax.imshow(phi, cmap=plt.get_cmap(colortype), extent=ext,
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax = dr.append_axes('right', size=percent, pad=0.05)
    bar = plt.colorbar(im, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
        plt.close()
    else:
        plt.show(block=False)
    

def plotOnePhase(cgl, a0, d=50, size=[6, 4], axisLabelSize=20,
                 save=False, name='out.png'):
    phi = cgl.Fourier2Phase(a0)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, d, phi.size), phi)
    ax.set_xlabel('x', fontsize=axisLabelSize)
    ax.set_ylabel(r'$\phi$', fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)

        
def sortByReal(eigvalue, eigvector=None):
    indx = np.argsort(eigvalue.real)
    indx = indx[::-1]
    if eigvector is None:
        return eigvalue[indx]
    else:
        return eigvalue[indx], eigvector[:, indx]


def cqcglSaveReqEV(fileName, groupName, a, wth, wphi, err, er, ei, vr, vi):
    f = h5py.File(fileName, 'a')
    req = f.create_group(groupName)
    req.create_dataset("a", data=a)
    req.create_dataset("wth", data=wth)
    req.create_dataset('wphi', data=wphi)
    req.create_dataset('err', data=err)
    req.create_dataset('er', data=er)
    req.create_dataset('ei', data=ei)
    req.create_dataset('vr', data=vr)
    req.create_dataset('vi', data=vi)
    f.close()


def cqcglSaveReqEVdi(fileName, di, index, a, wth, wphi, err, er, ei, vr, vi):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglSaveReqEV(fileName, groupName, a, wth, wphi, err, er, ei, vr, vi)


def cqcglReadReqAll(fileName, index, hasEV):
    f = h5py.File(fileName, 'r')
    gs = f.keys()
    f.close()
    xx = []                     # all req
    dis = []                    # all di
    for i in gs:
        di = float(i)
        dis.append(di)
        if hasEV:
            x = cqcglReadReqEVdi(fileName, di, index)
        else:
            x = cqcglReadReqdi(fileName, di, index)
        xx.append(x)
    return dis, xx


def cqcglReadReqEV(fileName, groupName):
    f = h5py.File(fileName, 'r')
    req = '/' + groupName + '/'
    a = f[req+'a'].value
    wth = f[req+'wth'].value
    wphi = f[req+'wphi'].value
    err = f[req+'err'].value
    er = f[req+'er'].value
    ei = f[req+'ei'].value
    vr = f[req+'vr'].value
    vi = f[req+'vi'].value
    f.close()
    return a, wth, wphi, err, er, ei, vr, vi


def cqcglReadReqEVdi(fileName, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglReadReqEV(fileName, groupName)


def cqcglReadReqEVAll(fileName, index):
    f = h5py.File(fileName, 'r')
    gs = f.keys()
    f.close()
    xx = []                     # all req
    dis = []                    # all di
    for i in gs:
        di = float(i)
        dis.append(di)
        x = cqcglReadReqEVdi(fileName, di, index)
        xx.append(x)
    return dis, xx


def cqcglAddEV2Req(fileName, groupName, er, ei, vr, vi):
    """
    try to write stability exponents and vectors
    to the existing rpo data group.
    parameters:
    er : real part of exponents
    ei : imaginary part of exponents
    vr : real part of vectors
    vi : imaginary part of vectors
    """
    f = h5py.File(fileName, 'a')
    f.create_dataset(groupName + '/' + 'er', data=er)
    f.create_dataset(groupName + '/' + 'ei', data=ei)
    f.create_dataset(groupName + '/' + 'vr', data=vr)
    f.create_dataset(groupName + '/' + 'vi', data=vi)
    f.close()


def cqcglMoveReqEV(inputFile, ingroup, outputFile, outgroup):
    """
    move a group from one file to another group of a another file
    """
    a, wth, wphi, err, er, ei, vr, vi = cqcglReadReqEV(inputFile, ingroup)
    cqcglSaveReqEV(outputFile, outgroup, a, wth, wphi, err, er, ei, vr, vi)


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


def cqcglMoveRPO(inputFile, ingroup, outputFile, outgroup):
    x, T, nstp, th, phi, err = cqcglReadRPO(inputFile, ingroup)
    cqcglSaveRPO(outputFile, outgroup, x, T, nstp, th, phi, err)

    
def cqcglMoveRPOdi(inputFile, outputFile, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglMoveRPO(inputFile, groupName, outputFile, groupName)


def cqcglReadRPOEV(fileName, groupName):
    f = h5py.File(fileName, 'r')
    rpo = '/' + groupName + '/'
    x = f[rpo+'x'].value
    T = f[rpo+'T'].value
    nstp = f[rpo+'nstp'].value
    th = f[rpo+'th'].value
    phi = f[rpo+'phi'].value
    err = f[rpo+'err'].value
    e = f[rpo+'e'].value
    v = f[rpo+'v'].value
    f.close()
    # return x, T[0], nstp, th[0], phi[0], err[0], e, v
    return x, T, nstp, th, phi, err, e, v


def cqcglReadRPOEVdi(fileName, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglReadRPOEV(fileName, groupName)


def cqcglReadRPOEVonly(fileName, groupName):
    f = h5py.File(fileName, 'r')
    rpo = '/' + groupName + '/'
    e = f[rpo+'e'].value
    v = f[rpo+'v'].value
    f.close()
    return e, v


def cqcglReadRPOEVonlydi(fileName, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglReadRPOEVonly(fileName, groupName)


def cqcglSaveRPOEV(fileName, groupName, x, T, nstp, th, phi, err, e, v):
    f = h5py.File(fileName, 'a')
    rpo = f.create_group(groupName)
    rpo.create_dataset("x", data=x)
    rpo.create_dataset("T", data=T)
    rpo.create_dataset("nstp", data=nstp)
    rpo.create_dataset("th", data=th)
    rpo.create_dataset('phi', data=phi)
    rpo.create_dataset('err', data=err)
    rpo.create_dataset('e', data=e)
    rpo.create_dataset('v', data=v)
    f.close()


def cqcglSaveRPOEVdi(fileName, di, index, x, T, nstp, th, phi, err, e, v):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglSaveRPOEV(fileName, groupName, x, T, nstp, th, phi, err, e, v)


def cqcglSaveRPOEVonly(fileName, groupName, e, v):
    f = h5py.File(fileName, 'a')
    # rpo = f.create_group(groupName)
    rpo = f[groupName]
    rpo.create_dataset('e', data=e)
    rpo.create_dataset('v', data=v)
    f.close()


def cqcglSaveRPOEVonlydi(fileName, di, index, e, v):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglSaveRPOEVonly(fileName, groupName, e, v)


def cqcglMoveRPOEV(inputFile, ingroup, outputFile, outgroup):
    x, T, nstp, th, phi, err, e, v = cqcglReadRPOEV(inputFile, ingroup)
    cqcglSaveRPOEV(outputFile, outgroup, x, T, nstp, th, phi, err, e, v)

    
def cqcglMoveRPOEVdi(inputFile, outputFile, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglMoveRPOEV(inputFile, groupName, outputFile, groupName)


def cqcglMoveRPOEVonly(inputFile, ingroup, outputFile, outgroup):
    e, v = cqcglReadRPOEVonly(inputFile, ingroup)
    cqcglSaveRPOEVonly(outputFile, outgroup, e, v)


def cqcglMoveRPOEVonlydi(inputFile, outputFile, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    cqcglMoveRPOEVonly(inputFile, groupName, outputFile, groupName)


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
    ratios = np.zeros((0,))
    for i in range(N-1):
        if x[i, 0] < 0 and x[i+1, 0] >= 0:
            ratio = -x[i, 0] / (x[i+1, 0] - x[i, 0])
            y = (x[i+1, 1] - x[i, 1]) * ratio + x[i, 1]
            z = (x[i+1, 2] - x[i, 2]) * ratio + x[i, 2]
            points = np.vstack((points, np.array([y, z])))
            index = np.append(index, i)
            ratios = np.append(ratios, ratio)
    if getIndex:
        return points, index, ratios
    else:
        return points


def getCurveIndex(points):
    """
    Obtain the curvalinear Index of a set of points.
    
    parameter:
    points   :  each row is a point
    """
    n = points.shape[0]         # number of points
    indices = range(n)
    sortedIndices = []

    norms = [np.linalg.norm(points[i]) for i in range(n)]
    start = np.argmin(norms)
    sortedIndices.append(start)
    indices.remove(start)
    
    for i in range(n-1):
        last = sortedIndices[-1]
        norms = [np.linalg.norm(points[i] - points[last]) for i in indices]
        minimal = indices[np.argmin(norms)]  # careful
        sortedIndices.append(minimal)
        indices.remove(minimal)

    return sortedIndices


def getCurveCoor(points):
    """
    Obtain the curvalinear coordinate of a set of points.
    
    parameter:
    points   :  each row is a point
    """
    n = points.shape[0]         # number of points
    indices = getCurveIndex(points)
    dis = np.empty(n)

    dis[0] = np.linalg.norm(points[indices[0]])
    for i in range(1, n):
        dis[i] = np.linalg.norm(points[indices[i]] - points[indices[i-1]])

    return dis
   
##################################################
#            2d CQCGL related                    #
##################################################


class CQCGL2dPlot():

    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def load(self, fileName, i, flag=1):
        """
        Load one state in the original storage order

        Parameters
        ==========
        flag : what to return
        """
        f = h5py.File(fileName, 'r')
        ds = '/' + format(i, '06d') + '/'
        if flag == 1 or flag == 0:
            a = f[ds+'ar'].value + 1j*f[ds+'ai'].value
        if flag == 2 or flag == 0:
            v = f[ds+'vr'].value + 1j*f[ds+'vi'].value

        f.close()

        if flag == 0:
            return a, v
        elif flag == 1:
            return a
        elif flag == 2:
            return v

    def loadSeq(self, fileName, x, y, ids):
        """
        Avoid using the whole mesh since it is slow.

        Parameters
        ==========
        x, y : the x and y location of the mesh
        """
        n = len(ids)
        data = np.zeros(n, dtype=np.complex)
        f = h5py.File(fileName, 'r')
        for i in ids:
            ds = '/' + format(i, '06d') + '/'
            data[i] = f[ds+'ar'][x, y] + 1j*f[ds+'ai'][x, y]
        f.close()
        return data

    def plotOneState(self, cgl, a, save=False, name='out.png',
                     colortype='jet', percent=0.05, size=[7, 5],
                     barTicks=None, axisLabelSize=25,
                     plotType=0):
        """
        Parameters
        ==========
        plotType : 0 => heat plot
                   1 => 3d mesh plot
                   else => both together
        """
        A = cgl.Fourier2Config(a)
        aA = np.abs(A).T

        if plotType == 0:
            fig, ax = pl2d(size=size, labs=[None, None],
                           axisLabelSize=axisLabelSize)
            ax.grid('on')
            im = ax.imshow(aA, cmap=plt.get_cmap(colortype), aspect='equal',
                           origin='lower', extent=[0, self.dx, 0, self.dy])
            dr = make_axes_locatable(ax)
            cax = dr.append_axes('right', size=percent, pad=0.05)
            if barTicks is not None:
                plt.colorbar(im, cax=cax, ticks=barTicks)
            else:
                plt.colorbar(im, cax=cax)
            ax2d(fig, ax, save=save, name=name)
            # plotMat(aA, save=save, name=name)

        else:
            X = np.linspace(0, self.dx, aA.shape[1])
            Y = np.linspace(0, self.dy, aA.shape[0])
            X, Y = np.meshgrid(X, Y)

            if plotType == 1:
                fig, ax = pl3d(size=size, labs=[r'$x$', r'$y$', r'$|A|$'],
                               axisLabelSize=axisLabelSize)
                surf = ax.plot_surface(X, Y, aA, rstride=10, cstride=10,
                                       cmap=plt.get_cmap(colortype),
                                       linewidth=0, antialiased=False)

                if barTicks is not None:
                    fig.colorbar(surf, fraction=percent, shrink=0.6, aspect=20,
                                 ticks=barTicks)
                else:
                    fig.colorbar(surf, fraction=percent, shrink=0.6, aspect=20,
                                 ticks=barTicks)

                ax3d(fig, ax, save=save, name=name)

            else:
                fig = plinit(size)
                ax1 = ax3dinit(fig, 121, labs=[r'$x$', r'$y$', r'$|A|$'],
                               axisLabelSize=axisLabelSize)
                ax2 = ax2dinit(fig, 122, labs=[None, None])
                ax2.grid('on')
                surf = ax1.plot_surface(X, Y, aA, rstride=10, cstride=10,
                                        cmap=plt.get_cmap(colortype),
                                        linewidth=0, antialiased=False)

                im = ax2.imshow(aA, cmap=plt.get_cmap(colortype),
                                aspect='equal', origin='lower',
                                extent=[0, self.dx, 0, self.dy])
                dr = make_axes_locatable(ax2)
                cax = dr.append_axes('right', size=percent, pad=0.05)
                if barTicks is not None:
                    plt.colorbar(im, cax=cax, ticks=barTicks)
                else:
                    plt.colorbar(im, cax=cax)
                ax2d(fig, ax2, save=save, name=name)

    def makeMovie(self, folderName, name='out.mp4'):
        """
        Parameters
        ==========
        folderName : name of the figure folder
        """
        files = "'" + folderName + '/*.png' + "'"
        command = ('ffmpeg -f image2 -r 6 -pattern_type glob -i' +
                   ' ' + files + ' ' + name)
        os.system(command)

    def savePlots(self, cgl, f1, sids, f2, plotType=0, size=[7, 5]):
        """
        Parameters
        ==========
        f1 : h5 data file
        f2 : save folder name
        onlyMovie: true => figures are deleted
        """
        if os.path.exists(f2):
            print 'folder already exists'
        else:
            os.makedirs(f2)
            for i in sids:
                name = f2+'/a'+format(i, '06d')+'.png'
                a = self.load(f1, i)
                self.plotOneState(cgl, a, save=True, size=size,
                                  name=name, plotType=plotType)

    def saveReq(self, fileName, groupName, a, wthx, wthy, wphi, err):
        f = h5py.File(fileName, 'a')
        req = f.create_group(groupName)
        req.create_dataset("ar", data=a.real)
        req.create_dataset("ai", data=a.imag)
        req.create_dataset("wthx", data=wthx)
        req.create_dataset("wthy", data=wthy)
        req.create_dataset('wphi', data=wphi)
        req.create_dataset('err', data=err)
        f.close()

    def saveReqdi(self, fileName, di, index, a, wthx, wthy, wphi, err):
        groupName = format(di, '.6f') + '/' + str(index)
        return self.saveReq(fileName, groupName, a, wthx, wthy, wphi, err)


############################################################
#                        KS related                        #
############################################################


def KSreadEq(fileName, idx):
    f = h5py.File(fileName, 'r')
    req = '/' + 'E' + '/' + str(idx) + '/'
    a = f[req+'a'].value
    err = f[req+'err'].value
    f.close()

    return a, err


def KSreadReq(fileName, idx):
    f = h5py.File(fileName, 'r')
    req = '/' + 'tw' + '/' + str(idx) + '/'
    a = f[req+'a'].value
    w = f[req+'w'].value
    err = f[req+'err'].value
    f.close()

    return a, w, err


def KSstabEig(ks, a0):
    stab = ks.stab(a0).T
    eigvalue, eigvector = eig(stab)
    eigvalue, eigvector = sortByReal(eigvalue, eigvector)
    return eigvalue, eigvector


def KSstabReqEig(ks, a0, w):
    stab = ks.stabReq(a0, w).T
    eigvalue, eigvector = eig(stab)
    eigvalue, eigvector = sortByReal(eigvalue, eigvector)
    return eigvalue, eigvector

    
def KSreadPO(fileName, poType, idx):
    f = h5py.File(fileName, 'r')
    po = '/' + poType + '/' + str(idx) + '/'
    a = f[po+'a'].value
    T = f[po+'T'].value
    nstp = np.int(f[po+'nstp'].value)
    r = f[po+'r'].value
    s = 0
    if poType == 'rpo':
        s = f[po+'s'].value
    f.close()
    return a, T, nstp, r, s


def KSreadFE(fileName, poType, idx):
    f = h5py.File(fileName, 'r')
    po = '/' + poType + '/' + str(idx) + '/'
    fe = f[po+'e'].value
    f.close()
    return fe


def KSreadFV(fileName, poType, idx):
    f = h5py.File(fileName, 'r')
    po = '/' + poType + '/' + str(idx) + '/'
    fv = f[po+'ve'].value
    f.close()
    return fv


def KSreadFEFV(fileName, poType, idx):
    f = h5py.File(fileName, 'r')
    po = '/' + poType + '/' + str(idx) + '/'
    fe = f[po+'e'].value
    fv = f[po+'ve'].value
    f.close()
    return fe, fv


def KScopyTo(inFile, outFile, poType, r):
    inF = h5py.File(inFile, 'r')
    outF = h5py.File(outFile, 'a')
    if not ('/' + poType in outF):
        outF.create_group('/' + poType)
    for i in r:
        print i
        ds = '/' + poType + '/' + str(i)
        h5py.h5o.copy(inF.id, ds, outF.id, ds)
        
    inF.close()
    outF.close()
    

def KSplotColorMapOrbit(aa, ext, barTicks=[-0.03, 0.03], colortype='jet',
                        percent='5%', size=[3, 6], axisLabelSize=20,
                        axisOn=True, barOn=True,
                        save=False, name='out'):
    """
    plot the color map of the states
    """
    half1 = aa[:, 0::2] + 1j*aa[:, 1::2]
    half2 = aa[:, 0::2] - 1j*aa[:, 1::2]
    M = half1.shape[0]
    aaWhole = np.hstack((np.zeros((M, 1)), half1,
                         np.zeros((M, 1)), half2[:, ::-1]))
    AA = np.fft.ifftn(aaWhole, axes=(1,)).real  # only the real part
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    if axisOn:
        ax.set_xlabel(r'$x$', fontsize=axisLabelSize)
        ax.set_ylabel(r'$t$', fontsize=axisLabelSize)
        
    im = ax.imshow(AA, cmap=plt.get_cmap(colortype), extent=ext,
                   aspect='auto', origin='lower')
    ax.grid('on')
    if barOn:
        dr = make_axes_locatable(ax)
        cax = dr.append_axes('right', size=percent, pad=0.05)
        bar = plt.colorbar(im, cax=cax, ticks=barTicks)
        
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name+'.png', format='png')
        plt.savefig(name+'.eps', format='eps')
    else:
        plt.show(block=False)


def KSplotPoHeat(ks, fileName, poType, poId, NT=1, Ts=100, fixT=False):
    """
    plot the heat map of ppo/rpo.
    Sometimes, a few periods make it easy to observe the state space.
    Also, fixing an integration time makes it easy to see the transition
    
    NT : the number of periods need to be ploted
    """
    a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
    h = T / nstp
    if fixT:
        aa = ks.intg(a0, h, np.int(Ts/h), 5)
        KSplotColorMapOrbit(aa, [0, ks.d, 0, Ts])
    else:
        aa = ks.intg(a0, h, nstp*NT, 5)
        KSplotColorMapOrbit(aa, [0, ks.d, 0, T*NT])


############################################################
#                        Common functions                  #
############################################################


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

