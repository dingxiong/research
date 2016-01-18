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
                    axisLabelSize=20,
                    save=False, name='out.png'):
    """
    plot the color map of the states
    """
    Ar = AA[:, 0::2]
    Ai = AA[:, 1::2]
    Aamp = abs(Ar + 1j*Ai)
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


def plotConfigSpaceFromFourier(cgl, aa, ext, barTicks=[0, 3],
                               colortype='jet',
                               percent='5%', size=[4, 5],
                               axisLabelSize=20,
                               save=False, name='out.png'):
    """
    plot the configuration from Fourier mode
    """
    plotConfigSpace(cgl.Fourier2Config(aa), ext, barTicks,
                    colortype, percent, size,
                    axisLabelSize, save, name)


def plotConfigSurface(AA, ext, barTicks=[2, 4], colortype='jet',
                      percent='5%', size=[7, 5], axisLabelSize=25,
                      save=False, name='out.png'):
    """
    plot the color map of the states
    """
    Ar = AA[:, 0::2]
    Ai = AA[:, 1::2]
    Aamp = abs(Ar + 1j*Ai)
    
    X = np.linspace(ext[0], ext[1], Aamp.shape[1])
    Y = np.linspace(ext[2], ext[3], Aamp.shape[0])
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Aamp, rstride=10, cstride=10,
                           cmap=plt.get_cmap(colortype),
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, fraction=0.05, shrink=0.6, aspect=20, ticks=barTicks)
    
    ax.set_xlabel('x', fontsize=axisLabelSize)
    ax.set_ylabel('t', fontsize=axisLabelSize)
    ax.set_zlabel(r'$|A|$', fontsize=axisLabelSize)

    # dr = make_axes_locatable(ax)
    # cax = dr.append_axes('right', size=percent, pad=0.05)
    # plt.colorbar(surf, cax=cax, ticks=barTicks)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)


def plotConfigSurfaceFourier(cgl, aa, ext, barTicks=[2, 4],
                             colortype='jet',
                             percent='5%', size=[7, 5],
                             axisLabelSize=25,
                             save=False, name='out.png'):
    plotConfigSurface(cgl.Fourier2Config(aa), ext, barTicks,
                      colortype, percent, size,
                      axisLabelSize,
                      save, name)


def plotOneConfig(A, d=50, size=[6, 4], axisLabelSize=20,
                  save=False, name='out.png'):
    """
    plot the configuration at one point
    """
    Aamp = abs(A[0::2]+1j*A[1::2])
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, d, Aamp.size), Aamp)
    ax.set_xlabel('x', fontsize=axisLabelSize)
    ax.set_ylabel(r'$|A|$', fontsize=axisLabelSize)
    fig.tight_layout(pad=0)
    if save:
        plt.savefig(name)
    else:
        plt.show(block=False)


def plotOneConfigFromFourier(cgl, a0, d=50, size=[6, 4], axisLabelSize=20,
                             save=False, name='out.png'):
    """
    plot the configuration at one point from Fourier mode
    """
    plotOneConfig(cgl.Fourier2Config(a0).squeeze(), d, size,
                  axisLabelSize, save, name)


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


def cqcglSaveReq(fileName, groupName, a, wth, wphi, err):
    f = h5py.File(fileName, 'a')
    req = f.create_group(groupName)
    req.create_dataset("a", data=a)
    req.create_dataset("wth", data=wth)
    req.create_dataset('wphi', data=wphi)
    req.create_dataset('err', data=err)
    f.close()


def cqcglSaveReqdi(fileName, di, index, a, wth, wphi, err):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglSaveReq(fileName, groupName, a, wth, wphi, err)


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


def cqcglReadReq(fileName, groupName):
    f = h5py.File(fileName, 'r')
    req = '/' + groupName + '/'
    a = f[req+'a'].value
    wth = f[req+'wth'].value
    wphi = f[req+'wphi'].value
    err = f[req+'err'].value
    f.close()
    return a, wth, wphi, err


def cqcglReadReqdi(fileName, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglReadReq(fileName, groupName)


def cqcglReadReqAll(fileName, index):
    f = h5py.File(fileName, 'r')
    gs = f.keys()
    f.close()
    xx = []                     # all req
    dis = []                    # all di
    for i in gs:
        di = float(i)
        dis.append(di)
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


def cqcglReadRPO(fileName, groupName):
    f = h5py.File(fileName, 'r')
    req = '/' + groupName + '/'
    x = f[req+'x'].value
    T = f[req+'T'].value
    nstp = f[req+'nstp'].value
    th = f[req+'th'].value
    phi = f[req+'phi'].value
    err = f[req+'err'].value
    f.close()
    # return x[0], T[0], nstp[0], th[0], phi[0], err[0]
    return x, T, nstp, th, phi, err


def cqcglReadRPOdi(fileName, di, index):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglReadRPO(fileName, groupName)

    
def cqcglReadRPOAll(fileName, index, hasEV):
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


def cqcglSaveRPO(fileName, groupName, x, T, nstp, th, phi, err):
    f = h5py.File(fileName, 'a')
    rpo = f.create_group(groupName)
    rpo.create_dataset("x", data=x)
    rpo.create_dataset("T", data=T)
    rpo.create_dataset("nstp", data=nstp)
    rpo.create_dataset("th", data=th)
    rpo.create_dataset('phi', data=phi)
    rpo.create_dataset('err', data=err)
    f.close()


def cqcglSaveRPOdi(fileName, di, index, x, T, nstp, th, phi, err):
    groupName = format(di, '.6f') + '/' + str(index)
    return cqcglSaveRPO(fileName, groupName, x, T, nstp, th, phi, err)


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
    
############################################################
#                        KS related                        #
############################################################


def KSreadPO(fileName, poType, idx):
    f = h5py.File(fileName, 'r')
    po = '/' + poType + '/' + str(idx) + '/'
    a = f[po+'a'].value
    T = f[po+'T'].value[0]
    nstp = np.int(f[po+'nstp'].value[0])
    r = f[po+'r'].value[0]
    s = 0
    if poType == 'rpo':
        s = f[po+'s'].value[0]
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
        print i;
        ds =  '/' + poType + '/' + str(i)
        h5py.h5o.copy(inF.id, ds, outF.id, ds)
        
    inF.close()
    outF.close()
    

def KSplotColorMapOrbit(aa, ext, barTicks=[-0.03, 0.03], colortype='jet',
                        percent='5%', size=[3, 6],
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
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('t', fontsize=20)
        
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
    c = LA.norm(dot(Q.T, V))
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
            rve[:, i] = ve[:, i]
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
