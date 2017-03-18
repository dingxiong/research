from personalFunctions import *

##################################################
#            1d CQCGL related                    #
##################################################
class CQCGLplot():
    def __init__(self, cgl=None):
        self.cgl = cgl

    def oneConfig(self, A, d=50, isFourier=True, 
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
            ids = [bisect_left(tt, yl) for yl in yls]
            yts = [x / float(n) * ext[3] for x in ids]
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
        load a sequence of points but only at location (x, y)
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

    def oneState(self, cgl, a, save=False, name='out.png',
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

    def savePlots(self, cgl, f1, f2, sids=None, plotType=0, size=[7, 5]):
        """
        Parameters
        ==========
        f1 : h5 data file
        f2 : save folder name
        sids : a list which contains the ids of the states
               if it is None, then all states will be loaded
        """
        if os.path.exists(f2):
            print 'folder already exists'
        else:
            os.makedirs(f2)
            if sids == None:
                f = h5py.File(f1, 'r')
                sids = range(len(f.keys()))
                f.close()

            for i in sids:
                name = f2+'/a'+format(i, '06d')
                a = self.load(f1, i)
                self.oneState(cgl, a, save=True, size=size,
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

    def toStr(self, Bi, Gi, index):
        if abs(Bi) < 1e-6:
            Bi = 0
        if abs(Gi) < 1e-6:
            Gi = 0
        return (format(Bi, '013.6f') + '/' + format(Gi, '013.6f') +
                '/' + str(index))

    def loadReq(self, fileName, groupName):
        f = h5py.File(fileName, 'r')
        req = '/' + groupName + '/'
        a = f[req+'ar'].value + 1j*f[req+'ai'].value
        wthx = f[req+'wthx'].value
        wthy = f[req+'wthy'].value
        wphi = f[req+'wphi'].value
        err = f[req+'err'].value
        f.close()

        return a, wthx, wthy, wphi, err
