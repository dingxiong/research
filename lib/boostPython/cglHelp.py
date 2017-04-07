from personalFunctions import *

##################################################
#            1d CQCGL related                    #
##################################################
class CQCGLBase():

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
    
    def config(self, AA, ext, isFourier=True, tt=None, yls=None, timeMode=1,
               barTicks=[1, 2, 3], colortype='jet',
               percent='5%', size=[4, 5], labs=[r'$x$', r'$t$'],
               axisLabelSize=20, tickSize=None,
               save=False, name='out.png'):
        """
        plot the color map of the states
        """
        if tt is not None:
            if timeMode == 1:
                n = len(tt)
                ids = [bisect_left(tt, yl) for yl in yls]
                yts = [x / float(n) * ext[3] for x in ids]
                ax.set_yticks(yts)
                ax.set_yticklabels(yls)
            else:
                idx = self.sliceTime(tt)
                AA = AA[idx, :]

        if isFourier:
            AA = self.cgl.Fourier2Config(AA)
        Aamp = np.abs(AA)
        fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize, tickSize=tickSize)
        im = ax.imshow(Aamp, cmap=plt.get_cmap(colortype), extent=ext,
                       aspect='auto', origin='lower')
        ax.grid('on')
        dr = make_axes_locatable(ax)
        cax = dr.append_axes('right', size=percent, pad=0.05)
        plt.colorbar(im, cax=cax, ticks=barTicks)

        ax2d(fig, ax, save=save, name=name)

    def sliceTime(self, t):
        n = len(t)
        hs = t[1:] - t[:-1]
        hmax = np.max(hs)
        idx = [0]
        s = 0
        for i in range(n-1):
            s += hs[i]
            if(s >= hmax):
                idx.append(i)
                s = 0
        return idx

class CQCGLreq(CQCGLBase):
    """
    relative equilibrium related 
    """
    def __init__(self, cgl=None):
        CQCGLBase.__init__(self, cgl)
    
        
    def read(self, fileName, groupName, sub=False, flag=0):
        """
        sub : is subspace ?
        """
        f = h5py.File(fileName, 'r')
        req = '/' + groupName + '/'
        a = f[req+'a'].value
        wth = f[req+'wth'].value if not sub else None
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

    def readDi(self, fileName, di, index, flag=0):
        groupName = format(di, '.6f') + '/' + str(index)
        return self.read(fileName, groupName, flag)
        
    def eigReq(self, a0, wth0, wphi0, sub=False):
        if sub:
            stabMat = self.cgl.stabReq(a0, wphi0).T 
        else:
            stabMat = self.cgl.stabReq(a0, wth0, wphi0).T 
        e, v = LA.eig(stabMat)
        e, v = sortByReal(e, v)
        v = v.T.copy()

        return e, v

    def getAxes(self, fileName, Bi, Gi, index, flag):
        a, wth, wphi, err, e, v = self.read(fileName, self.toStr(Bi, Gi, index), flag=2)
        aH = self.cgl.orbit2slice(a, flag)[0]
        vrH = self.cgl.ve2slice(v.real.copy(), a, flag)
        viH = self.cgl.ve2slice(v.imag.copy(), a, flag)
        
        return e, aH, vrH + 1j*viH
        

class CQCGLrpo(CQCGLBase):

    def __init__(self, cgl=None):
        CQCGLBase.__init__(self, cgl)

    def read(self, fileName, groupName, flag=0):
        """
        read rpo.
        
        flag : = 0 : only load basic initial condition
               = 1 : also load multiplier
               = 2 : load multiplier and eigenvectors

        Example usage:
        >> rpo = CQCGLrpo(cgl)
        >> a0, T, nstp, th0, phi0, err = rpo.read(f, rpo.toStr(Bi, Gi, index), 0)
        """
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
        elif flag == 1:
            return x, T, nstp, th, phi, err, e
        elif flag == 2:
            return x, T, nstp, th, phi, err, e, v
        else :
            print "invalid flag"
    
    def readAll(self, fileName, index, hasEV):
        f = h5py.File(fileName, 'r')
        gs = f.keys()
        f.close()
        xx = []                     # all rpo
        dis = []                    # all di
        for i in gs:
            di = float(i)
            dis.append(di)
            if hasEV:
                x = cqcglReadEVdi(fileName, di, index)
            else:
                x = cqcglReaddi(fileName, di, index)
            xx.append(x)
        return dis, xx

    def save(self, fileName, groupName, x, T, nstp, th, phi, err, e=None, v=None):
        """
        save rpo.
        
        Example usage:
        >> rpo = CQCGLrpo(cgl)
        >> rpo.save(f, rpo.toStr(Bi, Gi, index), x, T, nstp, th, phi, err)
        """
        f = h5py.File(fileName, 'a')
        rpo = f.create_group(groupName)
        rpo.create_dataset("x", data=x)
        rpo.create_dataset("T", data=T)
        rpo.create_dataset("nstp", data=nstp)
        rpo.create_dataset("th", data=th)
        rpo.create_dataset('phi', data=phi)
        rpo.create_dataset('err', data=err)
        if e is not None:
            rpo.create_dataset('er', data=e.real)
            rpo.create_dataset('ei', data=e.imag)
        if v is not None:
            rpo.create_dataset('v', data=v)
        f.close()

    def move(self, inFile, ingroup, outFile, outgroup, flag=0):
        e, v = None, None
        pack = self.read(inFile, ingroup, flag)
        if flag == 0:
            x, T, nstp, th, phi, err = pack
        elif flag == 1:
            x, T, nstp, th, phi, err, e = pack
        elif flag == 2:
            x, T, nstp, th, phi, err, e, v = pack
        else:
            print "error of move"
            
        self.save(outFile, outgroup, x, T, nstp, th, phi, err, e, v)


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
