from personalFunctions import *

class KSplot():
    """
    KS help class.
    Its main funciton is to load data and plot figures.
    
    """
    
    def __init__(self, ks=None):
        """
        initialization with 
        >> ksp = KSplot(ks)
        or
        >> ksp = KSplot()
        """
        self.ks = ks
        self.N = ks.N if ks is not None else None
        self.L = ks.d if ks is not None else None
        # pass

    def F2C(self, aa):
        """
        Fourier modes a_k(t) to physical state u(x,t)

        Parameter
        ======
        aa : real and imaignary part of the Fourier modes. Can be 1d or 2d 

        Return
        ======
        AA : the physical space states (real).
        """
        if aa.ndim == 1:
            half1 = aa[0::2] + 1j*aa[1::2]
            half2 = aa[0::2] - 1j*aa[1::2]
            aaWhole = np.hstack(([0], half1, [0], half2[::-1]))
            AA = np.fft.ifft(aaWhole).real  # only the real part
        else:
            half1 = aa[:, 0::2] + 1j*aa[:, 1::2]
            half2 = aa[:, 0::2] - 1j*aa[:, 1::2]
            M = half1.shape[0]
            aaWhole = np.hstack((np.zeros((M, 1)), half1,
                                 np.zeros((M, 1)), half2[:, ::-1]))
            AA = np.fft.ifftn(aaWhole, axes=(1,)).real  # only the real part

        # because we have a different normalization conversion for FFT,
        # we need to multiply the result with N.
        return self.N * AA     

    def readEq(self, fileName, idx):
        f = h5py.File(fileName, 'r')
        req = '/' + 'E' + '/' + str(idx) + '/'
        a = f[req+'a'].value
        err = f[req+'err'].value
        f.close()

        return a, err

    def readReq(self, fileName, idx):
        f = h5py.File(fileName, 'r')
        req = '/' + 'tw' + '/' + str(idx) + '/'
        a = f[req+'a'].value
        w = f[req+'w'].value
        err = f[req+'err'].value
        f.close()

        return a, w, err
    
    def stabEig(self, a0):
        stab = self.ks.stab(a0).T
        eigvalue, eigvector = LA.eig(stab)
        eigvalue, eigvector = sortByReal(eigvalue, eigvector)
        return eigvalue, eigvector


    def stabReqEig(self, a0, w):
        stab = self.ks.stabReq(a0, w).T
        eigvalue, eigvector = LA.eig(stab)
        eigvalue, eigvector = sortByReal(eigvalue, eigvector)
        return eigvalue, eigvector

    def toStr(self, poType, idx, L=22, flag=0):
        if flag == 0:
            return poType + '/' + format(idx, '06d')
        else:
            return format(L, '010.6f') + '/' + poType + '/' + format(idx, '06d')

    def checkExist(self, fileName, groupName):
        f = h5py.File(fileName, 'r')
        x = groupName in f
        f.close()
        return x

    def readPO(self, fileName, groupName, isRPO, hasNstp=True, flag=0):
        f = h5py.File(fileName, 'r')
        DS = '/' + groupName + '/'
        a = f[DS+'a'].value
        T = f[DS+'T'].value
        nstp = f[DS+'nstp'].value if hasNstp else 0
        err = f[DS+'err'].value
        theta = f[DS+'theta'].value if isRPO else 0
        e = f[DS+'e'].value if flag > 0 else None
        v = f[DS+'v'].value if flag > 1 else None
        f.close()
        return a, T, nstp, theta, err, e, v


    def copyTo(self, inFile, outFile, poType, r):
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


    def oneConfig(self, A, isFourier=True, 
                  size=[6, 5], labs=[r'$x$', r'$u$'],
                  xlim=[0, 22],
                  axisLabelSize=20, tickSize=None,
                  save=False, name='out.png'):
        """
        plot the configuration at one point 
        or plot one eigen vector.
        
        Parameter
        ======
        isFourier : is the input the Fourier modes of A. If it is, then we need back FFT.
        """
        fig, ax = pl2d(size=size, labs=labs, xlim=xlim, 
                       axisLabelSize=axisLabelSize, tickSize=tickSize)
        if isFourier:
            A = self.F2C(A)
        ax.plot(np.linspace(0, self.L, A.shape[0]), A, lw=1.5)
        ax2d(fig, ax, save=save, name=name)

    def config(self, A, ext, isFourier=True, barOn=True,
               barTicks=[-3, -2, -1, 0, 1, 2, 3], colortype='jet',
               percent='5%', size=[3, 6], labs=[r'$x$', r'$t$'],
               axisLabelSize=20, tickSize=None,
               save=False, name='out.png'):
        """
        plot the color map of the states
        """
        if isFourier:
            A = self.F2C(A)

        fig, ax = pl2d(size=size, labs=labs, axisLabelSize=axisLabelSize, tickSize=tickSize)
        im = ax.imshow(A, cmap=plt.get_cmap(colortype), extent=ext,
                       aspect='auto', origin='lower')
        ax.grid('on')
        if barOn:
            dr = make_axes_locatable(ax)
            cax = dr.append_axes('right', size=percent, pad=0.05)
            bar = plt.colorbar(im, cax=cax, ticks=barTicks)

        ax2d(fig, ax, save=save, name=name)


    def poHeat(self, fileName, poType, poId, NT=1, Ts=100, fixT=False):
        """
        plot the heat map of ppo/rpo.
        Sometimes, a few periods make it easy to observe the state space.
        Also, fixing an integration time makes it easy to see the transition

        NT : the number of periods need to be ploted
        """
        a0, T, nstp, r, s = self.readPO(fileName, poType, poId)
        h = T / nstp
        if fixT:
            aa = self.ks.intg(a0, h, np.int(Ts/h), 5)
            self.config(aa, [0, ks.d, 0, Ts])
        else:
            aa = self.ks.intg(a0, h, nstp*NT, 5)
            self.config(aa, [0, ks.d, 0, T*NT])

