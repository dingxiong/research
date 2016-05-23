from personalFunctions import *
from py_ks import *
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV


class KSReq():
    """
    This class is designed to study the structure of KS
    system in the symmetry reduced fundamental state space.
    
    Slice is chosen to make the pth mode's real part zero.
    Note, for 1st mode slice, E2 and E3 are on the slice border,
    it makes no sence to reduce symmetry for them, nor for the projection of
    vectors.
    """
    
    def __init__(self, N, L, fileName, p):
        """
        N : number of Fourier modes in KS
        L : domain size of KS
        """
        self.N = N
        self.L = L
        self.fileName = fileName
        self.p = p
        self.ks = pyKS(N, L)
        
        self.req, self.ws, self.reqr, self.eq, self.eqr = self.loadRE(fileName, p)

        self.Eg = self.EqGroupOrbit()
        self.Es, self.Ev, self.Evr = self.calEsEv(p)

    def loadRE(self, fileName, p):
        """
        load all req and eq and their corresponding
        symmetry reduced states in the fundamental
        domain.

        fileName : path to the req data file
        p : the fourier mode used to reduce symmetry
        """
        N = self.N
        ks = self.ks

        req = np.zeros((2, N-2))
        ws = np.zeros(2)
        reqr = np.zeros((2, N-2))
        for i in range(2):
            a0, w, err = KSreadReq(fileName, i+1)
            req[i] = a0
            ws[i] = w
            tmp = ks.redO2f(a0, p)
            reqr[i] = tmp[0]

        eq = np.zeros((3, N-2))
        eqr = np.zeros((3, N-2))
        for i in range(3):
            a0, err = KSreadEq(fileName, i+1)
            eq[i] = a0
            tmp = ks.redO2f(a0, p)
            eqr[i] = tmp[0]

        return req, ws, reqr, eq, eqr

    def EqGroupOrbit(self):
        """
        E2 and E3 may be in the slice border, so
        just obtain their group orbits
        """
        n = 100
        E2 = np.zeros((n, self.N-2))
        E3 = np.zeros((n, self.N-2))
        for i in range(n):
            th = 2*i*np.pi / n
            a1 = self.ks.Rotation(self.eqr[1], th)
            a2 = self.ks.Rotation(self.eqr[2], th)
            E2[i] = a1
            E3[i] = a2
        
        Eg = {'E2': E2, 'E3': E3}
        return Eg

    def calEsEv(self, p):
        """
        get the eigenvector of eq/req
        """
        Eqe = []
        EqV = []
        EqVr = []
        for i in range(3):
            es, evt = KSstabEig(self.ks, self.eqr[i])
            ev = Tcopy(realve(evt))
            pev = self.ks.redV(ev, self.eqr[i], p, True)
            Eqe.append(es)
            EqV.append(ev)
            EqVr.append(pev)
            
        Reqe = []
        ReqV = []
        ReqVr = []
        for i in range(2):
            if i == 0:
                w = -self.ws[i]
            if i == 1:
                w = self.ws[i]
            es, evt = KSstabReqEig(self.ks, self.reqr[i], w)
            ev = Tcopy(realve(evt))
            pev = self.ks.redV(ev, self.reqr[i], p, True)
            Reqe.append(es)
            ReqV.append(ev)
            ReqVr.append(pev)
        
        Es = {'e': Eqe, 'tw': Reqe}
        Ev = {'e': EqV, 'tw': ReqV}
        Evr = {'e': EqVr, 'tw': ReqVr}
        return Es, Ev, Evr
        
    def loadPO(self, fileName, poIds, p, bases=None, x0=None):
        """
        Load rpo and ppo and reduce the symmetry.
        If bases and orgin x0 are given, then also return the
        projection.
        
        p : the mode used to reduce symmetry
        return :
        aas : symmetry reduced state space
        pas : projected state space
        """
        aas = []
        aars = []
        aaps = []
        types = ['rpo', 'ppo']
        for i in range(2):
            poType = types[i]
            for poId in poIds[i]:
                a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
                h = T / nstp
                aa = self.ks.intg(a0, h, nstp, 5)
                aar = self.ks.redO2f(aa, p)[0]
                aas.append(aa)
                aars.append(aar)
                if bases is not None:
                    aaps.append(aar.dot(bases.T) - x0)

        if bases is not None:
            return aas, aars, aaps
        else:
            return aas, aars

    def getMu(self, x0, v0, p, T=200, r0=1e-5, nn=30):
        """
        get the unstable mainifold

        p : the mode used to reduce symmetry
        ---------
        return :
        aars : orbits in reduced fundamental domain
        dom : domain indices
        jumps : jumping indices
        """
        aars = []
        dom = []
        jumps = []
        for i in range(nn):
            a0 = x0 + r0 * (i+1) * v0
            aa = self.ks.aintg(a0, 0.01, T, 1)
            raa, dids, ths = self.ks.redO2f(aa, p)
            aars.append(raa)
            dom.append(dids)
            jumps.append(getJumpPts(dids))
        
        return aars, dom, jumps

    def getMuAll(self, p, T=200, r0=1e-5, nn=30):
        """
        get the unstable manifolds af all eq/req
        """
        MuE = []
        for i in range(3):
            a0 = self.eqr[i]
            v0 = ksreq.Ev['e'][i][0]
            aars, dom, jumps = self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)
            MuE.append([aars, dom, jumps])
            if i == 0:
                v0 = ksreq.Ev['e'][i][2]
                aars, dom, jumps = self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)
                MuE.append([aars, dom, jumps])
            if i == 2:
                v0 = ksreq.Ev['e'][i][1]
                aars, dom, jumps = self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)
                MuE.append([aars, dom, jumps])

        MuTw = []
        for i in range(2):
            a0 = self.reqr[i]
            v0 = ksreq.Ev['tw'][i][0]
            aars, dom, jumps = self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)
            MuTw.append([aars, dom, jumps])
            if i == 0:
                v0 = ksreq.Ev['tw'][i][2]
                aars, dom, jumps = self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)
                MuTw.append([aars, dom, jumps])

        return MuE, MuTw

    def plotRE(self, ax, ii, do3d=True):
        """
        construct the plotting block for req/eq
        """
        reqr = self.reqr
        eqr = self.eqr

        c1 = ['r', 'b']
        for i in range(2):
            if do3d:
                ax.scatter(reqr[i, ii[0]], reqr[i, ii[1]], reqr[i, ii[2]],
                           c=c1[i], s=70,
                           edgecolors='none', label='TW'+str(i+1))
            else:
                ax.scatter(reqr[i, ii[0]], reqr[i, ii[1]],
                           c=c1[i], s=70, edgecolors='none', label='TW'+str(i+1))

        c2 = ['c', 'k', 'y']
        for i in range(3):
            if do3d:
                ax.scatter(eqr[i, ii[0]], eqr[i, ii[1]], eqr[i, ii[2]],
                           c=c2[i], s=70,
                           edgecolors='none', label='E'+str(i+1))
            else:
                ax.scatter(eqr[i, ii[0]], eqr[i, ii[1]], c=c2[i], s=70,
                           edgecolors='none', label='E'+str(i+1))

    def plotFundOrbit(self, ax, faa, jumps, ii, c=None, alpha=0.5):
        """
        plot orbit in the fundamental domain. sudden jumps are avoided.
        
        faa : a single orbit in the fundamental domain
        jumps : indices where orbit jumps from one domain to another
        ii : the plot index
        """
        if c is None:
            c = rand(3, 1)

        x = concatenate(([-1], jumps, [len(jumps)-1]))
        for i in range(len(x)-1):
            r = range(x[i]+1, x[i+1]+1)
            ax.plot(faa[r, ii[0]], faa[r, ii[1]], faa[r, ii[2]],
                    c=c, alpha=alpha)

    def getBases(self, etype, a, ii, w=0):
        """
        get projection bases

        pev : in-slice vector
        bases : selected bases
        """
        if etype == 'eq':
            es, evt = KSstabEig(self.ks, a)
        elif etype == 'req':
            es, evt = KSstabReqEig(self.ks, a, w)
            
        ev = Tcopy(realve(evt))
        pev = ks.redV(ev, a)[1]
        v1, v2, v3 = orthAxes(pev[ii[0]], pev[ii[1]], pev[ii[2]])
        bases = np.vstack((v1, v2, v3))

        return pev, bases

    """
    def getPoinc(data, theta):
        x, y = rotz(data[:, 0], data[:, 1], theta)
        aa = np.vstack((x, y, data[:, 2])).T
        n, m = aa.shape
        pc = np.zeros((n, m))
        pcf = np.zeros((n, m))
        num = 0
        for i in range(n-1):
            if aa[i, 0] < 0 and aa[i+1, 0] >= 0:
                p = int2p(aa[i], aa[i+1])
                pc[num] = p
                x, y = rotz(p[0], p[1], -theta)
                pcf[num] = np.array([x, y, p[2]])
                num += 1
        pc = pc[:num]
        pcf = pcf[:num]

        return pc, pcf


    def getPoinc2(data, theta1, theta2):
        x1, y1 = rotz(data[:, 0], data[:, 1], theta1)
        x2, y2 = rotz(data[:, 0], data[:, 1], theta2)
        aa1 = np.vstack((x1, y1, data[:, 2])).T
        aa2 = np.vstack((x2, y2, data[:, 2])).T
        n, m = aa1.shape
        pc = np.zeros((n, m))
        pcf = np.zeros((n, m))
        coe = np.zeros(n)
        ixs = np.zeros(n, dtype=np.int)
        num = 0
        for i in range(n-1):
            if aa1[i, 0] < 0 and aa1[i+1, 0] >= 0:
                p, c1 = int2p(aa1[i], aa1[i+1])
                if p[1] >= 0:
                    pc[num] = p
                    coe[num] = c1
                    ixs[num] = i
                    x, y = rotz(p[0], p[1], -theta1)
                    pcf[num] = np.array([x, y, p[2]])
                    num += 1
            if aa2[i, 0] < 0 and aa2[i+1, 0] >= 0:
                p, c1 = int2p(aa2[i], aa2[i+1])
                if p[1] <= 0:
                    pc[num] = p
                    coe[num] = c1
                    ixs[num] = i
                    x, y = rotz(p[0], p[1], -theta2)
                    pcf[num] = np.array([x, y, p[2]])
                    num += 1
        pc = pc[:num]
        pcf = pcf[:num]
        coe = coe[:num]
        ixs = ixs[:num]

        return pc, pcf, coe, ixs


    def ergoPoinc(ks, bases, x0, theta, si):
        a0 = rand(N-2) * 0.1
        aa = ks.intg(a0, 10000, 10000)
        poinc = np.zeros((0, 3))
        poincf = np.zeros((0, 3))
        paas = np.zeros((0, 3))
        for i in range(15):
            a0 = aa[-1]
            aa = ks.intg(a0, 850000, 100)[1:]
            raa, ths = ks.redO2(aa)
            paa = raa.dot(bases.T)
            paa -= x0
            pc, pcf = getPoinc(paa, theta)
            if si == 'p':
                ix = pc[:, 1] >= 0
                pc = pc[ix]
                pcf = pcf[ix]
            else:
                ix = pc[:, 1] <= 0
                pc = pc[ix]
                pcf = pcf[ix]
            poinc = np.vstack((poinc, pc))
            poincf = np.vstack((poincf, pcf))
            paas = np.vstack((paas, paa))
        return paas, poinc, poincf


    def ergoPoinc2(ks, bases, x0, theta1, theta2):
        N = ks.N
        a0 = rand(N-2) * 0.1
        aa = ks.intg(a0, 0.002, 10000, 10000)
        poinc = np.zeros((0, 3))
        poincf = np.zeros((0, 3))
        poincRaw = np.zeros((0, N-2))
        paas = np.zeros((0, 3))

        for i in range(20):
            a0 = aa[-1]
            aa = ks.intg(a0, 0.002, 500000, 10)[1:]
            raa, ths = ks.redO2(aa)
            paa = raa.dot(bases.T)
            paa -= x0
            pc, pcf, coe, ixs = getPoinc2(paa, theta1, theta2)
            raw = (coe * raa[ixs].T + (1-coe) * raa[ixs+1].T).T
            poincRaw = np.vstack((poincRaw, raw))
            poinc = np.vstack((poinc, pc))
            poincf = np.vstack((poincf, pcf))
            paas = np.vstack((paas, paa))
        return paas, poinc, poincf, poincRaw


    def poPoinc(fileName, poIds, bases, x0, theta1, theta2):
        '''
        pas : projected orbits
        poinc : poincare intersection points on the plane
        poincf : intersection points in the 3d space
        '''
        N = 64
        aas, pas = loadPO2(fileName, poIds, bases, x0)
        poinc = np.zeros((0, 3))
        poincf = np.zeros((0, 3))
        poincRaw = np.zeros((0, N-2))
        nums = []
        for i in range(len(pas)):
            pc, pcf, coe, ixs = getPoinc2(pas[i], theta1, theta2)
            nums.append(pc.shape[0])
            raw = (coe * aas[i][ixs].T + (1-coe) * aas[i][ixs+1].T).T
            poincRaw = np.vstack((poincRaw, raw))
            poinc = np.vstack((poinc, pc))
            poincf = np.vstack((poincf, pcf))

        return pas, poinc, poincf, poincRaw, np.array(nums)


    def getCurveIndex(x, y):
        '''
        for each row of x, try to find the corresponding
        row of y such that these two rows have the minimal
        distance.
        '''
        m, n = x.shape
        minDs = np.zeros(m)
        minIds = np.zeros(m, dtype=np.int)
        for i in range(m):
            dif = x[i]-y
            dis = norm(dif, axis=1)
            minDs[i] = np.min(dis)
            minIds[i] = np.argmin(dis)

        return minIds, minDs


    def getCurveCoordinate(sortId, poinc):
        '''
        get the curve coordinate
        '''
        m = sortId.shape[0]
        dis = np.zeros(m)
        coor = np.zeros(m)
        for i in range(1, m):
            dis[i] = dis[i-1] + norm(poinc[sortId[i]]-poinc[sortId[i-1]])

        for i in range(m):
            coor[sortId[i]] = dis[i]

        return dis, coor

        """
##############################################################################################################

if __name__ == '__main__':

    N = 64
    L = 22
    ksreq = KSReq(N, L, '../../data/ks22Reqx64.h5', 1)

    case = 10

    if case == 10:
        """
        view the unstable manifold of E2
        """
        nn = 30
        a0 = ksreq.eqr[0]
        v0 = ksreq.Ev['e'][0][0]
        aas, dom, jumps = ksreq.getMu(a0, v0, 1, nn=nn, T=110)
        ii = [1, 5, 3]

        doProj = False
        if doProj:
            Ori = eqr[1].copy()
            pev, bases = getBases(ks, 'eq', Ori, [0, 1, 3])
            OriP = Ori.dot(bases.T)
            reqr = reqr.dot(bases.T) - OriP
            eqr = eqr.dot(bases.T) - OriP
            paas = []
            for i in range(len(aas)):
                paas.append(aas[i].dot(bases.T) - OriP)

            ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ksreq.plotRE(ax, ii)
        if doProj:
            for i in range(nn):
                ksreq.plotFundOrbit(ax, paas, jumps[i], ii)
            ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
        else:
            for i in range(nn):
                ksreq.plotFundOrbit(ax, aas[i], jumps[i], ii)
            E2, E3 = ksreq.Eg['E2'], ksreq.Eg['E3']
            ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]])
            ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
        ax3d(fig, ax)

    if case == 20:
        """
        visualize the unstable manifold of eq and req together
        """
        nn = 10
        MuE, MuTw = ksreq.getMuAll(1, nn=nn)
        ii = [1, 3, 5]
        
        cs = ['r', 'b', 'c', 'k', 'y']
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ksreq.plotRE(ax, ii)
        for k in range(1, len(MuE)-3):
            for i in range(nn):
                ksreq.plotFundOrbit(ax, MuE[0][0][i], MuE[0][2][i],
                                    ii, c=cs[k])
        E2, E3 = ksreq.Eg['E2'], ksreq.Eg['E3']
        ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]])
        ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
        ax3d(fig, ax)

    if case == 40:
        """
        watch an ergodic trajectory after reducing O2 symmetry
        """
        N = 64
        d = 22
        ks = pyKS(N, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

        a0 = rand(N-2) * 0.1
        aa = ks.aintg(a0, 0.001, 300, 1)
        raa, dids, ths = ks.redO2f(aa, 1)

        ii = [1, 2, 3]

        doProj = False
        if doProj:
            pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
            paas = raa.dot(bases.T)

            reqr = reqr.dot(bases.T)
            eqr = eqr.dot(bases.T)
            paas -= eqr[1]
            reqr -= eqr[1]
            eqr -= eqr[1]

            ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        plotRE(ax, reqr, eqr, ii)
        if doProj:
            ax.plot(paas[:, ii[0]], paas[:, ii[1]],
                    paas[:, ii[2]], alpha=0.5)
        else:
            ax.plot(raa[:, ii[0]], raa[:, ii[1]], raa[:, ii[2]],
                    alpha=0.5)
        ax3d(fig, ax)

        doMovie = False
        if doMovie:
            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'],
                           xlim=[-1, 0.4], ylim=[-0.6, 0.6], zlim=[-0.15, 0.15],
                           isBlack=False)
            frame, = ax.plot([], [], [], c='gray', ls='-', lw=1, alpha=0.5)
            frame2, = ax.plot([], [], [], c='r', ls='-', lw=1.5, alpha=1)
            pts, = ax.plot([], [], [], 'co', lw=3)

            def anim(i):
                k = max(0, i-500)
                j = min(i, paas.shape[0])
                frame.set_data(paas[:k, ii[0]], paas[:k, ii[1]])
                frame.set_3d_properties(paas[:k, ii[2]])
                frame2.set_data(paas[k:j, ii[0]], paas[k:j, ii[1]])
                frame2.set_3d_properties(paas[k:j, ii[2]])
                pts.set_data(paas[j, ii[0]], paas[j, ii[1]])
                pts.set_3d_properties(paas[j, ii[2]])

                ax.view_init(30, 0.5 * i)
                return frame, frame2, pts

            ani = animation.FuncAnimation(fig, anim, frames=paas.shape[0],
                                          interval=0, blit=False, repeat=False)
            # ax3d(fig, ax)
            ax.legend()
            fig.tight_layout(pad=0)
            # ani.save('ani.mp4', dpi=200, fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

    if case == 50:
        """
        view a collection of rpo and ppo
        """
        N = 64
        L = 22
        ks = pyKS(N, L)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        poIds = [[1]+range(1, 10), range(1, 10)]
        aas = loadPO('../../data/ks22h001t120x64EV.h5', poIds)

        ii = [1, 2, 3]
        # ii = [7, 8, 11]

        doProj = False
        if doProj:
            pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
            reqr = reqr.dot(bases.T)
            eqr = eqr.dot(bases.T)
            paas = []
            for i in range(len(aas)):
                paas.append(aas[i].dot(bases.T) - eqr[1])
            reqr -= eqr[1]
            eqr -= eqr[1]

            ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        plotRE(ax, reqr, eqr, ii)
        if doProj:
            for i in range(len(aas)):
                ax.plot(paas[i][:, ii[0]], paas[i][:, ii[1]], paas[i][:, ii[2]],
                        alpha=0.2)
            ax.plot(paas[0][:, ii[0]], paas[0][:, ii[1]], paas[0][:, ii[2]], c='k',
                    label=r'$rpo_{16.31}$')
        else:
            for i in range(len(aas)):
                ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], aas[i][:, ii[2]],
                        alpha=0.2)
        ax3d(fig, ax)


    if case == 60:
        """
        view rpo/ppo pair one at a time
        """
        N = 64
        L = 22
        h = 0.001
        ks = pyKS(N, h, L)

        for i in range(1, 20):
            req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)

            poIds = [[1] + range(i, i+1), range(i, i+1)]
            aas = loadPO('../../data/ks22h001t120x64EV.h5', poIds)

            ii = [0, 3, 4]

            doProj = True
            if doProj:
                pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])
                # pev, bases = getBases(ks, 'eq', eq[0], [2, 3, 5])
                # pev, bases = getBases(ks, 'req', req[0], [0, 1, 3], ws[0])
                reqr = reqr.dot(bases.T)
                eqr = eqr.dot(bases.T)
                paas = []
                for i in range(len(aas)):
                    paas.append(aas[i].dot(bases.T) - eqr[1])
                reqr -= eqr[1]
                eqr -= eqr[1]

                ii = [0, 1, 2]

            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            plotRE(ax, reqr, eqr, ii)
            if doProj:
                for i in range(1, len(aas)):
                    ax.plot(paas[i][:, ii[0]], paas[i][:, ii[1]],
                            paas[i][:, ii[2]],
                            alpha=0.8)
                ax.plot(paas[0][:, ii[0]], paas[0][:, ii[1]], paas[0][:, ii[2]],
                        c='k', ls='--',
                        label=r'$rpo_{16.31}$')
            else:
                for i in range(len(aas)):
                    ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], aas[i][:, ii[2]],
                            alpha=0.7)
            ax3d(fig, ax, doBlock=True)

    if case == 70:
        """
        construct poincare section in ergodic trajectory
        """
        N = 64
        d = 22
        h = 0.001
        ks = pyKS(N, h, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        x0 = eqr[1]

        paas, poinc, poincf = ergoPoinc(ks, bases, x0,  2*np.pi/6, 'n')
        eqr -= eqr[1]

        ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        plotRE(ax, reqr, eqr, ii)
        ax.plot(paas[:, ii[0]], paas[:, ii[1]],
                paas[:, ii[2]], alpha=0.5)
        ax.scatter(poincf[:, 0], poincf[:, 1], poincf[:, 2])
        ax3d(fig, ax)

        scatter2dfig(poinc[:, 1], poinc[:, 2], ratio='equal')

    if case == 80:
        """
        construct poincare section with po
        """
        N = 64
        d = 22
        h = 0.001
        ks = pyKS(N, h, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        x0 = eqr[1]

        i = 40
        poIds = [range(1, i+1), range(1, i+1)]
        # poIds = [[], [2, 4, 8]]
        aas, poinc, nums = poPoinc('../../data/ks22h001t120x64EV.h5', poIds,
                                   bases, x0,  0.5 * np.pi/6, 'p')
        eqr -= eqr[1]

        ii = [0, 1, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        plotRE(ax, reqr, eqr, ii)
        for i in range(1, len(aas)):
            ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]],
                    aas[i][:, ii[2]],
                    alpha=0.2)
        ax.plot(aas[0][:, ii[0]], aas[0][:, ii[1]], aas[0][:, ii[2]],
                c='k', ls='--',
                label=r'$rpo_{16.31}$')
        ax.scatter(poinc[:, 0], poinc[:, 1], poinc[:, 2])
        ax3d(fig, ax)

        fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
        for i in range(1, len(aas)):
            ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]],
                    alpha=0.2)
        ax.plot(aas[0][:, ii[0]], aas[0][:, ii[1]],
                c='k', ls='--',
                label=r'$rpo_{16.31}$')
        ax.scatter(poinc[:, 0], poinc[:, 1])
        ax3d(fig, ax)

        scatter2dfig(poinc[:, 1], poinc[:, 2], ratio='equal')
        plot1dfig(nums)


    if case == 90:
        """
        construct poincare section in ergodic trajectory and
        try to  find the map
        """
        N = 64
        d = 22
        ks = pyKS(N, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        x0 = eqr[1].copy()
        eqr -= eqr[1]

        paas, poinc, poincf, poincRaw = ergoPoinc2(ks, bases, x0,
                                                   2*np.pi/6, 2.0/3*np.pi/6)

        ii = [0, 1, 2]

        fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
        plotRE2d(ax, reqr, eqr, ii)
        ax.plot(paas[:, ii[0]], paas[:, ii[1]], c='b', alpha=0.5)
        ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
        ax2d(fig, ax)

        fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                       ratio='equal')
        ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
        ax2d(fig, ax)

        plt.hold('on')
        for i in range(40):
            print i
            ax.scatter(poinc[i, 1], poinc[i, 2], c='g', s=20)
            plt.savefig(str(i))

    if case == 100:
        """
        New version to get Poincare points from pos
        """
        N = 64
        d = 22
        ks = pyKS(N, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        x0 = eqr[1].copy()
        eqr -= eqr[1]

        i = 100
        poIds = [range(1, i+1), range(1, i+1)]
        aas, poinc, poincf, poincRaw, nums = poPoinc(
            '../../data/ks22h001t120x64EV.h5',
            poIds, bases, x0,  2*np.pi/6, 2.0/3*np.pi/6)
        ii = [0, 1, 2]

        fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
        plotRE2d(ax, reqr, eqr, ii)
        for i in range(len(aas)):
            ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], c='gray', alpha=0.2)
        ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
        ax2d(fig, ax)

        fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                       ratio='equal')
        ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
        ax2d(fig, ax)

        plot1dfig(nums)

    if case == 110:
        """
        Get the return map from the Poincare section points
        """
        N = 64
        d = 22
        h = 0.001
        ks = pyKS(N, d)

        req, ws, reqr, eq, eqr = loadRE('../../data/ks22Reqx64.h5', N)
        pev, bases = getBases(ks, 'eq', eq[1], [6, 7, 10])

        reqr = reqr.dot(bases.T)
        eqr = eqr.dot(bases.T)
        reqr -= eqr[1]
        x0 = eqr[1].copy()
        eqr -= eqr[1]

        i = 100
        poIds = [range(1, i+1), range(1, i+1)]
        aas, poinc, poincf, poincRaw, nums = poPoinc(
            '../../data/ks22h001t120x64EV.h5', poIds,
            bases, x0,  2*np.pi/6, 2.0/3*np.pi/6)
        ii = [0, 1, 2]

        fig, ax = pl2d(labs=[r'$v_1$', r'$v_2$'])
        plotRE2d(ax, reqr, eqr, ii)
        for i in range(len(aas)):
            ax.plot(aas[i][:, ii[0]], aas[i][:, ii[1]], c='gray', alpha=0.2)
        ax.scatter(poincf[:, 0], poincf[:, 1], c='r', edgecolors='none')
        ax2d(fig, ax)

        fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                       ratio='equal')
        ax.scatter(poinc[:, 1], poinc[:, 2], c='r', edgecolors='none')
        ax2d(fig, ax)

        plot1dfig(nums)

        xf = poinc[:, 1:]
        sel = xf[:, 0] > 0
        # xf = xf[sel]
        # poincRaw = poincRaw[sel]
        scale = 10
        nps = 5000
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                       "gamma": np.logspace(-2, 2, 5),
                                       "degree": [3]})

        svr.fit(xf[:, 0:1], xf[:, 1]*scale)
        xp = linspace(0.43, -0.3, nps)  # start form right side
        xpp = xp.reshape(nps, 1)
        yp = svr.predict(xpp)/scale
        fig, ax = pl2d(size=[8, 3], labs=[None, 'z'], axisLabelSize=20,
                       ratio='equal')
        ax.scatter(poinc[:, 1], poinc[:, 2], c='r', s=10, edgecolors='none')
        ax.plot(xp, yp, c='g', ls='-', lw=2)
        ax2d(fig, ax)

        curve = np.zeros((nps, 2))
        curve[:, 0] = xp
        curve[:, 1] = yp
        minIds, minDs = getCurveIndex(xf, curve)
        sortId = np.argsort(minIds)

        dis, coor = getCurveCoordinate(sortId, poincRaw)
        fig, ax = pl2d(size=[6, 4], labs=[r'$S_n$', r'$S_{n+1}$'],
                       axisLabelSize=15)
        ax.scatter(coor[:-1], coor[1:], c='r', s=10, edgecolors='none')
        ax2d(fig, ax)

