from personalFunctions import *
from py_ks import *
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from personalPlotly import *


class KSReq():
    """
    This class is designed to study the structure of KS
    system in the symmetry reduced fundamental state space.
    
    Slice is chosen to make the pth mode's real part zero.
    Note, for 1st mode slice, E2 and E3 are on the slice border,
    it makes no sence to reduce symmetry for them, nor for the projection of
    vectors.
    """
    
    def __init__(self, N, L, fileName, poFile, p):
        """
        N : number of Fourier modes in KS
        L : domain size of KS
        poFile: the file storing ppo/rpo and Floquet vectors
        p : mode used to reduce SO(2)
        -------
        Eq : eq and req in the full state sapce
        ws : phase shift of req
        Eqr : eq and req in the fundamental domain
        Eg : group orbits of E2 and E3
        Es : eigenvalues of eq/req
        Ev : eigenvector in the full state space
        Evr : eigvenctor projected onto slice
        EqrP : projected Eqr
        EgP : projected Eg
        """
        self.N = N
        self.L = L
        self.fileName = fileName
        self.poFile = poFile
        self.p = p
        self.ks = pyKS(N, L)
        
        self.req, self.ws, self.reqr, self.eq, self.eqr = self.loadRE(fileName, p)
        self.Eq = {'e': self.eq, 'tw': self.req}
        self.Eqr = {'e': self.eqr, 'tw': self.reqr}

        self.Eg = self.EqGroupOrbit()
        self.Es, self.Ev, self.Evr = self.calEsEv(p)

        self.EqrP = {'e': np.zeros([3, 3]), 'tw': np.zeros([2, 3])}
        self.EgP = None

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
        -----
        return:
        Es : eigenvalues
        Ev : eigenvectors in the full state space
        Evr: eigenvectors projected in the slice
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
        
    def loadPO(self, poIds, p, bases=None, x0=None):
        """
        Load rpo and ppo and reduce the symmetry.
        If bases and orgin x0 are given, then also return the
        projection.
        
        ----------
        Parameters
        p : the mode used to reduce symmetry
        ----------
        Return :
        aars : symmetry reduced state space
        aaps : projected state space
        dom  : domain info
        jumps : jumps times
        """
        aars = []
        aaps = []
        dom = []
        jumps = []
        types = ['rpo', 'ppo']
        for i in range(2):
            poType = types[i]
            for poId in poIds[i]:
                a0, T, nstp, r, s = KSreadPO(self.poFile, poType, poId)
                h = T / nstp
                aa = self.ks.intg(a0, h, nstp, 5)
                aar, dids, ths = self.ks.redO2f(aa, p)
                aars.append(aar)
                dom.append(dids)
                jumps.append(getJumpPts(dids))
                if bases is not None:
                    aaps.append(aar.dot(bases.T) - x0)

        if bases is not None:
            return aars, aaps, dom, jumps
        else:
            return aars, dom, jumps

    def poFvPoinc(self, poType, poId, ptsId, p, ii, reflect=False):
        """
        get the in-slice Floquet vector
        """
        a0, T, nstp, r, s = KSreadPO(self.poFile, poType, poId)
        h = T / nstp
        aa = self.ks.intg(a0, h, nstp, 5)
        fe = KSreadFE(self.poFile, poType, poId)
        fv = KSreadFV(self.poFile, poType, poId)[ptsId].reshape(30, self.N-2)

        x0 = aa[ptsId]
        if reflect:
            x0 = self.ks.Reflection(x0)
            fv = self.ks.Reflection(fv)

        rfv = self.ks.redV(fv, x0, p, True)
        
        v1, v2, v3 = orthAxes(rfv[ii[0]], rfv[ii[1]], rfv[ii[2]])
        bases = np.vstack((v1, v2, v3))

        return fe, fv, rfv, bases
        
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

    def getMuEq(self, etype, eId, vId, p, T=200, r0=1e-5, nn=30):
        """
        obtain the unstable manifold of eq/req
        """
        if etype == 'e':
            a0 = self.eqr[eId]
            v0 = self.Ev['e'][eId][vId]  # Ev not Evr
        elif etype == 'tw':
            a0 = self.reqr[eId]
            v0 = self.Ev['tw'][eId][vId]
        
        return self.getMu(a0, v0, p, T=T, r0=r0, nn=nn)

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

    def plotRE(self, ax, ii, doProject=False, do3d=True):
        """
        construct the plotting block for req/eq
        """
        if doProject:
            x = self.EqrP['tw']
            y = self.EqrP['e']
        else:
            x = self.Eqr['tw']
            y = self.Eqr['e']

        c1 = ['r', 'b']
        for i in range(2):
            if do3d:
                ax.scatter(x[i, ii[0]], x[i, ii[1]], x[i, ii[2]],
                           c=c1[i], s=70,
                           edgecolors='none', label='TW'+str(i+1))
            else:
                ax.scatter(x[i, ii[0]], x[i, ii[1]], c=c1[i], s=70,
                           edgecolors='none', label='TW'+str(i+1))

        c2 = ['c', 'k', 'y']
        for i in range(3):
            if do3d:
                ax.scatter(y[i, ii[0]], y[i, ii[1]], y[i, ii[2]],
                           c=c2[i], s=70,
                           edgecolors='none', label='E'+str(i+1))
            else:
                ax.scatter(y[i, ii[0]], y[i, ii[1]], c=c2[i], s=70,
                           edgecolors='none', label='E'+str(i+1))

    def plotFundOrbit(self, ax, faa, jumps, ii, c=None, alpha=0.5, lw=1,
                      label=None):
        """
        plot orbit in the fundamental domain. sudden jumps are avoided.
        
        faa : a single orbit in the fundamental domain
        jumps : indices where orbit jumps from one domain to another
        ii : the plot index
        """
        if c is None:
            c = rand(3, 1)

        x = concatenate(([-1], jumps, [len(faa)-1]))
        for i in range(len(x)-1):
            r = range(x[i]+1, x[i+1]+1)
            if i == 0:
                ax.plot(faa[r, ii[0]], faa[r, ii[1]], faa[r, ii[2]],
                        c=c, alpha=alpha, lw=lw, label=label)
            else:
                ax.plot(faa[r, ii[0]], faa[r, ii[1]], faa[r, ii[2]],
                        c=c, alpha=alpha, lw=lw)

    def getBases(self, etype, eId, ii):
        """
        get projection bases

        x0 : origin
        bases : selected bases
        """
        v = self.Evr[etype][eId]
        v1, v2, v3 = orthAxes(v[ii[0]], v[ii[1]], v[ii[2]])
        bases = np.vstack((v1, v2, v3))

        x0 = self.Eqr[etype][eId].dot(bases.T)

        self.EqrP['e'] = self.Eqr['e'].dot(bases.T) - x0
        self.EqrP['tw'] = self.Eqr['tw'].dot(bases.T) - x0
        
        E2 = self.Eg['E2'].dot(bases.T) - x0
        E3 = self.Eg['E3'].dot(bases.T) - x0
        self.EgP = {'E2': E2, 'E3': E3}

        return x0, bases

    def getPoinc(self, raa, dids, jumps):
        """
        get the poincare intersection points. Poincare section is
        b_2 = 0 from negative to positive. Note it is very important to
        record the point after crossing Poincare section. Also both
        crossings are recorded due to reflection symmetry, but the
        order should be shuffled.
        --------
        Paramters
        raa : orbit in the fundamental domain
        dids : indices of regions
        jumps : indices when orbit crosses Poincare section
        """
        x = dids[jumps]
        y = dids[jumps+1]
        x1 = jumps[x == 2]
        y1 = jumps[y == 1]
        assert np.array_equal(x1, y1)
        x2 = jumps[x == 1]
        y2 = jumps[y == 2]
        assert np.array_equal(x2, y2)

        pcN = len(x1)           # positive crossing number
        ncN = len(x2)
        assert pcN + ncN == len(jumps)
        
        borderIds = np.append(x1+1, x2+1)
        borderPts = raa[borderIds]
        
        return borderIds, borderPts, pcN, ncN

##############################################################################################################

if __name__ == '__main__':

    N = 64
    L = 22
    ksreq = KSReq(N, L, '../../data/ks22Reqx64.h5',
                  '../../data/ks22h001t120x64EV.h5', 1)

    case = 20

    if case == 10:
        """
        view the unstable manifold of eq/req
        """
        nn = 30
        aas, dom, jumps = ksreq.getMuEq('tw', eId=0, vId=0, p=1, nn=nn, T=100)
        ii = [1, 5, 3]
        spt = aas
        E2, E3 = ksreq.Eg['E2'], ksreq.Eg['E3']

        doProj = False
        if doProj:
            x0, bases = ksreq.getBases('tw', 0, [0, 1, 3])
            aap = []
            for i in range(len(aas)):
                aap.append(aas[i].dot(bases.T) - x0)
            ii = [0, 1, 2]
            spt = aap
            E2, E3 = ksreq.EgP['E2'], ksreq.EgP['E3']
            
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ksreq.plotRE(ax, ii, doProject=doProj)
        for i in range(nn):
            ksreq.plotFundOrbit(ax, spt[i], jumps[i], ii)
        ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]])
        ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
        ax3d(fig, ax)

    if case == 14:
        """
        save the symbolic dynamcis
        """
        data = np.load('PoPoincare.npz')
        borderPtsP = data['borderPtsP']
        pcN = data['pcN']
        ncN = data['ncN']
        M = len(borderPtsP)

        ii = [0, 1, 2]
        for k in range(100, 150):
            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            ax.scatter(0, 0, 0, c='m', edgecolors='none', s=100, marker='^')
            for i in range(M):
                x, y, z = [borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                           borderPtsP[i][:, ii[2]]]
                ax.scatter(x, y, z, c=[0, 0, 0, 0.2], marker='o', s=10,
                           edgecolors='none')
            x, y, z = [borderPtsP[k][:, ii[0]], borderPtsP[k][:, ii[1]],
                       borderPtsP[k][:, ii[2]]]
            for i in range(pcN[k]+ncN[k]):
                ax.scatter(x[i], y[i], z[i], c='r' if i < pcN[k] else 'b',
                           marker='o', s=70, edgecolors='none')
                ax.text(x[i], y[i], z[i], str(i+1), fontsize=20, color='g')
            s = 'ppo'+str(k+1-100)
            ax3d(fig, ax, angle=[40, -120], save=True, name=s+'.png', title=s)
        
    if case == 15:
        """
        visulaize the symmbolic dynamcis
        """
        data = np.load('PoPoincare.npz')
        borderPtsP = data['borderPtsP']
        pcN = data['pcN']
        ncN = data['ncN']
        M = len(borderPtsP)

        ii = [0, 1, 2]
        fig, ax = pl3d(size=[12, 10], labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        for i in range(M):
            x, y, z = [borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                       borderPtsP[i][:, ii[2]]]
            ax.scatter(x, y, z, c=[0, 0, 0, 0.2], marker='o', s=10,
                       edgecolors='none')
        ax3d(fig, ax)
        for i in range(100, 110):
            x, y, z = [borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                       borderPtsP[i][:, ii[2]]]
            add3d(fig, ax, x, y, z, maxShow=20)

    if case == 16:
        """
        plot 3 cases together
        """
        trace = []
        trace.append(ptlyTrace3d([0], [0], [0], plotType=1, ms=7, mc='black'))
        
        ii = [0, 1, 2]
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        if True:
            borderPtsP = np.load('ErgodicPoincare.npy')
            ax.scatter(borderPtsP[:, ii[0]], borderPtsP[:, ii[1]],
                       borderPtsP[:, ii[2]], c='r', s=20,
                       edgecolors='none')
            trace.append(ptlyTrace3d(borderPtsP[:, ii[0]],
                                     borderPtsP[:, ii[1]],
                                     borderPtsP[:, ii[2]],
                                     plotType=1, ms=2, mc='red'))
        if True:
            borderPtsP = np.load('E2Poincare.npy')
            M = borderPtsP.shape[0]
            for i in range(M):
                ax.scatter(borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                           borderPtsP[i][:, ii[2]], c='b', marker='s',
                           s=20, edgecolors='none')
            for i in range(M):
                trace.append(ptlyTrace3d(borderPtsP[i][:, ii[0]],
                                         borderPtsP[i][:, ii[1]],
                                         borderPtsP[i][:, ii[2]],
                                         plotType=1, ms=2, mc='blue'))
        if True:
            borderPtsP = np.load('PoPoincare.npy')
            M = borderPtsP.shape[0]
            for i in range(M):
                ax.scatter(borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                           borderPtsP[i][:, ii[2]], c='g', marker='o',
                           s=20, edgecolors='none')
            for i in range(M):
                trace.append(ptlyTrace3d(borderPtsP[i][:, ii[0]],
                                         borderPtsP[i][:, ii[1]],
                                         borderPtsP[i][:, ii[2]],
                                         plotType=1, ms=2, mc='green'))
        ax3d(fig, ax)
        
        if True:
            ptly3d(trace, 'mixPoincare', off=True,
                   labs=['$v_1$', '$v_2$', '$v_3$'])

    if case == 17:
        """
        use the poincare section b2=0 and b2 from negative to positive for
        unstable manifold of E2.
        Note, the unstable manifold lives in the invariant subspace,
        which is also the poincare section border.
        """
        nn = 50
        pos, poDom, poJumps = ksreq.getMuEq('e', eId=1, vId=0, p=1, nn=nn,
                                            T=100)
        M = len(pos)
        borderPts = []
        for i in range(M):
            borderPts.append(pos[i][::100, :])

        data = np.load('bases.npz')
        Ori = data['Ori']
        bases = data['bases']
        borderPtsP = []
        for i in range(M):
            p = (borderPts[i]-Ori).dot(bases.T)
            borderPtsP.append(p)
           
        ii = [0, 1, 2]
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        for i in range(M):
            ax.scatter(borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                       borderPtsP[i][:, ii[2]], c='b' if i < 50 else 'r',
                       marker='o' if i < 50 else 's',
                       s=20, edgecolors='none')
        ax3d(fig, ax)
        
        if False:
            trace = []
            trace.append(ptlyTrace3d(0, 0, 0, plotType=1, ms=7, mc='black'))
            for i in range(M):
                trace.append(ptlyTrace3d(borderPtsP[i][:, ii[0]],
                                         borderPtsP[i][:, ii[1]],
                                         borderPtsP[i][:, ii[2]],
                                         plotType=1, ms=2, mc='red'))
            ptly3d(trace, 'E2Poincare', labs=['$v_1$', '$v_2$', '$v_3$'])
        
        spt = pos
        ii = [1, 5, 3]
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ksreq.plotRE(ax, ii)
        for i in range(nn):
            ksreq.plotFundOrbit(ax, spt[i], poJumps[i], ii)
        ax3d(fig, ax)

        # np.save('E2Poincare', borderPtsP)

    if case == 18:
        """
        use the poincare section b2=0 and b2 from negative to positive for
        ergodic trajectories
        """
        a0 = rand(N-2)
        aa = ksreq.ks.aintg(a0, 0.01, 100, 1)
        aa = ksreq.ks.aintg(aa[-1], 0.01, 6000, 1)
        raa, dids, ths = ksreq.ks.redO2f(aa, 1)
        jumps = getJumpPts(dids)

        borderIds, borderPts, pcN, ncN = ksreq.getPoinc(raa, dids, jumps)

        data = np.load('bases.npz')
        Ori = data['Ori']
        bases = data['bases']
        borderPtsP = (borderPts - Ori).dot(bases.T)

        ii = [0, 1, 2]
        x, y, z = [borderPtsP[:, ii[0]], borderPtsP[:, ii[1]],
                   borderPtsP[:, ii[2]]]
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        ax.scatter(x, y, z, c='y', s=20, edgecolors='none')
        ax3d(fig, ax)
        add3d(fig, ax, x, y, z, maxShow=20)

        if False:
            trace = []
            trace.append(ptlyTrace3d(0, 0, 0, plotType=1, ms=7, mc='black'))
            trace.append(ptlyTrace3d(borderPtsP[:, ii[0]],
                                     borderPtsP[:, ii[1]],
                                     borderPtsP[:, ii[2]],
                                     plotType=1, ms=2, mc='red'))
            ptly3d(trace, 'ErgodicPoincare', labs=['$v_1$', '$v_2$', '$v_3$'])

        # np.save('ErgodicPoincare', borderPtsP)
        
    if case == 19:
        """
        use the poincare section b2=0 and b2 from negative to positive
        """
        poIds = [range(1, 101), range(1, 101)]
        pos, poDom, poJumps = ksreq.loadPO(poIds, 1)
        M = len(pos)
        borderIds = []
        borderPts = []
        pcN = []
        ncN = []
        borderNum = 0
        for i in range(M):
            ids, pts, pcn, ncn = ksreq.getPoinc(pos[i], poDom[i], poJumps[i])
            borderIds.append(ids)
            borderPts.append(pts)
            pcN.append(pcn)
            ncN.append(ncn)
            borderNum += len(ids)

        ptsId = borderIds[0][1]
        fe, fv, rfv, bases = ksreq.poFvPoinc('rpo', 1, ptsId, 1, [0, 3, 4],
                                             reflect=True)
        Ori = borderPts[0][1]
        borderPtsP = []
        for i in range(M):
            p = (borderPts[i]-Ori).dot(bases.T)
            borderPtsP.append(p)

        ii = [0, 1, 2]
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        for i in range(M):
            x, y, z = [borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                       borderPtsP[i][:, ii[2]]]
            ax.scatter(x, y, z, c='y' if i < 50 else 'y',
                       marker='o' if i < 50 else 's',
                       s=20, edgecolors='none')
        ax3d(fig, ax)
        i = 1
        x, y, z = [borderPtsP[i][:, ii[0]], borderPtsP[i][:, ii[1]],
                   borderPtsP[i][:, ii[2]]]
        add3d(fig, ax, x, y, z)
        
        if False:
            trace = []
            trace.append(ptlyTrace3d(0, 0, 0, plotType=1, ms=7, mc='black'))
            for i in range(M):
                trace.append(ptlyTrace3d(borderPtsP[i][:, ii[0]],
                                         borderPtsP[i][:, ii[1]],
                                         borderPtsP[i][:, ii[2]],
                                         plotType=1, ms=2, mc='red',
                                         mt='circle' if i < 50 else 'square'))
                ptly3d(trace, 'PoPoincare', labs=['$v_1$', '$v_2$', '$v_3$'])

        # np.savez_compressed('bases', Ori=Ori, bases=bases)
        # np.savez_compressed('PoPoincare', borderPtsP=borderPtsP,
        #                     pcN=pcN, ncN=ncN)
        if False:
            ii = [2, 6, 4]
            # ii = [1, 5, 3]
            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            for i in range(len(pos)):
                ksreq.plotFundOrbit(ax, pos[i], poJumps[i], ii, alpha=0.5)
                ax.scatter(borderPts[i][:, ii[0]], borderPts[i][:, ii[1]],
                           borderPts[i][:, ii[2]], c='k')
            ax3d(fig, ax)

    if case == 20:
        """
        view the unstable manifold of a single eq/req and one po/rpo
        This is done in Fourier projection space
        """
        nn = 30
        aas, dom, jumps = ksreq.getMuEq('e', eId=1, vId=0, p=1, nn=nn, T=100)

        poIds = [[], [1, 12]]
        pos, poDom, poJumps = ksreq.loadPO(poIds, 1)
        
        ii = [1, 5, 3]
        # ii = [2, 6, 4]
        spt = aas
        sptPo = pos
        E2, E3 = ksreq.Eg['E2'], ksreq.Eg['E3']

        doProj = False
        if doProj:
            x0, bases = ksreq.getBases('tw', 0, [0, 1, 3])
            aap = []
            for i in range(len(aas)):
                aap.append(aas[i].dot(bases.T) - x0)
            pop = []
            for i in range(len(pos)):
                pop.append(pos[i].dot(bases.T) - x0)
            ii = [0, 1, 2]
            spt = aap
            sptPo = pop
            E2, E3 = ksreq.EgP['E2'], ksreq.EgP['E3']

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ksreq.plotRE(ax, ii, doProject=doProj)
        for i in range(len(spt)):
            ksreq.plotFundOrbit(ax, spt[i], jumps[i], ii)
        cs = ['r', 'b']
        for i in range(len(sptPo)):
            ksreq.plotFundOrbit(ax, sptPo[i], poJumps[i], ii,
                                c='k' if i > 1 else cs[i],
                                alpha=1, lw=1.5)
        #ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]], c='c')
        #ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]], c='c')
        ax3d(fig, ax)

    if case == 21:
        """
        same with 20 but to save figures
        """
        nn = 30
        aas, dom, jumps = ksreq.getMuEq('e', eId=1, vId=0, p=1, nn=nn, T=100)

        for k in range(1, 51):
            poIds = [[], [2, k]]
            pos, poDom, poJumps = ksreq.loadPO('../../data/ks22h001t120x64EV.h5',
                                               poIds, 1)

            ii = [1, 5, 3]
            spt = aas
            sptPo = pos
            E2, E3 = ksreq.Eg['E2'], ksreq.Eg['E3']

            doProj = False
            if doProj:
                x0, bases = ksreq.getBases('tw', 0, [0, 1, 3])
                aap = []
                for i in range(len(aas)):
                    aap.append(aas[i].dot(bases.T) - x0)
                pop = []
                for i in range(len(pos)):
                    pop.append(pos[i].dot(bases.T) - x0)
                ii = [0, 1, 2]
                spt = aap
                sptPo = pop
                E2, E3 = ksreq.EgP['E2'], ksreq.EgP['E3']

            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            ksreq.plotRE(ax, ii, doProject=doProj)
            for i in range(len(spt)):
                ksreq.plotFundOrbit(ax, spt[i], jumps[i], ii)
            for i in range(len(sptPo)):
                ksreq.plotFundOrbit(ax, sptPo[i], poJumps[i], ii,
                                    c='k' if i > 0 else 'r',
                                    alpha=1, lw=1.5,
                                    label='ppo'+str(poIds[1][i]))
            ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]])
            ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
            ax3d(fig, ax, save=True, name='rpo'+str(k)+'.png')

    if case == 25:
        """
        view 2 shadowing orbits
        """
        poIds = [[1, 26], range(4, 4)]
        pos, poDom, poJumps = ksreq.loadPO('../../data/ks22h001t120x64EV.h5',
                                           poIds, 1)
        ii = [1, 3, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        for i in range(len(pos)):
            ksreq.plotFundOrbit(ax, pos[i], poJumps[i], ii,
                                c='k' if i > 0 else 'r',
                                alpha=1, lw=1.5)
        ax3d(fig, ax)
        
    if case == 30:
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

