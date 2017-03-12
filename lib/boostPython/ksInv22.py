from ksHelp import *
from py_ks import *
from personalFunctions import *


class Sym():
    """
    Parameters
    ==========
    N      : number of Fourier modes in KS
    L      : domain size of KS
    pSlice : Fourier mode index used to define slice
    pFund  : Fourier mode index used to define fundamental domain
    toY    : whether rotate to the positive Y axis
    """
    N, L = 64, 22
    ks = pyKS(N, L)
    ksp = KSplot(ks)

    def __init__(self, pSlice, toY, pFund):
        self.pSlice = pSlice
        self.pFund = pFund
        self.toY = toY

    def getMuFullState(self, x0, v0, v1=None, T=200, r0=1e-5, nn=30, b=1):
        """
        get the unstable mainifold in the full state space
        
        case 1:
        -------
        For a complex pair (the case of E1 and E2), we use distribution
        a0 = x0 + r0 * exp(s) * v0
        Here, s = np.arange(0, nn, b). For real case, b = 1. For complex case,
        b = 2 * pi * |mu / nv| with (mu + i * nv) the stability exponent.
        
        case 2:
        -------
        For two degerate real unstable directions (the case of E3), we use distribution
        a0 = x0 + r0 * (v1*cos + v2*sin)
        Here, we v1 and v2 are orthornormal.
        
        ---------
        return :
        aars : orbits in reduced fundamental domain
        dom : domain indices
        jumps : jumping indices
        """
        aars = []
        dom = []
        jumps = []

        
        if v1 is not None:
            v0 = v0 / LA.norm(v0)
            v1p = v1 - np.dot(v0, v1) * v0
            v1p = v1p / LA.norm(v1p)

        for i in range(nn):
            if v1 is None:      # case 1
                s = np.double(i) / nn * b
                a0 = x0 + r0 * np.exp(s) * v0
            else:               # case 2
                th = i * 2 * np.pi / nn 
                a0 = x0 + r0 * (np.cos(th) * v0 + np.cos(th) * v1p)
                
            aa = self.ks.aintg(a0, 0.01, T, 1)
            raa, dids, ths = self.ks.redO2f(aa, self.pSlice, self.pFund)
            aars.append(raa)
            dom.append(dids)
            jumps.append(getJumpPts(dids))
        
        return aars, dom, jumps


class Eq(Sym):
    """
    equilibrium class
    """
    def __init__(self, a0, r, pSlice, toY, pFund, 
                 DoGetSliceState=True, DoCalculateEsEv=True, DoGetGroupOrbit=True):
        Sym.__init__(self, pSlice, toY, pFund)

        self.a0 = a0
        self.r = r              # error
        
        self.go = None          # group orbit
        self.so = None          # state in the slice
        self.es = None          # stability exponents
        self.ev = None          # stability eigenvectors
        self.evr = None         # stability eigenvectors in real format
        self.rev = None         # in-slice stability eigenvectors
       
        # ev is in cloumn format. While evr and rev are in row format

        if DoGetSliceState:
            self.so = self.ks.redO2f(a0, pSlice, pFund)[0]

        # E2 and E3 may be in the slice border, so
        # just obtain their group orbits  
        if DoGetGroupOrbit:
            n = 100
            self.go = np.zeros((n, self.N-2))
            for i in range(n):
                th = 2*i*np.pi / n
                self.go[i] = self.ks.Rotation(a0, th)

        # get the eigenvector of eq/req
        if DoCalculateEsEv:
            self.es, self.ev = self.getEsEv()
            self.evr = Tcopy(realve(self.ev))
            self.rev = self.ks.redV(self.evr, self.a0, self.pSlice, True)

    def getEsEv(self):
        es, ev = self.ksp.stabEig(self.a0)
        return es, ev
            

    def getMu(self, vId, T=200, r0=1e-5, nn=30):
        """
        obtain the unstable manifold of eq

        Some properties:
        1) unstable manifold of E2 is in the antisymmetric subspace.
        2) When we check the the correctness of rev of E3. rev[2]
           does not vanish becuase E3 is in the slice border.
        """
        v0, v1 = self.evr[vId], self.evr[vId+1]  # default case 2
        b = 1
        if np.iscomplex(self.es[vId]):  # case 1
            v1 = None
            b = 2 * np.pi * np.abs(self.es[vId].real / self.es[vId].imag)
        return self.getMuFullState(self.a0, v0, v1, T=T, r0=r0, nn=nn, b=b)

    
class Req(Eq):
    """
    relative equilibrium class
    """
    def __init__(self, a0, w, r, pSlice, toY, pFund,
                 DoGetSliceState=True, DoCalculateEsEv=True, DoGetGroupOrbit=True):
        self.w = w
        Eq.__init__(self, a0, r, pSlice, toY, pFund, DoGetSliceState, DoCalculateEsEv, DoGetGroupOrbit)

    def getEsEv(self):
         es, ev = self.ksp.stabReqEig(self.a0, self.w)
         return es, ev

class PPO(Sym):
    """
   ppo class
    """
    def __init__(self, a0, T, nstp, r, pSlice, toY, pFund,
                 DoGetFullOrbit=True, DoGetInsliceOrbit=True,
                 DoGetFundOrbit=True):
        Sym.__init__(self, pSlice, toY, pFund)

        self.a0 = a0
        self.T = T
        self.nstp = nstp
        self.r = r              # error

        self.fo = None          # full state space orbit
        self.so = None          # slice orbit
        self.fdo = None         # fundamental domain orbit
        self.po = None          # projected orbit

        self.pp = None          # Poincare point
        self.ppp = None         # Projected Poincare point

        self.dids = None        # domain ids
        self.jumps = None       # domain jump index
        self.ths = None         # slice angles
        
        ##########
        if DoGetFullOrbit:
            h = T / nstp
            self.fo = self.ks.intg(a0, h, nstp, 5)
            if DoGetInsliceOrbit:
                self.so, self.ths = self.ks.redSO2(self.fo, pSlice, toY)
                if DoGetFundOrbit:
                    self.fdo, self.dids = self.ks.fundDomain(self.so, pFund)
                    self.jumps = getJumpPts(self.dids)

        

class RPO(PPO):
    """
    rpo class
    """
    def __init__(self, a0, T, nstp, r, s, pSlice, toY, pFund,
                 DoGetFullOrbit=True, DoGetInsliceOrbit=True,
                 DoGetFundOrbit=True):
        PPO.__init__(self, a0, T, nstp, r, pSlice, toY, pFund, 
                     DoGetFullOrbit, DoGetInsliceOrbit, DoGetFundOrbit)
        self.s = s              # shift
        
    

class Inv(Sym):
    """
    This class is designed to study the structure of KS
    system in the symmetry reduced fundamental state space.
    "Inv" means invariant structures in this system.
    
    Slice is chosen to make the pth mode's real part zero.
    Note, for 1st mode slice, E2 and E3 are on the slice border,
    it makes no sence to reduce symmetry for them, nor for the projection of
    vectors.
    """
    
    def __init__(self, reqFile, poFile, pSlice, toY, pFund, ppoIds=[], rpoIds=[]):
        """
        reqFile : the file storing reqs
        poFile: the file storing ppo/rpo and Floquet vectors
        -------
        Es      : the 3 equilibria 
        TWs     : the 2 relative equilibria
        ppos    : a list of ppo
        rpos    : a list of rpo
        ppoIds  : indices of ppo
        rpoIds  : indices of rpo
        """
        Sym.__init__(self, pSlice, toY, pFund)
        self.reqFile = reqFile
        self.poFile = poFile
        self.ppoIds = ppoIds
        self.rpoIds = rpoIds

        self.Es, self.TWs, self.ppos, self.rpos = [], [], [], []

        self.loadRE()
        self.loadPO()

    def loadRE(self):
        """
        load all req and eq and their corresponding
        symmetry reduced states in the fundamental
        domain.

        reqFile : path to the req data file
        p : the fourier mode used to reduce symmetry
        """
        
        # load req
        for i in range(2):
            a, w, err = self.ksp.readReq(self.reqFile, i+1)
            self.TWs.append(Req(a, w, err, self.pSlice, self.toY, self.pFund, self.ksp))
        
        # load eq
        for i in range(3):
            a, err = self.ksp.readEq(self.reqFile, i+1)
            self.Es.append(Eq(a, err, self.pSlice, self.toY, self.pFund, self.ksp))

        
    def loadPO(self):
        """
        Load rpo and ppo and reduce the symmetry.
        If bases and orgin x0 are given, then also return the
        projection.
        
        This functions is called in the constructor, but it can also 
        be called afterwards so as to refresh the rpo/ppo set        
        """
        self.ppos, self.rpos = [], []
        for i in self.ppoIds:
            a0, T, nstp, r, s = self.ksp.readPO(self.poFile, 'ppo', i)
            self.ppos.append(PPO(a0, T, nstp, r, self.pSlice, self.toY, self.pFund))
             
        for i in self.rpoIds:
            a0, T, nstp, r, s =  self.ksp.readPO(self.poFile, 'rpo', i)
            self.rpos.append(RPO(a0, T, nstp, r, s, self.pSlice, self.toY, self.pFund))


    def poPoincProject(self, bases, x0):
        for i in range(len(self.ppos)):
            self.ppos[i].ppp = (self.ppos[i].pp - x0).dot(bases.T)
        for i in range(len(self.rpos)):
            self.rpos[i].ppp = (self.rpos[i].pp - x0).dot(bases.T)

    def savePoPoincProject(self):
        ppo = []
        rpo = []
        for i in range(len(self.ppos)):
            ppo.append(self.ppos[i].ppp)
        for i in range(len(self.rpos)):
            rpo.append(self.rpos[i].ppp)
        np.savez_compressed('PoPoincare', ppo=ppo, rpo=rpo)

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
        c1 = ['r', 'b']
        for i in range(2):
            x = self.TWs[i].so
            if do3d:
                ax.scatter(x[ii[0]], x[ii[1]], x[ii[2]],
                           c=c1[i], s=70,
                           edgecolors='none', label='TW'+str(i+1))
            else:
                ax.scatter(x[ii[0]], x[ii[1]], c=c1[i], s=70,
                           edgecolors='none', label='TW'+str(i+1))

        c2 = ['c', 'k', 'y']
        for i in range(3):
            x = self.Es[i].so
            if do3d:
                ax.scatter(x[ii[0]], x[ii[1]], x[ii[2]],
                           c=c2[i], s=70,
                           edgecolors='none', label='E'+str(i+1))
            else:
                ax.scatter(x[ii[0]], x[ii[1]], c=c2[i], s=70,
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

        x = np.concatenate(([-1], jumps, [len(faa)-1]))
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

    def getPoinc(self, raa, jumps, first):
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
        
        Return
        borderIds : indices of intersection points which are in the 1st region
        borderPts : the corresponding points
        start : starting index from negative to positive
        """
        case = 1
        if case == 1:
            """
            Poincare section is the reflection border
            """
            n = len(jumps)
            borderPts = np.zeros((n, raa.shape[1]))
            # for i in range(n):
            #     j = jumps[i]
            #     if j < 0:
            #         # interp1d works for every row
            #         f = interp1d(raa[j-1:j+2, 2], raa[j-1:j+2].T, kind='quadratic')
            #     else:
            #         f = interp1d(raa[j:j+2, 2], raa[j:j+2].T, kind='linear')
            #     borderPts[i] = f(0)
            
            for i in range(n):
                j = jumps[i]
                x1, x2 = raa[j], raa[j+1]
                x1 = self.ks.Reflection(x1)
                p = np.vstack((x1, x2))
                f = interp1d(p[:, 2], p.T, kind='linear')
                borderPts[i] = f(0)

        if case == 2:
            """
            c_1 = 0.3 is the Poincare section
            """
            n = raa.shape[0]
            borderPts = np.zeros((0, raa.shape[1]))
            for i in range(n-1):
                if raa[i, 6] < 0 and raa[i+1, 6] > 0:
                    borderPts = np.vstack((borderPts, raa[i]))
            
        return borderPts
