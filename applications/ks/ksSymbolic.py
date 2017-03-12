from personalFunctions import *
from py_ks import *
from scipy.interpolate import interp1d
from personalPlotly import *
from ksInv22 import *


###############################################################################

if __name__ == '__main__':

    
    reqFile = '../../data/ks22Reqx64.h5'
    poFile = '../../data/ks22h001t120x64EV.h5'
    
    case = 10

    if case == 10:
        """
        view the unstable manifold of eq/req
        """
        inv = Inv(reqFile, poFile, 2, True, 3)
        nn = 30
        aas, dom, jumps = inv.Es[1].getMu(vId=0, nn=nn, T=100)
        ii = [3, 7, 11]
        E2, E3 = inv.Es[1].go, inv.Es[2].go
            
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        inv.plotRE(ax, ii)
        for i in range(nn):
            inv.plotFundOrbit(ax, aas[i], jumps[i], ii)
        # ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]])
        ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]])
        ax3d(fig, ax)

    if case == 20:
        """
        view the unstable manifold of a single eq/req and one po/rpo
        This is done in Fourier projection space
        """
        inv = Inv(reqFile, poFile, 2, True, 3, [], [1, 4])
        nn = 30
        aas, dom, jumps = inv.Es[1].getMu(vId=0, nn=nn, T=100)
        
        ii = [3, 7, 11]
        E2, E3 = inv.Es[1].go, inv.Es[2].go

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        inv.plotRE(ax, ii)
        for i in range(nn):
            inv.plotFundOrbit(ax, aas[i], jumps[i], ii)
        cs = ['r', 'b']
        for i in range(len(inv.rpos)):
            inv.plotFundOrbit(ax, inv.rpos[i].fdo, inv.rpos[i].jumps, ii,
                                c='k' if i > 0 else cs[i],
                                alpha=1, lw=1.5)
        #ax.plot(E2[:, ii[0]], E2[:, ii[1]], E2[:, ii[2]], c='c')
        ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]], c='b')
        ax3d(fig, ax, angle=[50, 80])

    if case == 30:
        """
        same with 20 but to save figures
        """
        inv = Inv(reqFile, poFile, 2, True, 3, [], [])
        nn = 30
        aas, dom, jumps = inv.Es[1].getMu(vId=0, nn=nn, T=100)

        ii = [3, 7, 11]
        E2, E3 = inv.Es[1].go, inv.Es[2].go


        for k in range(2, 51):
            inv.rpoIds = [1, k]
            inv.loadPO()

            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            inv.plotRE(ax, ii)
            for i in range(nn):
                inv.plotFundOrbit(ax, aas[i], jumps[i], ii)
            for i in range(len(inv.rpos)):
                inv.plotFundOrbit(ax, inv.rpos[i].fdo, inv.rpos[i].jumps, ii,
                                    c='k' if i > 0 else 'r',
                                    alpha=1, lw=1.5,
                                    label='rpo'+str(inv.rpoIds[i]))
            ax.plot(E3[:, ii[0]], E3[:, ii[1]], E3[:, ii[2]], c='b')
            ax3d(fig, ax, angle=[50, 80], save=True, name='rpo'+str(k)+'.png')

    if case == 14:
        """
        save the symbolic dynamcis
        """
        data = np.load('PoPoincare.npz')
        ppo = data['ppo']
        rpo = data['rpo']

        po = rpo
        poName = 'rpo'

        for i in range(len(po)):
            x, y, z = po[i][:, 0], po[i][:, 1], po[i][:, 2]
            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            ax.scatter(0, 0, 0, c='m', edgecolors='none', s=100, marker='^')
            for j in range(len(ppo)):
                p = ppo[j]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[0, 0, 0, 0.2],
                           marker='o', s=10, edgecolors='none')
            for j in range(len(rpo)):
                p = rpo[j]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[0, 0, 0, 0.2],
                           marker='o',
                           s=10, edgecolors='none')
            ax.scatter(x[::2], y[::2], z[::2], c='r', marker='o', s=40,
                       edgecolors='none')
            ax.scatter(x[1::2], y[1::2], z[1::2], c='b', marker='o', s=40,
                       edgecolors='none')
            for j in range(po[i].shape[0]):
                ax.text(x[j], y[j], z[j], str(j+1), fontsize=20, color='g')
                s = poName + str(i+1)
            # ax3d(fig, ax, angle=[20, -120], save=True, name=s+'.png', title=s)
            ax3d(fig, ax, angle=[30, 180], save=True, name=s+'.png', title=s)
            
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
        pos, poDom, poJumps = inv.getMuEq('e', eId=1, vId=0, p=1, nn=nn,
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
        inv.plotRE(ax, ii)
        for i in range(nn):
            inv.plotFundOrbit(ax, spt[i], poJumps[i], ii)
        ax3d(fig, ax)

        # np.save('E2Poincare', borderPtsP)

    if case == 18:
        """
        use the poincare section b2=0 and b2 from negative to positive for
        ergodic trajectories
        """
        a0 = rand(N-2)
        aa = inv.ks.aintg(a0, 0.01, 100, 1)
        aa = inv.ks.aintg(aa[-1], 0.01, 6000, 1)
        raa, dids, ths = inv.ks.redO2f(aa, 1)
        jumps = getJumpPts(dids)

        borderIds, borderPts, pcN, ncN = inv.getPoinc(raa, dids, jumps)

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
        inv.loadPO(1, ppoIds=range(1, 101), rpoIds=range(1, 101))
       
        # ptsId = inv.rpos[0].jumps[0]
        # Ori = inv.rpos[0].pp[0]
        # fe, fv, rfv, bases = inv.poFvPoinc('rpo', 1, ptsId, 1, [0, 3, 4],
        #                                      reflect=True)
        
        ptsId = inv.rpos[1].jumps[1]
        Ori = inv.rpos[1].pp[1]
        fe, fv, rfv, bases = inv.poFvPoinc('rpo', 2, ptsId, 1, [0, 1, 4],
                                             reflect=False)

        inv.poPoincProject(bases, Ori)

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        ax.scatter(0, 0, 0, c='k', s=70, edgecolors='none')
        for i in range(len(inv.ppos)):
            p = inv.ppos[i].ppp
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='y', marker='o',
                       s=20, edgecolors='none')
        for i in range(len(inv.rpos)):
            p = inv.rpos[i].ppp
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='y', marker='o',
                       s=20, edgecolors='none')
        ax3d(fig, ax)
        
        if True:
            trace = []
            trace.append(ptlyTrace3d([0], [0], [0], plotType=1, ms=7,
                                     mc='black'))
            for i in range(len(inv.ppos)):
                p = inv.ppos[i].ppp
                trace.append(ptlyTrace3d(p[:, 0], p[:, 1], p[:, 2],
                                         plotType=1, ms=2, mc='red',
                                         mt='circle'))
            for i in range(len(inv.rpos)):
                p = inv.rpos[i].ppp
                trace.append(ptlyTrace3d(p[:, 0], p[:, 1], p[:, 2],
                                         plotType=1, ms=2, mc='red',
                                         mt='circle'))
            ptly3d(trace, 'PoPoincare', labs=['$v_1$', '$v_2$', '$v_3$'])

        if False:
            ii = [2, 6, 4]
            # ii = [1, 5, 3]
            fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
            for i in range(len(pos)):
                inv.plotFundOrbit(ax, pos[i], poJumps[i], ii, alpha=0.5)
                ax.scatter(borderPts[i][:, ii[0]], borderPts[i][:, ii[1]],
                           borderPts[i][:, ii[2]], c='k')
            ax3d(fig, ax)

        np.savez_compressed('bases', Ori=Ori, bases=bases)
        inv.savePoPoincProject()

    if case == 25:
        """
        view 2 shadowing orbits
        """
        poIds = [[1, 26], range(4, 4)]
        pos, poDom, poJumps = inv.loadPO('../../data/ks22h001t120x64EV.h5',
                                           poIds, 1)
        ii = [1, 3, 2]

        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        for i in range(len(pos)):
            inv.plotFundOrbit(ax, pos[i], poJumps[i], ii,
                                c='k' if i > 0 else 'r',
                                alpha=1, lw=1.5)
        ax3d(fig, ax)
        
    if case == 200:
        """
        visualize the unstable manifold of eq and req together
        """
        nn = 10
        MuE, MuTw = inv.getMuAll(1, nn=nn)
        ii = [1, 3, 5]
        
        cs = ['r', 'b', 'c', 'k', 'y']
        fig, ax = pl3d(labs=[r'$v_1$', r'$v_2$', r'$v_3$'])
        inv.plotRE(ax, ii)
        for k in range(1, len(MuE)-3):
            for i in range(nn):
                inv.plotFundOrbit(ax, MuE[0][0][i], MuE[0][2][i],
                                    ii, c=cs[k])
        E2, E3 = inv.Eg['E2'], inv.Eg['E3']
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

