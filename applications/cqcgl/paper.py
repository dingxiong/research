from py_CQCGL1d import *
from py_CQCGL1dEIDc import *
from cglHelp import *
import matplotlib.gridspec as gridspec

case = 150


if case == 10:
    """
    plot the Bi - Gi req stability plot using the saved date from running viewReq.py
    data format : {Gi, Bi, #unstable}
    #unstalbe is 0, 2, 4, 6 ...
    """
    saveData = np.loadtxt("BiGiReqStab.dat")
    cs = {0: 'c', 1: 'm', 2: 'g', 4: 'r', 6: 'b'}#, 8: 'y', 56: 'r', 14: 'grey'}
    ms = {0: '^', 1: '^', 2: 'x', 4: 'o', 6: 'D'}#, 8: '8', 56: '4', 14: 'v'}
    fig, ax = pl2d(size=[8, 6], labs=[r'$\gamma_i$', r'$\beta_i$'], axisLabelSize=30,
                   tickSize=20, xlim=[-6, 0], ylim=[-4, 6])
    for item in saveData:
        Gi, Bi, m = item
        ax.scatter(Gi, Bi, s=18 if m > 0 else 18, edgecolors='none',
                   marker=ms.get(m, 'v'), c=cs.get(m, 'k'))
    # ax.grid(which='both')
    ax2d(fig, ax)

if case == 15:
    """
    same as case 10, bu plot contourf plot
    Note: the problem of contourf() is that whenever there is a jump
    in the levels, say from lv1 to lv2 but lv2 = lv1 + 2, then it will
    interpolate these two levels, thus a line for lv1 + 1 is plotted
    between these two regions.
    I need to draw each region indivisually. 
    """
    saveData = np.loadtxt("BiGiReqStab.dat")
    BiRange = [-3.2, 4]
    GiRange = [-5.6, -0.9]
    inc = 0.1

    nx = np.int(np.rint( (BiRange[1] - BiRange[0])/inc ) + 1)
    ny = np.int(np.rint( (GiRange[1] - GiRange[0])/inc ) + 1)

    x = np.linspace(BiRange[0], BiRange[1], nx)
    y = np.linspace(GiRange[0], GiRange[1], ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros([ny, nx]) - 1  # initialized with -1

    for item in saveData:
        Gi, Bi, m = item
        idx = np.int(np.rint((Bi - BiRange[0]) / inc))
        idy = np.int(np.rint((Gi - GiRange[0]) / inc))
        if idx >= 0 and idx < nx and idy >= 0 and idy < ny:
            Z[idy, idx] = m if m < 8 else 8

    levels = [-1, 0, 2, 4, 6, 8]
    cs = ['w', 'm', 'c', 'grey', 'r', 'y']
    fig, ax = pl2d(size=[8, 6], labs=[r'$\beta_i$', r'$\gamma_i$'], axisLabelSize=30,
                   tickSize=20)
    for i in range(len(levels)):
        vfunc = np.vectorize(lambda t : 0 if t == levels[i] else 1)
        Z2 = vfunc(Z)
        ax.contourf(X, Y, Z2, levels=[-0.1, 0.9], colors=cs[i])
    ax.text(0, -2, r'$S$', fontsize=30)
    ax.text(0, -4.5, r'$U_1$', fontsize=30)
    ax.text(1.8, -2.5, r'$U_2$', fontsize=30)
    ax.text(2., -4.8, r'$U_3$', fontsize=30)
    ax.text(3., -3, r'$U_{\geq 4}$', fontsize=30)
    ax.scatter(2.0,-5, c='k', s=100, edgecolor='none')
    ax.scatter(2.7,-5, c='k', s=100, edgecolor='none')
    ax.scatter(1.4,-3.9, c='k', marker='*', s=200, edgecolor='none')
    ax.set_xlim(BiRange)
    ax.set_ylim(GiRange)
    ax2d(fig, ax)
            
    
if case == 20:
    """
    use L = 50 to view the rpo, its unstable manifold and the difference plot
    """
    N, d = 1024, 50
    params = [[4.6, -5.0], [4.8, -4.5]]
    intgOfT = [19, 7]
    baseT = [[0, 4], [0, 4]]
    WuT = [[12, 19], [2, 7]]
    difT = [[7, 16], [0, 6]]
    skipRate = 10
    fig = plt.figure(figsize=[8, 7])
    for i in range(2):
        Bi, Gi = params[i]

        cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
        rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)
        x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5', rpo.toStr(Bi, Gi, 1), flag=2)
        a0 = x[:cgl.Ndim]
        print e[:10]
        aa0 = cgl.intg(a0, T/nstp, nstp, 10)[:-1]

        a0 += 0.1*norm(a0)*v[0]

        aaBase = aa0
        for k in range(1, intgOfT[i]):
            aaBase = np.vstack((aaBase, cgl.Rotate(aa0, -k*th, -k*phi)))

        aa = cgl.intg(a0, T/nstp, intgOfT[i]*nstp, skipRate)
        dif = aa[:-1] - aaBase
        cellSize = nstp / skipRate
        # cp.config(dif, [0, d, 0*T, 6*T])
        
        aaBase = aaBase[baseT[i][0]*cellSize : baseT[i][1]*cellSize]
        aa = aa[WuT[i][0]*cellSize : WuT[i][1]*cellSize]
        dif = dif[difT[i][0]*cellSize : difT[i][1]*cellSize]

        states = [aaBase, aa, dif]
        for j in range(3):
            ax = fig.add_subplot('23' + str(3*i+j+1))
            ax.text(0.1, 0.9, '('+ chr(ord('a')+3*i+j) + ')', horizontalalignment='center',
                    transform=ax.transAxes, fontsize=20, color='white')
            if i == 1:
                ax.set_xlabel(r'$x$', fontsize=25)
            if j == 0:
                ax.set_ylabel(r'$t$', fontsize=25)
            if i == 0:
                ax.set_xticklabels([])
            ax.tick_params(axis='both', which='major', labelsize=12)
            if j == 0:
                extent = [0, d, baseT[i][0]*T, baseT[i][1]*T]
            elif j == 1:
                extent = [0, d, WuT[i][0]*T, WuT[i][1]*T]
            else:
                extent = [0, d, difT[i][0]*T, difT[i][1]*T]
            im = ax.imshow(np.abs(cgl.Fourier2Config(states[j])), cmap=plt.get_cmap('jet'), extent=extent, 
                           aspect='auto', origin='lower')
            ax.grid('on')
            dr = make_axes_locatable(ax)
            cax =dr.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
            fig.tight_layout(pad=0)

    plt.show(block=False)

if case == 30:
    """
    view the leading FV of two rpo.
    The leading FV of these two rpo are real, so there is no need to 
    special transformation
    """
    N, d = 1024 , 50
    params = [[4.6, -5.0], [4.8, -4.5]]
    fig = plt.figure(figsize=[8, 4])
    for i in range(2):
        Bi, Gi = params[i]
        cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
        rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)
        x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5', rpo.toStr(Bi, Gi, 1), flag=2)
        ax = fig.add_subplot('12' + str(i+1))
        ax.text(0.1, 0.9, '('+ chr(ord('a')+i) + ')', horizontalalignment='center',
                transform=ax.transAxes, fontsize=20)
        ax.set_xlabel(r'$x$', fontsize=30)
        # ax.set_ylabel(r'$|v_1|$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=13)
        Aamp = np.abs(cgl.Fourier2Config(v[0]))
        ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp.shape[0]*Aamp, lw=1.5)
    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 35:
    """
    Similar to case = 30, but for rpo(4.6, -5.0), we plot all its unstable Floquet vectors.
    """
    N, d = 1024 , 50
    
    vs = []
    Bi, Gi = 4.6, -5.0
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)
    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5', rpo.toStr(Bi, Gi, 1), flag=2)
    for i in range(3):
        vs.append(v[i])

    # do not need to define another instance of cgl
    Bi, Gi = 4.8, -4.5
    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5', rpo.toStr(Bi, Gi, 1), flag=2)
    vs.insert(1, v[0])

    fig = plt.figure(figsize=[8, 7])
    for i in range(4):
        ax = fig.add_subplot('22' + str(i+1))
        ax.text(0.1, 0.9, '('+ chr(ord('a')+i) + ')', 
                horizontalalignment='center', transform=ax.transAxes, fontsize=20)
        if i > 1:
            ax.set_xlabel(r'$x$', fontsize=30)
        # ax.set_ylabel(r'$|v_1|$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=13)
        Aamp = np.abs(cgl.Fourier2Config(vs[i]))
        ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp.shape[0]*Aamp, lw=1.5)


    fig.tight_layout(pad=0)
    plt.show(block=False)
    

if case == 40:
    """
    use L = 50 to view the one req and one rpo int the symmetry reduced state space
    """
    N, d = 1024 , 50
    h = 2e-3
    sysFlag = 1
    
    fig = plt.figure(figsize=[8, 8])
    nx, ny = 16, 4
    gs = gridspec.GridSpec(nx, ny)

    # req
    Bi, Gi = 2, -2

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=2)    
    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50
    aa = EI.intg(aE, h, T, 10)
    aaH = cgl.orbit2slice(aa, sysFlag)[0]
    
    ax = fig.add_subplot(gs[:nx/2-1, 0])
    ax.text(0.9, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=20, color='white')
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$t$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    aaTime = aa[cp.sliceTime(EI.Ts()), :]
    im = ax.imshow(np.abs(cgl.Fourier2Config(aaTime)), 
                   cmap=plt.get_cmap('jet'), extent=[0, d, 0, T], 
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    
    ax = fig.add_subplot(gs[:nx/2, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(b)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=20)
    ax.set_xlabel(r'$Re(a_0)$', fontsize=18)
    ax.set_ylabel(r'$Re(a_2)$', fontsize=18)
    ax.set_zlabel(r'$Im(a_2)$', fontsize=18)
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([0, 250])
    # ax.set_zlim([-30, 30])
    ax.locator_params(nbins=4)
    ax.scatter(a0H[0], a0H[4], a0H[5], c='r', s=100, edgecolor='none')
    ax.plot(aaH[:, 0], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.5)
    
    # rpo
    Bi, Gi = 4.8, -4.5

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)

    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5',
                                              rpo.toStr(Bi, Gi, 1), flag=2)
    a0 = x[:cgl.Ndim]
    po = EI.intg(a0, T/nstp, T, 1)
    poH = cgl.orbit2slice(po, sysFlag)[0]

    aE = a0 + 0.001 * norm(a0) * v[0];

    aa = EI.intg(aE, T/nstp, 16*T, 1)
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    
    ax = fig.add_subplot(gs[nx/2:nx-1, 0])
    ax.text(0.9, 0.9, '(c)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=20, color='white')
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$t$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    aaTime = aa[cp.sliceTime(EI.Ts()), :]
    im = ax.imshow(np.abs(cgl.Fourier2Config(aaTime)), 
                   cmap=plt.get_cmap('jet'), extent=[0, d, 0, 16*T], 
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    
    ax = fig.add_subplot(gs[nx/2:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(d)', horizontalalignment='center',
              transform=ax.transAxes, fontsize=20)
    ax.set_xlabel(r'$Re(a_{0})$', fontsize=18)
    ax.set_ylabel(r'$Re(a_{2})$', fontsize=18)
    ax.set_zlabel(r'$Im(a_{2})$', fontsize=18)
    # ax.set_ylim([-50, 200])
    ax.locator_params(nbins=4)
    ax.plot(poH[:, 0], poH[:, 4], poH[:, 5], c='r', lw=2)
    ax.plot(aaH[:, 0], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.5)

    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 50:
    """
    plot one req explosion and its symmetry reduced plots
    use it to test case = 40
    """
    N, d = 1024, 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi = 2, -2

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=2)

    # get the bases
    vt = np.vstack([v[0].real, v[0].imag, v[6].real])
    vRed = cgl.ve2slice(vt, a0, sysFlag)
    Q = orthAxes(vRed[0], vRed[1], vRed[2])

    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    a0P = a0H.dot(Q)
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50
    aa = EI.intg(aE, h, T, 2)
    aaH = cgl.orbit2slice(aa, sysFlag)[0]
    aaP = aaH.dot(Q) - a0P
    
    fig, ax = pl3d(size=[8, 6], labs=[r'$v_1$', r'$v_2$', r'$v_3$'],
                   axisLabelSize=20, tickSize=20)
    ax.scatter(0, 0, 0, c='k', s=100, edgecolors='none')
    id1 = bisect_left(EI.Ts(), 24)
    ax.plot(aaP[id1/2:id1, 0], aaP[id1/2:id1, 1], aaP[id1/2:id1, 2], c='m', lw=1, alpha=0.5)
    id2 = bisect_left(EI.Ts(), 33)
    ax.plot(aaP[id2:, 0], aaP[id2:, 1], aaP[id2:, 2], c='b', lw=1, alpha=0.6)
    ax3d(fig, ax)
    
if case == 55:
    """
    plot one rpo explosion and its symmetry reduced plots
    use it to test case = 40
    """
    N, d = 1024, 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi = 4.8, -4.5

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    EI = pyCQCGL1dEIDc(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)

    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5',
                                              rpo.toStr(Bi, Gi, 1), flag=2)
    a0 = x[:cgl.Ndim]
    
    # get the bases
    vt = np.vstack([v[0], v[1], v[3]])
    vRed = cgl.ve2slice(vt, a0, sysFlag)
    Q = orthAxes(vRed[0], vRed[1], vRed[2])

    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    a0P = a0H.dot(Q)
    
    po = EI.intg(a0, T/nstp, T, 1)
    poH = cgl.orbit2slice(po, sysFlag)[0]
    poP = poH.dot(Q) - a0P

    aE = a0 + 0.001 * norm(a0) * v[0];

    aa = EI.intg(aE, T/nstp, 16*T, 2)
    aaH = cgl.orbit2slice(aa, sysFlag)[0]
    aaP = aaH.dot(Q) - a0P
    
    fig, ax = pl3d(size=[8, 6], labs=[r'$v_1$', r'$v_2$', r'$v_3$'],
                   axisLabelSize=20, tickSize=20)
    ax.scatter(0, 0, 0, c='k', s=100, edgecolors='none')
    ax.plot(poP[:, 0], poP[:, 1], poP[:, 2], c='r', lw=2)
    id1 = bisect_left(EI.Ts(), 23)
    ax.plot(aaP[id1/2:id1, 0], aaP[id1/2:id1, 1], aaP[id1/2:id1, 2], c='m', lw=1, alpha=0.5)
    id2 = bisect_left(EI.Ts(), 35)
    ax.plot(aaP[id2:, 0], aaP[id2:, 1], aaP[id2:, 2], c='b', lw=1, alpha=0.6)
    ax3d(fig, ax)
    
    
if case == 60:
    """
    plot the stability table of the rpos
    """
    N, d = 1024, 50

    fileName = '../../data/cgl/rpoBiGiEV.h5'
    
    cs = {0: 'g'}#, 10: 'm', 2: 'c', 4: 'r', 6: 'b', 8: 'y'}#, 56: 'r', 14: 'grey'}
    ms = {0: 's'}#, 10: '^', 2: '+', 4: 'o', 6: 'D', 8: 'v'}#, 56: '4', 14: 'v'}

    fig, ax = pl2d(size=[8, 6], labs=[r'$\beta_i$', r'$\gamma_i$'], axisLabelSize=30, tickSize=20,
                   ylim=[-5.65, -3.95], xlim=[1.8, 5.8])
    for i in range(39):
        Bi = 1.9 + i*0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            rpo = CQCGLrpo()
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1) + '/er'):
                x, T, nstp, th, phi, err, e, v = rpo.read(fileName, rpo.toStr(Bi, Gi, 1), flag=2)
                m, ep, accu = numStab(e, nmarg=3, tol=1e-4, flag=1)
                ax.scatter(Bi, Gi, s=60, edgecolors='none',
                           marker=ms.get(m, 'o'), c=cs.get(m, 'r'))

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.grid(which='major', linewidth=2)
    #ax.grid(which='minor', linewidth=1)
    ax2d(fig, ax)
    

if case == 70:
    """
    plot one Hopf bifurcation
    subplot (a) : heat map of the limit cycle
    subplot (b) : symmetry-reduced state space figure
    """
    N, d = 1024, 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi = 1.4, -3.9
    index = 1
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)

    # obtain unstable manifold of req
    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, index), flag=2)

    # get the bases
    vt = np.vstack([v[0].real, v[0].imag, v[4].real])
    vRed = cgl.ve2slice(vt, a0, sysFlag)
    Q = orthAxes(vRed[0], vRed[1], vRed[2])

    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    a0P = a0H.dot(Q)

    aE = a0 + 0.1*norm(a0)*v[0].real
    T = 800
    aa = cgl.intg(aE, h, np.int(100/h), 100000)
    aE = aa[-1] 
    aa = cgl.intg(aE, h, np.int(T/h), 50)
    aaH = cgl.orbit2slice(aa, sysFlag)[0]
    aaP = aaH.dot(Q) - a0P
    
    # obtain limit cycle
    x, T, nstp, th, phi, err = rpo.read('../../data/cgl/rpoHopfBiGi.h5', rpo.toStr(Bi, Gi, 1))
    ap0 = x[:-3]
    hp = T / nstp
    aap = cgl.intg(ap0, hp, nstp, 10)
    aapH = cgl.orbit2slice(aap, sysFlag)[0]
    aapP = aapH.dot(Q) - a0P
    
    aap4 = aap
    for k in range(1, 4):
        aap4 = np.vstack((aap4, cgl.Rotate(aap, -k*th, -k*phi)))
    
    # plot figure 
    fig = plt.figure(figsize=[8, 4])
    nx, ny = 8, 4
    gs = gridspec.GridSpec(nx, ny)

    ax = fig.add_subplot(gs[:nx-1, 0])
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='white')
    ax.set_xlabel(r'$x$', fontsize=25)
    ax.set_ylabel(r'$t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=12)
    im = ax.imshow(np.abs(cgl.Fourier2Config(aap4)), cmap=plt.get_cmap('jet'), extent=[0, d, 0, 4*T], 
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    
    ax = fig.add_subplot(gs[:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(b)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)
    ax.set_xlabel(r'$v_1$', fontsize=25)
    ax.set_ylabel(r'$v_2$', fontsize=25)
    ax.set_zlabel(r'$v_3$', fontsize=25)
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([0, 250])
    # ax.set_zlim([-30, 30])
    ax.locator_params(nbins=4)
    # ax.scatter(a0H[], a0H[4], a0H[5], c='r', s=40)
    # ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    # ax.plot(aapH[:, -1], aapH[:, 4], aapH[:, 5], c='r', lw=2)
    ax.scatter(0, 0, 0, c='r', s=40)
    ax.plot(aaP[:, 0], aaP[:, 1], aaP[:, 2], c='b', lw=1, alpha=0.7)
    ax.plot(aapP[:, 0], aapP[:, 1], aapP[:, 2], c='r', lw=2)

    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 75:
    """
    same as case 70 but only save the data
    """
    N, d = 1024, 50
    h = 2e-3
    sysFlag = 1

    Bi, Gi = 1.4, -3.9
    index = 1
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)

    # obtain unstable manifold of req
    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, index), flag=2)

    # get the bases
    vt = np.vstack([v[0].real, v[0].imag, v[4].real])
    vRed = cgl.ve2slice(vt, a0, sysFlag)
    Q = orthAxes(vRed[0], vRed[1], vRed[2])

    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    a0P = a0H.dot(Q)

    aE = a0 + 0.1*norm(a0)*v[0].real
    T = 800
    aa = cgl.intgC(aE, h, 100, 100000)
    aE = aa[-1] if aa.ndim == 2 else aa 
    aa = cgl.intgC(aE, h, T, 50)
    aaH = cgl.orbit2slice(aa, sysFlag)[0]
    aaP = aaH.dot(Q) - a0P
    
    # obtain limit cycle
    x, T, nstp, th, phi, err = rpo.read('../../data/cgl/rpoHopfBiGi.h5', rpo.toStr(Bi, Gi, 1))
    ap0 = x[:-3]
    hp = T / nstp
    aap = cgl.intgC(ap0, hp, T, 10)
    aapH = cgl.orbit2slice(aap, sysFlag)[0]
    aapP = aapH.dot(Q) - a0P
    
    aap4 = aap
    for k in range(1, 4):
        aap4 = np.vstack((aap4, cgl.Rotate(aap, -k*th, -k*phi)))
    
    # save the data

    hopfCycleProfiles4Periods = np.abs(cgl.Fourier2Config(aap4))
    np.savez_compressed('hopfCycleAndWuOfReq', hopfCycleProfiles4Periods=hopfCycleProfiles4Periods, T=T, aaP=aaP, aapP=aapP)

if case == 80:
    """
    new size L = 50
    plot the plane soliton and composite soliton in the same figure.
    """
    N, d = 1024, 50
    
    fig, ax = pl2d(size=[8, 6], labs=[r'$x$', r'$|A|$'], axisLabelSize=30, tickSize=20)

    Bi, Gi = 2.0, -5
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    a0, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=0)
    Aamp = np.abs(cgl.Fourier2Config(a0))
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2, ls='-', c='k')

    Bi, Gi = 2.7, -5
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    a0, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=0)
    Aamp = np.abs(cgl.Fourier2Config(a0))
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2, ls='--', c='k')
    
    ax2d(fig, ax)

if case == 90:
    """
    save the data for the two coexsting solitons for the same parameter
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    a1, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=0)
    a2, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 2), flag=0)
    absA1 = np.abs(cgl.Fourier2Config(a1))
    absA2 = np.abs(cgl.Fourier2Config(a2))
    np.savez_compressed('solitonProfileFor08n06', absA1=absA1, absA2=absA2)

if case == 100:
    """
    same as case = 40. but in two different plots
    """
    N, d = 1024 , 50
    h = 2e-3
    sysFlag = 1
    
    fig = plt.figure(figsize=[8, 4])
    nx, ny = 8, 4
    gs = gridspec.GridSpec(nx, ny)

    # req
    Bi, Gi = 2, -2

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.readReqBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, 1, flag=2)
    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50
    aa = cgl.intg(aE, h, np.int(T/h), 1)
    # cp.config(aa, [0, d, 0, T])
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    
    ax = fig.add_subplot(gs[:nx-1, 0])
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='white')
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$t$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    im = ax.imshow(np.abs(cgl.Fourier2Config(aa)), cmap=plt.get_cmap('jet'), extent=[0, d, 0, T], 
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    
    ax = fig.add_subplot(gs[:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(b)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)
    ax.set_xlabel(r'$Re(a_{-1})$', fontsize=16)
    ax.set_ylabel(r'$Re(a_{2})$', fontsize=16)
    ax.set_zlabel(r'$Im(a_{2})$', fontsize=16)
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 250])
    ax.set_zlim([-30, 30])
    ax.locator_params(nbins=4)
    ax.scatter(a0H[-1], a0H[4], a0H[5], c='r', s=40)
    ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)

    fig.tight_layout(pad=0)
    plt.show(block=False)


    fig = plt.figure(figsize=[8, 4])
    nx, ny = 8, 4
    gs = gridspec.GridSpec(nx, ny)

    # rpo
    Bi, Gi = 4.8, -4.5

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)

    x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                               Bi, Gi, 1, flag=2)
    a0 = x[:cgl.Ndim]
    po = cgl.intg(a0, T/nstp, nstp, 10)
    poH = cgl.orbit2slice(po, sysFlag)[0]

    aE = a0 + 0.001 * norm(a0) * v[0];

    aa = cgl.intg(aE, T/nstp, 20*nstp, 1)
    # cp.config(aa, [0, d, 0, 20*T])
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    
    ax = fig.add_subplot(gs[:nx-1, 0])
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='white')
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$t$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    im = ax.imshow(np.abs(cgl.Fourier2Config(aa)), cmap=plt.get_cmap('jet'), extent=[0, d, 0, 20*T], 
                   aspect='auto', origin='lower')
    ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    
    ax = fig.add_subplot(gs[:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(b)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)
    ax.set_xlabel(r'$Re(a_{-1})$', fontsize=16)
    ax.set_ylabel(r'$Re(a_{2})$', fontsize=16)
    ax.set_zlabel(r'$Im(a_{2})$', fontsize=16)
    # ax.set_ylim([-50, 200])
    ax.locator_params(nbins=4)
    ax.plot(poH[:, -1], poH[:, 4], poH[:, 5], c='r', lw=2)
    ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)

    # fig, ax = pl3d(size=[8, 6], labs=[r'$Re(a_{-1})$', r'$Re(a_{2})$', r'$Im(a_{2})$'],
    #                axisLabelSize=20, tickSize=20)
    # ax.plot(poH[:, -1], poH[:, 4], poH[:, 5], c='r', lw=2)
    # ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    # ax3d(fig, ax)

    fig.tight_layout(pad=0)
    plt.show(block=False)



if case == 150:
    """
    find one representative symmetric and asymmetric explosions
    save the data
    """
    N, d = 1024 , 50
    h = 2e-3    
    Bi, Gi = 0.8, -0.6

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=2)    
    aE = a0 + 0.001 * norm(a0) * v[0].real

    # get rid of transicent
    aa = cgl.intgC(aE, h, 69, 100000)
    aE = aa[-1] if aa.ndim == 2 else aa

    # save symmetric explosion
    Tsymmetric = 10
    aa = cgl.intgC(aE, h, Tsymmetric, 10)
    AampSymmetric = np.abs(cgl.Fourier2Config(aa))
    aE = aa[-1] if aa.ndim == 2 else aa
    
    # save asymmetric explosion
    aa = cgl.intgC(aE, h, 189, 100000)
    aE = aa[-1] if aa.ndim == 2 else aa
    Tasymmetric = 10
    aa = cgl.intgC(aE, h, Tasymmetric, 10)
    AampAsymmetric = np.abs(cgl.Fourier2Config(aa))
    a0RealAsymmetric = aa[:, 0].real
    aE = aa[-1] if aa.ndim == 2 else aa

    np.savez_compressed('explosionExample08n06', AampSymmetric=AampSymmetric, AampAsymmetric=AampAsymmetric, 
                        Tsymmetric=Tsymmetric, Tasymmetric=Tasymmetric)
    
