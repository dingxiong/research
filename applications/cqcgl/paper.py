from py_CQCGL1d import *
from personalFunctions import *
import matplotlib.gridspec as gridspec

case = 10


if case == 10:
    """
    plot the Bi - Gi req stability plot using the saved date from running viewReq.py
    """
    saveData = np.loadtxt("BiGiReqStab.dat")
    cs = {0: 'c', 1: 'm', 2: 'm', 4: 'r', 6: 'b'}#, 8: 'y', 56: 'r', 14: 'grey'}
    ms = {0: '^', 1: '^', 2: 'x', 4: 'o', 6: 'D'}#, 8: '8', 56: '4', 14: 'v'}
    fig, ax = pl2d(size=[8, 6], labs=[r'$\gamma_i$', r'$\beta_i$'], axisLabelSize=25,
                   tickSize=20, xlim=[-6, 0], ylim=[-4, 6])
    for item in saveData:
        Gi, Bi, m = item
        ax.scatter(Gi, Bi, s=18 if m > 0 else 18, edgecolors='none',
                   marker=ms.get(m, 'v'), c=cs.get(m, 'k'))
    # ax.grid(which='both')
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
    fig = plt.figure(figsize=[8, 6])
    for i in range(2):
        Bi, Gi = params[i]

        cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
        rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)
        x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                                         Bi, Gi, 1, flag=2)
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
                    transform=ax.transAxes, fontsize=18, color='white')
            if i == 1:
                ax.set_xlabel(r'$x$', fontsize=20)
            if j == 0:
                ax.set_ylabel(r'$t$', fontsize=20)
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
        x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                                         Bi, Gi, 1, flag=2)
        ax = fig.add_subplot('12' + str(i+1))
        ax.text(0.1, 0.9, '('+ chr(ord('a')+i) + ')', horizontalalignment='center',
                    transform=ax.transAxes, fontsize=18)
        ax.set_xlabel(r'$x$', fontsize=20)
        # ax.set_ylabel(r'$|v_1|$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=12)
        Aamp = np.abs(cgl.Fourier2Config(v[0]))
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

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.readReqBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, 1, flag=2)
    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50
    aa = cgl.intg(aE, h, np.int(T/h), 1)
    # cp.config(aa, [0, d, 0, T])
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    
    ax = fig.add_subplot(gs[:nx/2-1, 0])
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
    
    ax = fig.add_subplot(gs[:nx/2, 1:], projection='3d')
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
    
    # fig, ax = pl3d(size=[8, 6], labs=[r'$Re(a_{-1})$', r'$Re(a_{2})$', r'$Im(a_{2})$'],
    #                axisLabelSize=20, tickSize=20)
    # ax.scatter(a0H[-1], a0H[4], a0H[5], c='r', s=40)
    # ax.set_ylim([-50, 200])
    # ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    # ax3d(fig, ax)

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
    
    ax = fig.add_subplot(gs[nx/2:nx-1, 0])
    ax.text(0.1, 0.9, '(c)', horizontalalignment='center',
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
    
    ax = fig.add_subplot(gs[nx/2:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(d)', horizontalalignment='center',
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

if case == 50:
    """
    plot one req explosion and its symmetry reduced plots
    """
    N, d = 1024, 50
    h = 2e-3
    
    Bi, Gi = 2, -2
    sysFlag = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.readReqBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, 1, flag=2)
    a0H = cgl.orbit2slice(a0, sysFlag)[0]
    
    aE = a0 + 0.001 * norm(a0) * v[0].real
    T = 50
    aa = cgl.intg(aE, h, np.int(T/h), 1)
    cp.config(aa, [0, d, 0, T])
    aaH, ths, phis = cgl.orbit2slice(aa, sysFlag)
    # cp.config(aaH, [0, d, 0, T])
    
    fig, ax = pl3d(size=[8, 6], labs=[r'$Re(a_{-1})$', r'$Re(a_{2})$', r'$Im(a_{2})$'],
                   axisLabelSize=20, tickSize=20)
    ax.scatter(a0H[-1], a0H[4], a0H[5], c='r', s=40)
    ax.set_ylim([-50, 200])
    ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    ax3d(fig, ax)
    
    
if case == 60:
    """
    plot the stability table of the rpos
    """
    N, d = 1024, 50

    fileName = '../../data/cgl/rpoBiGiEV.h5'
    
    cs = {0: 'g'}#, 10: 'm', 2: 'c', 4: 'r', 6: 'b', 8: 'y'}#, 56: 'r', 14: 'grey'}
    ms = {0: 's'}#, 10: '^', 2: '+', 4: 'o', 6: 'D', 8: 'v'}#, 56: '4', 14: 'v'}

    fig, ax = pl2d(size=[8, 6], labs=[r'$\gamma_i$', r'$\beta_i$'], axisLabelSize=25, tickSize=20,
                   xlim=[-5.7, -3.95], ylim=[1.8, 6])
    for i in range(39):
        Bi = 1.9 + i*0.1
        for j in range(55):
            Gi = -5.6 + 0.1*j
            rpo = CQCGLrpo()
            if rpo.checkExist(fileName, rpo.toStr(Bi, Gi, 1) + '/er'):
                x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi(fileName, Bi, Gi, 1, flag=2)
                m, ep, accu = numStab(e, nmarg=3, tol=1e-4, flag=1)
                ax.scatter(Gi, Bi, s=60, edgecolors='none',
                           marker=ms.get(m, 'o'), c=cs.get(m, 'r'))

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.grid(which='major', linewidth=2)
    #ax.grid(which='minor', linewidth=1)
    ax2d(fig, ax)
    

if case == 70:
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

