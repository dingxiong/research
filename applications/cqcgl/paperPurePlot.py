from cglHelp import *
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})  # I love Times
rc('text', usetex=True)
import matplotlib.gridspec as gridspec

case = 240

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
    BiRange = [-3.2, 3.5]
    GiRange = [-5.6, -0.5]
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
                   tickSize=25)
    for i in range(len(levels)):
        vfunc = np.vectorize(lambda t : 0 if t == levels[i] else 1)
        Z2 = vfunc(Z)
        ax.contourf(X, Y, Z2, levels=[-0.1, 0.9], colors=cs[i])

    # add zoom rectangle
    BiZoom = [1.9, 3.5]
    GiZoom = [-5.6, -4]
    ax.plot([BiZoom[0], BiZoom[0]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[1], BiZoom[1]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[0], GiZoom[0]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[1], GiZoom[1]], ls='--', lw=3, c='k')

    ax.text(0, -2, r'$S$', fontsize=30)
    ax.text(0, -4.5, r'$U_1$', fontsize=30)
    ax.text(1.8, -2.5, r'$U_2$', fontsize=30)
    ax.text(2., -4.8, r'$U_3$', fontsize=30)
    ax.text(2.7, -3, r'$U_{\geq 4}$', fontsize=30)
    #ax.scatter(2.0,-5, c='k', s=100, edgecolor='none')
    #ax.scatter(2.7,-5, c='k', s=100, edgecolor='none')
    ax.scatter(0.8, -0.6, c='k', s=100, edgecolor='none')
    ax.scatter(1.4,-3.9, c='k', marker='*', s=200, edgecolor='none')
    ax.set_xlim(BiRange)
    ax.set_ylim(GiRange)

    ax.set_xticks(range(-3, 4))
    ax2d(fig, ax)

if case == 60:
    """
    plot the stability table of the rpos
    """
    N, d = 1024, 50

    fileName = '../../data/cgl/rpoBiGiEV.h5'
    
    cs = {0: 'g'}#, 10: 'm', 2: 'c', 4: 'r', 6: 'b', 8: 'y'}#, 56: 'r', 14: 'grey'}
    ms = {0: 's'}#, 10: '^', 2: '+', 4: 'o', 6: 'D', 8: 'v'}#, 56: '4', 14: 'v'}

    fig, ax = pl2d(size=[8, 6], labs=[r'$\beta_i$', r'$\gamma_i$'], axisLabelSize=30, tickSize=25,
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

    # add zoom rectangle
    BiZoom = [1.9, 3.5]
    GiZoom = [-5.6, -4]
    ax.plot([BiZoom[1], BiZoom[1]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[0]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[0], GiZoom[0]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[1], GiZoom[1]], ls='--', lw=3, c='k')
    # ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[1], GiZoom[1]], ls='--', lw=3, c='k')

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.grid(which='major', linewidth=2)
    #ax.grid(which='minor', linewidth=1)
    ax.scatter(4.6, -5.0, c='k', s=200, edgecolor='none')
    ax.scatter(4.8, -4.5, c='k', marker='*', s=400, edgecolor='none')

    ax.set_yticks([-4, -4.5, -5, -5.5])
    ax.set_xticks(np.arange(2, 6, 0.5))
    ax2d(fig, ax)


if case == 90:
    """
    plot the two coexisting soltions for the same paramter in the same figure
    load saved data
    """
    N, d = 1024, 50
    
    fig, ax = pl2d(size=[8, 6], labs=[r'$x$', r'$|A|$'], axisLabelSize=30, tickSize=25)
    Bi, Gi = 0.8, -0.6
    
    profiles = np.load('solitonProfileFor08n06.npz')
    absA1, absA2 = profiles['absA1'], profiles['absA2']
    ax.plot(np.linspace(0, d, absA1.shape[0]), absA1, lw=2, ls='-', c='r')
    ax.plot(np.linspace(0, d, absA2.shape[0]), absA2, lw=2, ls='--', c='b')
    
    ax2d(fig, ax)

if case == 76:
    """
    plot one Hopf bifurcation using save data
    subplot (a) : heat map of the limit cycle
    subplot (b) : symmetry-reduced state space figure
    """
    N, d = 1024, 50
    data = np.load('hopfCycleAndWuOfReq.npz')
    hopfCycleProfiles4Periods, T, aaP, aapP = data['hopfCycleProfiles4Periods'], data['T'], data['aaP'], data['aapP']

    # plot figure 
    fig = plt.figure(figsize=[8, 4])
    nx, ny = 8, 4
    gs = gridspec.GridSpec(nx, ny)
    
    ax = fig.add_subplot(gs[:nx-1, 0])
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18, color='white')
    ax.set_xlabel(r'$x$', fontsize=25)
    ax.set_ylabel(r'$t$', fontsize=25)
    ax.set_xticks(range(0, 60, 10))
    ax.set_yticks(range(0, 16, 3))
    ax.tick_params(axis='both', which='major', labelsize=15)
    im = ax.imshow(hopfCycleProfiles4Periods, cmap=plt.get_cmap('jet'), extent=[0, d, 0, 4*T], 
                   aspect='auto', origin='lower')
    # ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 2, 0.4))
    cb.ax.tick_params(labelsize=15)
    
    ax = fig.add_subplot(gs[:, 1:], projection='3d')
    ax.text2D(0.1, 0.9, '(b)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)
    ax.set_xlabel(r'$v_1$', fontsize=25)
    ax.set_ylabel(r'$v_2$', fontsize=25)
    ax.set_zlabel(r'$v_3$', fontsize=25)
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([0, 250])
    # ax.set_zlim([-30, 30])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.locator_params(nbins=4)
    # ax.scatter(a0H[], a0H[4], a0H[5], c='r', s=40)
    # ax.plot(aaH[:, -1], aaH[:, 4], aaH[:, 5], c='b', lw=1, alpha=0.7)
    # ax.plot(aapH[:, -1], aapH[:, 4], aapH[:, 5], c='r', lw=2)
    ax.scatter(0, 0, 0, c='r', s=40)
    ax.plot(aaP[:, 0], aaP[:, 1], aaP[:, 2], c='b', lw=1, alpha=0.7)
    ax.plot(aapP[:, 0], aapP[:, 1], aapP[:, 2], c='r', lw=2)

    fig.tight_layout(pad=0)
    plt.show(block=False)

if case == 150:
    """
    plot the example explosions using loaded data
    """
    N, d = 1024, 50
    data = np.load('explosionExample08n06.npz')
    Aamp1 = data['AampSymmetric']
    T1 = data['Tsymmetric']
    Aamp2 = data['AampAsymmetric']
    T2 = data['Tasymmetric']
    
    # plot the figure
    fig = plt.figure(figsize=[8, 5])    

    ax = fig.add_subplot(121)
    ax.text(0.1, 0.9, '(a)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=30, color='white')
    ax.set_xlabel(r'$x$', fontsize=30)
    ax.set_ylabel(r'$t$', fontsize=30)
    ax.set_xticks(range(0, 60, 10))
    ax.set_yticks(range(0, 12, 2))
    ax.tick_params(axis='both', which='major', labelsize=20)
    im = ax.imshow(Aamp1, cmap=plt.get_cmap('jet'), extent=[0, d, 0, T1], 
                   aspect='auto', origin='lower')
    # ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 4, 1))
    cb.ax.tick_params(labelsize=20)


    ax = fig.add_subplot(122)
    ax.text(0.1, 0.9, '(b)', horizontalalignment='center',
            transform=ax.transAxes, fontsize=30, color='white')
    ax.set_xlabel(r'$x$', fontsize=30)
    ax.set_ylabel(r'$t$', fontsize=30)
    ax.set_xticks(range(0, 60, 10))
    ax.set_yticks(range(0, 12, 2))
    ax.tick_params(axis='both', which='major', labelsize=20)
    im = ax.imshow(Aamp2, cmap=plt.get_cmap('jet'), extent=[0, d, 0, T2], 
                   aspect='auto', origin='lower')
    # ax.grid('on')
    dr = make_axes_locatable(ax)
    cax =dr.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 4, 1))
    cb.ax.tick_params(labelsize=20)

    fig.tight_layout(pad=0)
    plt.show(block=False)


if case == 200:
    """
    print out the eigen exponents of Hopt RPO
    """
    Bi, Gi = 1.4, -3.9
    rpo = CQCGLrpo()
    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoHopfBiGi.h5', rpo.toStr(Bi, Gi, 1), flag=2)
    mu = np.log(np.abs(e)) / T
    theta = np.arctan2(e.imag, e.real)
    
if case == 220:
    """
    print out the eigen exponents of rpos
    """
    rpo = CQCGLrpo()
    Bi, Gi = (4.6, -5.0) if False else (4.8, -4.5)
    x, T, nstp, th, phi, err, e, v = rpo.read('../../data/cgl/rpoBiGiEV.h5', rpo.toStr(Bi, Gi, 1), flag=2)
    mu = np.log(np.abs(e)) / T
    theta = np.arctan2(e.imag, e.real)

if case == 240:
    """
    plot the stability exponent diagram
    """
    req = CQCGLreq()
    Bi, Gi = 0.8, -0.6
    
    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=2)
    def Lam(k):
        return -0.1 - (0.125 + 0.5j) * (2*np.pi/50 * k)**2
    # Lam = lambda k : -0.1 - (0.125 + 0.5j) * (2*np.pi/50 * k)**2
    
    M = (e.shape[0] - 2 ) / 4
    x = [0, 0]
    for i in range(1, M+1):
        x = x + [i]*4
    x = np.array(x)

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', which='major', labelsize=25)
    ax.set_xlabel(r'$j$', fontsize=30)
    ax.tick_params(axis='y', which='major', colors='r', labelsize=25)
    ax.set_ylabel(r'$\mu^{(j)}$', color='r', fontsize=30)

    ax.scatter(x, e.real, c='r', s=10)
    ax.plot(x, Lam(x).real, c='k', ls='--', lw=2)

    ax2 = ax.twinx()
    ax2.tick_params(axis='both', which='major', colors='b', labelsize=25)
    ax2.set_ylabel(r'$\omega^{(j)}$', color='b', fontsize=30)

    ax2.scatter(x, np.abs(e.imag), c='b', s=10)
    ax2.plot(x, np.abs(Lam(x).imag), c='k', ls='--', lw=2)
    ax2d(fig, ax)
    
    # fig, ax = pl2d(size=[8, 6], labs=[r'$j$', r'$\omega^{(j)}$'], axisLabelSize=30, tickSize=25)
    # ax.scatter(x, np.abs(e.imag), c='r', s=10)
    # #ax.plot(x, np.abs(e.imag))
    # ax.plot(x, np.abs(Lam(x).imag), c='b', ls='--', lw=2)
    # ax2d(fig, ax)
