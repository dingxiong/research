from cglHelp import *
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})  # I love Times
rc('text', usetex=True)

case = 90


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
                   tickSize=25)
    for i in range(len(levels)):
        vfunc = np.vectorize(lambda t : 0 if t == levels[i] else 1)
        Z2 = vfunc(Z)
        ax.contourf(X, Y, Z2, levels=[-0.1, 0.9], colors=cs[i])

    # add zoom rectangle
    BiZoom = [1.9, 4]
    GiZoom = [-5.6, -4]
    ax.plot([BiZoom[0], BiZoom[0]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[1], BiZoom[1]], [GiZoom[0], GiZoom[1]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[0], GiZoom[0]], ls='--', lw=3, c='k')
    ax.plot([BiZoom[0], BiZoom[1]], [GiZoom[1], GiZoom[1]], ls='--', lw=3, c='k')

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

    ax.set_xticks(range(-3, 5))
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
    BiZoom = [1.9, 4]
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
    ax.set_yticks([-4, -4.5, -5, -5.5])
    ax.set_xticks(np.arange(2, 6, 0.5))
    ax2d(fig, ax)


if case == 90:
    """
    plot the two coexisting soltions for the same paramter in the same figure
    """
    N, d = 1024, 50
    
    fig, ax = pl2d(size=[8, 6], labs=[r'$x$', r'$|A|$'], axisLabelSize=30, tickSize=25)

    Bi, Gi = 0.8, -0.6
    req = CQCGLreq()
    a0, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 1), flag=0)
    Aamp = np.abs(cgl.Fourier2Config(a0))
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2, ls='-', c='k')

    a0, wth0, wphi0, err0 = req.read('../../data/cgl/reqBiGiEV.h5', req.toStr(Bi, Gi, 2), flag=0)
    Aamp = np.abs(cgl.Fourier2Config(a0))
    ax.plot(np.linspace(0, d, Aamp.shape[0]), Aamp, lw=2, ls='--', c='k')
    
    ax2d(fig, ax)
