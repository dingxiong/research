from py_CQCGL1dSub import *
from cglHelp import *

################################################################################
#   view the reqs in the symmetric subspace
################################################################################

case = 70

if case == 10:
    """
    use the new data to calculate tyhe stability exponents of req
    """
    N, d = 1024, 50
    Bi, Gi = 2, -5
    index = 1

    cgl = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    
    a0, wth0, wphi0, err0 = req.read('../../data/cgl/reqsubBiGi.h5',
                                     req.toStr(Bi, Gi, index),
                                     sub=True)
    e, v = req.eigReq(a0, wth0, wphi0, sub=True)
    print e[:10]


if case == 20:
    """
    visualize the eigenvectors
    """
    N, d = 1024, 50
    Bi, Gi = 2, -2
    index = 1

    cgl = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)
    
    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqsubBiGiEV.h5',
                                           req.toStr(Bi, Gi, index),
                                           sub=True,flag=2)
    print e[:10]
    cp.oneConfig(v[0].real * cgl.N)
    cp.oneConfig(v[0].imag * cgl.N)

if case == 30:
    """
    check the accuracy of all req in symmetric subspace
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6
    index = 1

    cgl = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)
    fin = '../../data/cgl/reqsubBiGi.h5'
    
    gs = req.scanGroup(fin, 2)
    errs = []
    for item in gs:
        a0, wth0, wphi0, err0 = req.read(fin, item, sub=True)
        errs.append(err0)

    print np.max(np.abs(errs))

if case == 40:
    """
    save the data
    """
    req = CQCGLreq()
    fin = '../../data/cgl/reqsubBiGiEV.h5'
    index = 1
    
    gs = req.scanGroup(fin, 2)
    saveData = np.zeros([0, 3])

    for item in gs:
        param = item.split('/')
        Bi, Gi, index = np.double(param[0]), np.double(param[1]), np.int(param[2])
        if index == 1:
            a0, wth0, wphi0, err0, e, v = req.read(fin, item, sub=True, flag=2)
            m, ep, accu = numStab(e)
            saveData = np.vstack((saveData, np.array([Gi, Bi, m])))

    np.savetxt("BiGiReqSubStab.dat", saveData)

if case == 50:
    """
    plot the Bi - Gi req stability plot using the saved date from running viewReq.py
    data format : {Gi, Bi, #unstable}
    #unstalbe is 0, 2, 4, 6 ...
    """
    saveData = np.loadtxt("BiGiReqSubStab.dat")
    cs = {0: 'c', 1: 'm', 2: 'g', 3: 'r', 4: 'b'}#, 8: 'y', 56: 'r', 14: 'grey'}
    ms = {0: '^', 1: '^', 2: 'x', 3: 'o', 4: 'D'}#, 8: '8', 56: '4', 14: 'v'}
    fig, ax = pl2d(size=[8, 6], labs=[r'$\beta_i$', r'$\gamma_i$'], axisLabelSize=30,
                   tickSize=20, ylim=[-6, 0], xlim=[-4, 6])
    for item in saveData:
        Gi, Bi, m = item
        ax.scatter(Bi, Gi, s=18 if m > 0 else 18, edgecolors='none',
                   marker=ms.get(m, 'v'), c=cs.get(m, 'k'))
    ax2d(fig, ax)



if case == 70:
    """
    same as case 10, bu plot contourf plot
    Note: the problem of contourf() is that whenever there is a jump
    in the levels, say from lv1 to lv2 but lv2 = lv1 + 2, then it will
    interpolate these two levels, thus a line for lv1 + 1 is plotted
    between these two regions.
    I need to draw each region indivisually. 
    """
    saveData = np.loadtxt("BiGiReqSubStab.dat")
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
            Z[idy, idx] = m if m < 6 else 6

    levels = [-1, 0, 2, 4, 6]
    cs = ['w', 'm', 'c', 'r', 'y']
    fig, ax = pl2d(size=[8, 6], labs=[r'$\beta_i$', r'$\gamma_i$'], axisLabelSize=30,
                   tickSize=20)
    for i in range(len(levels)):
        vfunc = np.vectorize(lambda t : 0 if t == levels[i] else 1)
        Z2 = vfunc(Z)
        ax.contourf(X, Y, Z2, levels=[-0.1, 0.9], colors=cs[i])
    ax.text(0, -2, r'$S$', fontsize=30)
    ax.text(0, -4.5, r'$U_1$', fontsize=30)
    ax.text(2., -4.8, r'$U_2$', fontsize=30)
    ax.text(3., -3, r'$U_{\geq 3}$', fontsize=30)
    ax.scatter(2.0,-5, c='k', s=100, edgecolor='none')
    ax.scatter(2.7,-5, c='k', s=100, edgecolor='none')
    ax.scatter(1.4,-3.9, c='k', marker='*', s=200, edgecolor='none')
    ax.set_xlim(BiRange)
    ax.set_ylim(GiRange)
    ax2d(fig, ax)
