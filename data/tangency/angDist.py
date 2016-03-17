import os
from py_ks import *
from personalFunctions import *

##################################################


def add_subplot_axes(ax, rect, axisbg='w'):
    """
    function to add embedded plot
    rect = [ x of left corner, y of left corner, width, height ] in
    percentage
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def setAxis(ax):
    # ax.set_xlabel(r'$\theta$', fontsize=20, labelpad=-40)
    ax.set_xticks([0., .125*pi, 0.25*pi, 0.375*pi, 0.5*pi])
    ax.set_xticklabels(["$0$", r"$\frac{1}{8}\pi$", r"$\frac{1}{4}\pi$",
                        r"$\frac{3}{8}\pi$", r"$\frac{1}{2}\pi$"],
                       fontsize=15)
    # ax.legend(loc='best', fontsize=12)
    ax.set_yscale('log')
    # ax.set_ylim([0.01, 100])
    # ax.text(0.02, 20, r'$\rho(\theta)$', fontsize=20)
    # ax.set_ylabel(r'$\rho(\theta)$', fontsize=20, labelpad=-55)


def filterAng(a, ns, angSpan, angNum):
    """
    filter the bad statistic points
    because a very small fraction of Floquet vectors corresponding
    to high Fourier modes are not well told apart, so there are
    a small fraction of misleading close to zeros angles for
    non-physical space.
    """
    for i in range(29):
        n = a[0].shape[0]
        for j in range(n):
            y = a[i][j]*ns/(angSpan[i]*angNum[i])
            if y < 2e-2:
                a[i][j] = 0


situation = 4

if situation == 1:
    ##################################################
    # load data
    ixRange = range(0, 29)
    N = len(ixRange)

    folder = './anglePOs64/ppo/space/'
    folder2 = './anglePOs64/rpo/space/'
    ns = 1000

    # collect the data with rpo, ppo combined
    angs = []
    for i in range(N):
        print i
        as1 = np.empty(0)
        as2 = np.empty(0)
        for j in range(1, 201):
            f1 = folder + str(j) + '/ang' + str(i) + '.dat'
            if os.stat(f1).st_size > 0:
                ang = arccos(loadtxt(f1))
                as1 = np.append(as1, ang)

            f2 = folder2 + str(j) + '/ang' + str(i) + '.dat'
            if os.stat(f2).st_size > 0:
                ang2 = arccos(loadtxt(f2))
                as2 = np.append(as2, ang2)

        angs.append(np.append(as1, as2))
        # angs.append(as2)
        # angs.append(as1)

    ##################################################

    case = 1

    if case == 1:
        """
        angle distribution in ppo and rpo
        """
        a = []
        b = []
        angNum = []
        angSpan = []
        for i in range(N):
            print i
            angNum.append(angs[i].shape[0])
            angSpan.append(max(angs[i])-min(angs[i]))
            at, bt = histogram(angs[i], ns)
            a.append(at)
            b.append(bt)

        labs = ['k='+str(i+1) for i in ixRange]

        np.savez_compressed('ab', a=a, b=b, ns=ns, angSpan=angSpan,
                            angNum=angNum)

        fig = plt.figure(figsize=(4, 1.5))
        ax = fig.add_subplot(111)
        for i in range(7):
            ax.plot(b[i][:-1], a[i]*ns/(angSpan[i]*angNum[i]), label=labs[i],
                    lw=1.5)
        setAxis(ax)
        plt.tight_layout(pad=0)
        plt.show(block=False)

        fig = plt.figure(figsize=(4, 1.5))
        ax = fig.add_subplot(111)
        colors = cm.rainbow(linspace(0, 1, 11))
        for ix in range(11):
            i = 7 + 2*ix
            ax.plot(b[i][:-1], a[i]*ns/(angSpan[i]*angNum[i]), c=colors[ix], lw=1.5)
        setAxis(ax)
        plt.tight_layout(pad=0)
        plt.show(block=False)

    if case == 2:
        """
        try to locate the resource of bad distribution at the bottom
        """
        a = []
        b = []
        angNum = []
        angSpan = []

        for i in range(N):
            if angs[i].shape[0] > 0:
                angNum.append(angs[i].shape[0])
                angSpan.append(max(angs[i])-min(angs[i]))
                at, bt = histogram(angs[i], ns)
                a.append(at)
                b.append(bt)

        labs = ['k='+str(i+1) for i in ixRange]

        fig = plt.figure(figsize=(4, 1.5))
        ax = fig.add_subplot(111)
        for i in range(len(a)):
            ax.plot(b[i][:-1], a[i]*ns/(angSpan[i]*angNum[i]), label=labs[i],
                    lw=1.5)
        setAxis(ax)
        plt.tight_layout(pad=0)
        plt.show(block=False)

    if case == 20:
        a = []
        b = []
        angNum = []
        angSpan = []
        for i in range(N):
            print i
            angNum.append(angs[i].shape[0])
            angSpan.append(max(angs[i])-min(angs[i]))
            at, bt = histogram(log(angs[i]), ns)
            a.append(at)
            b.append(bt)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        colors = cm.rainbow(linspace(0, 1, N))
        #labs = ['(1-' + str(i+1) + ',' + str(i+2) + '-30)' for i in ixRange]
        labs = ['(' + str(i+1) + ',' + str(i+2) + ')' for i in ixRange]
        labx = [pi/16, pi/8*1.1, pi/8*0.9, pi/8*3*0.9, pi/2*0.8]
        laby = [0.9, 0.02, 0.1, 0.05, 0.005]

        ax.set_xlabel(r'$\theta$', size='large')
        ax.set_xticks([0., .125*pi, 0.25*pi, 0.375*pi])
        ax.set_xticklabels(["$0$", r"$\frac{1}{8}\pi$", r"$\frac{2}{8}\pi$",
                            r"$\frac{3}{8}\pi$"])

        for i in range(9):
            ax.scatter(b[i][:-1], a[i]*ns/(angSpan[i]*angNum[i]), s=7,
                       c=colors[i], edgecolor='none', label=labs[i])

        ax.legend(fontsize='small', loc='upper center', ncol=1,
                  bbox_to_anchor=(1.1, 1), fancybox=True)
        ax.set_yscale('log')
        ax.set_ylim([0.001, 1000])
        ax.set_ylabel(r'$\rho(\theta)$', size='large')

        plt.tight_layout(pad=0)
        plt.show()

    if case == 30:
        nstps = []
        for i in range(200):
            a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', 'rpo', i+1)
            nstps.append(nstp)
        nstps = np.array(nstps)


if situation == 2:
    """
    pick out the orbit with smallest angle
    """
    ixRange = range(0, 29)
    N = len(ixRange)

    folder = './anglePOs64/ppo/space/'
    folder2 = './anglePOs64/rpo/space/'
    ns = 1000

    # collect the data with rpo, ppo combined
    as1 = []
    as2 = []
    n1 = []
    n2 = []
    i = 1
    for j in range(1, 201):
        f1 = folder + str(j) + '/ang' + str(i) + '.dat'
        if os.stat(f1).st_size > 0:
            ang = arccos(loadtxt(f1))
            as1 = np.append(as1, ang)
            n1.append(ang.shape[0])
        else:
            n1.append(0)

        f2 = folder2 + str(j) + '/ang' + str(i) + '.dat'
        if os.stat(f2).st_size > 0:
            ang2 = arccos(loadtxt(f2))
            as2 = np.append(as2, ang2)
            n2.append(ang2.shape[0])
        else:
            n2.append(0)

    ma1 = np.min(as1)
    ma2 = np.min(as2)
    ix1 = np.argmin(as1)
    ix2 = np.argmin(as2)
    print ma1, ma2, ix1, ix2
    
    s = 0
    for i in range(len(n1)):
        s += n1[i]
        if s >= ix1:
            print i, s
            break

    mix = i + 1
    i1 = np.sum(n1[:i])
    i2 = np.sum(n1[:i+1])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(as1[i1:i2], lw=1.5)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\theta$')
    plt.tight_layout(pad=0)
    plt.show(block=False)

if situation == 3:
    """
    After we pick out the smallest angle orbit, let us
    plot the distribution
    """
    ixRange = range(0, 29)
    N = len(ixRange)

    folder = './anglePOs64/ppo/space/147'
    ns = 1000

    # collect the data with rpo, ppo combined
    as1 = []

    for i in range(N):
        f1 = folder + '/ang' + str(i) + '.dat'
        if os.stat(f1).st_size > 0:
            ang = arccos(loadtxt(f1))
            as1.append(ang)
        else:
            as1.append(np.array([]))
            
    a = []
    b = []
    for i in range(N):
        at, bt = histogram(as1[i], ns)
        # print as1[i].shape, at.shape, bt.shape
        a.append(at)
        b.append(bt)
    
    labs = ['k='+str(i+1) for i in ixRange]
    fig = plt.figure(figsize=(4, 1.5))
    ax = fig.add_subplot(111)
    for i in range(7):
        ax.plot(b[i][:-1], a[i], label=labs[i],
                lw=1.5)
    setAxis(ax)
    plt.tight_layout(pad=0)
    plt.show(block=False)

    fig = plt.figure(figsize=(4, 1.5))
    ax = fig.add_subplot(111)
    colors = cm.rainbow(linspace(0, 1, 11))
    for ix in range(11):
        i = 7 + 2*ix
        ax.plot(b[i][:-1], a[i], c=colors[ix], lw=1.5)
    setAxis(ax)
    plt.tight_layout(pad=0)
    plt.show(block=False)

    
if situation == 4:
    """
    There seems periodic peaks on top of the angle
    for ppo 147, we want to find out the reason
    """
    poType = 'ppo'
    poId = 147
    a0, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', poType, poId)
    h = T / nstp
    ks = pyKS(64, h, 22)
    aa = ks.intg(a0, nstp, 5)
    # aa = ks.reduceReflection(ks.orbitToSlice(aa)[0])

    peaks = [180, 1485, 3701, 6243, 8980, 11262, 13869, 16541]
    colors = plt.cm.winter(np.linspace(0, 1, len(peaks)))
    i1 = 0
    i2 = 2
    i3 = 3
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aa[:, i1], aa[:, i2], aa[:, i3], c='r', lw=2)
    for i, c in zip(peaks, colors):
        ax.scatter(aa[i, i1], aa[i, i2], aa[i, i3], marker='o', c=c,
                   edgecolor='none', s=50)
    ax.set_xlabel(r'$b_1$', fontsize=30)
    ax.set_ylabel(r'$b_2$', fontsize=30)
    ax.set_zlabel(r'$c_2$', fontsize=30)
    fig.tight_layout(pad=0)
    plt.show(block=False)


def getDet(poType, poId, k=30):
    fv = KSreadFV('../ks22h001t120x64EV.h5', poType, poId)
    n, m = fv.shape

    ds = np.zeros(n)
    for i in range(n):
        x = np.reshape(fv[i], (30, m/30))
        x = x[:k, :k]
        ds[i] = LA.det(x)

    return ds


def getDetK(poType, poId):
    dss = []
    for k in range(1, 8) + range(8, 31, 2):
        # print k
        ds = getDet(poType, poId, k)
        dss.append(ds)

    return np.array(dss)


if situation == 5:
    """
    try to see whether the determiant of Floquet vector matrix change sign
    for a specific orbit
    """
    poType = 'ppo'
    for i in range(146, 147):
        poId = i+1
        ds = getDet(poType, poId, 4)
        print poId, np.sign(np.max(ds)) * np.sign(np.min(ds))

    plot1dfig(abs(ds), yscale='log')

if situation == 6:
    """
    try to plot the determinant of the leading k block.
    """
    poType = 'ppo'
    poId = 147
    dss = getDetK(poType, poId)
    m, n = dss.shape
    ers = np.zeros(m-1)
    for i in range(m-1):
        ers[i] = norm(dss[i]-dss[-1])

    plot1dfig(ers, yscale='log')
    plot1dfig(abs(dss[3]), yscale='log', labs=[None, '|det|'],
              axisLabelSize=15, size=[6, 4])

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    ax.plot(abs(dss[2]), lw=1, label=r'$k=3$')
    ax.plot(abs(dss[5]), lw=2, ls='--', label=r'$k=6$')
    ax.plot(abs(dss[6]), lw=2, ls='-.', label=r'$k=7$')
    # ax.plot(abs(dss[7]), lw=1, label=r'$k=8$')
    # ax.plot(abs(dss[12]), lw=2, ls='--', label=r'$k=18$')
    # ax.plot(abs(dss[18]), lw=2, ls='-.', label=r'$k=30$')
    ax.set_ylabel('|det|', fontsize=15)
    ax.set_yscale('log')
    fig.tight_layout(pad=0)
    plt.legend(loc='best')
    plt.show(block=False)
