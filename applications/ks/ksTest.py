from personalFunctions import *
from py_ks import *

case = 10

if case == 10:
    N, L = 64, 22
    ks = pyKS(N, L)
    t_init = time();
    a0 = np.ones(N-2)*0.1
    for i in range(100):
        aa = ks.intgC(a0, 0.01, 30, 10)
    print time() - t_init
    t_init = time()
    for i in range(1):
        aa, daa = ks.intgjC(a0, 0.01, 30, 100000)
    print time() - t_init

if case == 15:
    ks = pyKSM1(32, 0.1, 22)
    t_init = time()
    for i in range(1000):
        a0 = np.ones(30)*0.1
        a0[1] = 0
        aa, tt = ks.intg(a0, 2000, 1)
    print time() - t_init

    t_init = time()
    for i in range(10):
        a0 = np.ones(30)*0.1
        a0[1] = 0
        aa, tt = ks.intg2(a0, 20, 1)
    print time() - t_init

if case == 20:
    ks = pyKS(32, 0.1, 22)
    aa = ks.intg(np.ones(30)*0.1, 20, 1)
    aaHat, ang = ks.orbitToSlice(aa)
    aaTilde = ks.reduceReflection(aaHat)

    # check p2
    print np.sqrt(aaHat[:, 2]**2 + aaHat[:, 5]**2)
    print aaTilde[:, 2]
    # check the last transformed term
    print aaHat[:, 26] * aaHat[:, 29] / np.sqrt(aaHat[:, 26]**2 + aaHat[:, 29]**2)
    print aaTilde[:, 29]
    # check the unchanged terms
    print aaHat[:, 3]
    print aaTilde[:, 3]

    x = 0.1*np.arange(30)
    ve = np.sin(range(30))
    veTilde = ks.reflectVe(ve, x)

if case == 30:
    """
    test the redSO2 and fundDomain() functions
    """
    N = 64
    d = 22
    ks = pyKS(N, d)
    
    a0 = rand(N-2)
    aa = ks.aintg(a0, 0.01, 100, 1000000)
    aa = ks.aintg(aa[-1], 0.01, 200, 1)
    plot3dfig(aa[:, 0], aa[:, 1], aa[:, 2])

    raa1, ang1 = ks.redSO2(aa, 1, True)
    faa1, dids1 = ks.fundDomain(raa1, 2)
    print norm(raa1[:, 0])
    plot3dfig(raa1[:, 3], raa1[:, 1], raa1[:, 2])
    plot3dfig(faa1[:, 3], faa1[:, 1], faa1[:, 2])
    plot1dfig(dids1)
    
    raa2, ang2 = ks.redSO2(aa, 2, True)
    faa2, dids2 = ks.fundDomain(raa2, 1)
    print norm(raa2[:, 2])
    plot3dfig(raa2[:, 0], raa1[:, 1], raa1[:, 3])
    plot3dfig(faa2[:, 0], faa2[:, 1], faa2[:, 3])
    plot1dfig(dids2)

if case == 40:
    """
    test the 2nd slice
    """
    N = 64
    d = 22
    ks = pyKS(N, d)

    x = rand(N-2)-0.5
    y = ks.Reflection(x)
    th0 = np.pi*2.47
    th1 = np.pi*4.22
    x1, t1 = ks.redO2(x)
    x2, t2 = ks.redO2(ks.Rotation(x, th0))
    y1, r1 = ks.redO2(y)
    y2, r2 = ks.redO2(ks.Rotation(y, th1))
    print norm(x2-x1), norm(y2-y1), norm(y2-x2)

if case == 50:
    """
    test the 2nd slice with fundamental domain
    """
    N = 64
    d = 22
    ks = pyKS(N, d)

    x = rand(N-2)-0.5
    y = ks.Reflection(x)
    th0 = np.pi*2.47
    th1 = np.pi*4.22
    x1, id1, t1 = ks.redO2f(x, 2)
    x2, id2, t2 = ks.redO2f(ks.Rotation(x, th0), 2)
    y1, id3, r1 = ks.redO2f(y, 2)
    y2, id4, r2 = ks.redO2f(ks.Rotation(y, th1), 2)
    print norm(x2-x1), norm(y2-y1), norm(y2-x2)

if case == 60:
    """
    test ks.stab() function
    """
    N = 64
    d = 22
    ks = pyKS(N, d)

    fileName = '/usr/local/home/xiong/00git/research/data/ks22Reqx64.h5'
    a0, err = KSreadEq(fileName, 3)
    print norm(ks.velocity(a0))

    A = ks.stab(a0)
    e, v = eig(A.T)
    idx = argsort(e.real)[::-1]
    e = e[idx]
    v = v[:, idx]
    print e

if case == 70:
    """
    visualize heat map of ppo/rpo
    """
    N = 64
    L = 22
    ks = pyKS(N, L)
    
    fileName = '/usr/local/home/xiong/00git/research/data/ks22h001t120x64EV.h5'
    poType = 'ppo'
    poId = 2
    
    KSplotPoHeat(ks, fileName, poType, poId, NT=2, Ts=100, fixT=True)

if case == 80:
    """
    print the period
    """
    fileName = '/usr/local/home/xiong/00git/research/data/ks22h001t120x64EV.h5'
    Ts = []
    for poId in range(1, 31):
        a0, T, nstp, r, s = KSreadPO(fileName, 'rpo', poId)
        print '%0.2f\n' % T,
        Ts.append(T)

    
