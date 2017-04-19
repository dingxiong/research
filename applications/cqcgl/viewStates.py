from py_CQCGL1d import *
from cglHelp import *

case = 55

if case == 1:
    """
    view an ergodic instance
    full state space => reduce continuous symmetry =>
    reduce reflection symmetry
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h,  True, 0, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6, 4)
    Ndim = cgl.Ndim

    A0 = centerRand(2*N, 0.2)
    a0 = cgl.Config2Fourier(A0)
    nstp = 15000
    aa = cgl.intg(a0, nstp, 1)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])

    aaHat, th, phi = cgl.orbit2sliceWrap(aa)
    aaHat2, th2, phi2 = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(aaHat), [0, d, 0, nstp*h])
    plotConfigSpace(cgl.Fourier2Config(aaHat2), [0, d, 0, nstp*h])
    aaTilde = cgl.reduceReflection(aaHat)
    plotConfigSpace(cgl.Fourier2Config(aaTilde), [0, d, 0, nstp*h])

    # rely on numpy's unwrap function
    th3 = unwrap(th*2.0)/2.0
    phi3 = unwrap(phi*2.0)/2.0
    aaHat3 = cgl.rotateOrbit(aa, -th3, -phi3)
    plotConfigSpace(cgl.Fourier2Config(aaHat3), [0, d, 0, nstp*h])

    # rotate by g(pi, pi)
    aaHat4 = cgl.Rotate(aaHat2, pi, pi)
    plotConfigSpace(cgl.Fourier2Config(aaHat4), [0, d, 0, nstp*h])
    aaTilde4 = cgl.reduceReflection(aaHat4)
    plotConfigSpace(cgl.Fourier2Config(aaTilde4), [0, d, 0, nstp*h])

    # reflection
    aa2 = cgl.reflect(aa)
    plotConfigSpace(cgl.Fourier2Config(aa2), [0, d, 0, nstp*h])
    aaHat5, th5, phi5 = cgl.orbit2slice(aa2)
    plotConfigSpaceFromFourier(cgl, aaHat5, [0, d, 0, nstp*h])
    aaTilde5 = cgl.reduceReflection(aaHat5)
    plotConfigSpace(cgl.Fourier2Config(aaTilde5), [0, d, 0, nstp*h])


if case == 2:
    # view relative equlibria
    N = 1024
    d = 30
    di = 0.06

    cgl = pyCQCGL(N, d, 4.0, 0.8, 0.01, di, -1, 4)
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN512.h5', '1')

    vReq = cgl.velocityReq(a0, wth0, wphi0)
    print norm(vReq)

    # check the reflected state
    a0Reflected = cgl.reflect(a0)
    vReqReflected = cgl.velocityReq(a0Reflected, -wth0, wphi0)
    print norm(vReqReflected)
    # plotOneConfigFromFourier(cgl, a0)
    # plotOneConfigFromFourier(cgl, a0Reflected)

    # obtain the stability exponents/vectors
    # eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvalues, eigvectors = eigReq(cgl, a0Reflected, -wth0, wphi0)
    print eigvalues[:10]

    # make sure you make a copy because Fourier2Config takes contigous memory
    tmpr = eigvectors[:, 0].real.copy()
    tmprc = cgl.Fourier2Config(tmpr)
    tmpi = eigvectors[:, 0].imag.copy()
    tmpic = cgl.Fourier2Config(tmpi)
    plotOneConfig(tmprc, size=[6, 4])
    plotOneConfig(tmpic, size=[6, 4])

    nstp = 2000
    aa = cgl.intg(a0Reflected, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h], [0, 2])
    raa, th, phi = cgl.orbit2slice(aa)
    plotConfigSpace(cgl.Fourier2Config(raa), [0, d, 0, nstp*h], [0, 2])
    plotOneFourier(aa[-1])

# test the ve2slice function
if case == 3:
    N = 512
    d = 50
    h = 0.01

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN512.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)

    # plot the last vector to see whether Tcopy works
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    plotOneConfigFromFourier(cgl, eigvectors[-1])

    veHat = cgl.ve2slice(eigvectors, a0)
    print veHat[:4, :8]
    # plotOneConfigFromFourier(cgl, eigvectors[0])
    # print out the norm of two marginal vectors. They should valish
    print norm(veHat[4]), norm(veHat[5])

    # test the angle between each eigenvector
    v1 = veHat[0]
    v2 = veHat[1]
    print np.dot(v1, v2) / norm(v1) / norm(v2)
    plotOneFourier(v1)
    plotOneFourier(v2)

if case == 4:
    """
    test the reflectVe function
    """
    N = 512
    d = 50
    h = 0.01

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    Ndim = cgl.Ndim
    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN512.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)

    a0Hat = cgl.orbit2slice(a0)[0]
    veHat = cgl.ve2slice(eigvectors, a0)
    veTilde = cgl.reflectVe(veHat, a0Hat)

    a0Ref = cgl.reflect(a0)
    veRef = cgl.reflect(eigvectors)
    a0RefHat = cgl.orbit2slice(a0Ref)[0]
    veRefHat = cgl.ve2slice(veRef, a0Ref)
    veRefTilde = cgl.reflectVe(veRefHat, a0RefHat)
    # why the hell is veHat the same as veRefHat ?

if case == 50:
    """
    use the new form of cqcgl
    test the transition with respect to di
    """
    N = 1024
    d = 30
    h = 1e-4
    di = 0.05

    cgl = pyCQCGL1d(N, d, 4.0, 0.8, 0.01, di, -1)
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    nstp = 20000
    x = []
    for i in range(3):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])
        # plotPhase(cgl, aa, [0, d, 0, nstp*h])
        # plotOneConfigFromFourier(cgl, aa[-1], d)
        # plotOnePhase(cgl, aa[-1], d)
        # plot1dfig(aa[:, 0])
        x.append(aa)

if case == 55:
    """
    reproduce the pulstating soliton with L = 50
    """
    N, d = 1024 , 50
    h = 1e-2
    cgl = pyCQCGL1d(N, d, -0.1, 0.08, 0.5, 0.782, 1, -0.1, -0.08, -1)
    rpo, cp = CQCGLrpo(cgl), CQCGLplot(cgl)
    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    a0 = cgl.intgC(a0, h, 100, 1000000)

    T = 70
    aa = cgl.intgC(a0, h, T, 100)
    aa2 = cgl.intg(a0, h, T, 100)
    cp.config(aa, [0, d, 0, T])
    np.savez_compressed('pulsatingSoliton', states=aa, statesAdapt=aa2, Ts=cgl.Ts(), T=T)
        
###############################################################################
if case == 60:
    """
    Try to reproduce the result of  Descalzi.
    =======
    O. Descalzi and H. R. Brand, "Transition from modulated to exploding
    dissipative solitons: Hysteresis, dynamics, and analytic aspects", Phys.
    Rev. E 82, 026203 (2010)
    ======
    But I failed to get the f1 and f1-f2 region and their hysteretic
    transition.
    """
    N = 1024
    d = 30
    h = 1e-4

    Mu = -1
    cgl = pyCQCGL1d(N, d, Mu, 0.125, 0.5, 1, 1, -0.1, -0.6, -1)

    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    a0 = cgl.intg(a0, h, np.int(50/h), np.int(50/h))[-1]

    # stable soliton
    T = 50
    aa = cgl.intg(a0, h, np.int(T/h), 10)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

    # oscillating soltion with one frequency
    cgl.changeMu(-0.21)
    T = 50
    aa = cgl.intg(aa[-1], h, np.int(5/h), np.int(5/h))
    aa = cgl.intg(aa[-1], h, np.int(T/h), 10)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

    # oscillating soltion with two frequency
    cgl.changeMu(-0.19)
    T = 100
    aa = cgl.intg(aa[-1], h, np.int(5/h), np.int(5/h))
    aa = cgl.intg(aa[-1], h, np.int(T/h), 100)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

    # aa = cgl.intg(a0, 0.01, np.int(50/0.01), 10)
    # plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

if case == 70:
    """
    Try to reproduce AKhmediev's paper
    ======
    Strongly asymmetric soliton explosions
    N Akhmediev, JM Soto-Crespo - Physical Review E, 2004 - APS
    =====
    """
    N = 1024
    d = 30
    h = 1e-4

    nu = -0.4
    cgl = pyCQCGL1d(N, d, -0.8, 0.15, 1, 1.4, -0.1, nu, -1)
      
    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    a0 = cgl.intg(a0, h, np.int(50/h), np.int(50/h))[-1]

    T = 500
    aa = cgl.intg(a0, h, np.int(T/h), 100)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

if case == 80:
    """
    Try to reproduce results of Deissler and Brand
    ======
    "Periodic, quasiperiodic, and chaotic localized solutions of the quintic
    complex Ginzburg-Landau equation"
    RJ Deissler, HR Brand - Physical review letters, 1994 - APS
    ======
    """
    N = 1024
    d = 120
    h = 1e-3

    Dr = 0.9
    
    cgl = pyCQCGL1d(N, d, -0.1, Dr, -1.1, 3, 1, -2.75, 1, -1)

    Ndim = cgl.Ndim
    A0 = 3*centerRand(N, 0.1, True)
    a0 = cgl.Config2Fourier(A0)
    a0 = cgl.intg(a0, h, np.int(100/h), np.int(100/h))[-1]

    T = 250
    aa = cgl.intg(a0, h, np.int(T/h), 20)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])

if case == 90:
    """
    Try to reproduce results in
    ======
    Creeping solitons in dissipative systems and their bifurcations
    Wonkeun Chang, Adrian Ankiewicz, Nail Akhmediev, and J. M. Soto-Crespo
    Phys. Rev. E 76, 016607 - Published 26 July 2007
    ======
    """
    N = 1024
    d = 60
    h = 1e-2

    epsilon = 0.844
    cgl = pyCQCGL1d(N, d, -0.1, 0.08, 1, epsilon, -0.11, -0.08, -1)
    cp = CQCGLplot(cgl)
      
    if True:
        Ndim = cgl.Ndim
        A0 = 3*centerRand(N, 0.2, True)
        a0 = cgl.Config2Fourier(A0)
        a0 = cgl.intgC(a0, h, 100, 1000000)

        T = 300
        aa = cgl.intgC(a0, h, T, 100)
        Q = cgl.calQ(aa)
        cp.config(aa, [0, d, 0, T])
        plot1dfig(Q)
        np.save('a0', aa[-1])

    if False:
        cgl.Br = 0.844
        a0 = load('a0.npy')
        a0 = cgl.intg(a0, h, np.int(100/h), np.int(100/h))[-1]
        T = 300
        aa = cgl.intg(a0, h, np.int(T/h), 100)
        plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, T])
        np.save('a1', aa[-1])

    cgl.Br = 0.844
    a0 = np.load('a0.npy')
    a0 = cgl.intgC(a0, h, 100, 1000000)
    T = 100
    aa = cgl.intg(a0, h, T, 20)
    Q = cgl.calQ(aa)
    cp.config(aa, [0, d, 0, T])
    plot1dfig(Q)
    
    
    
    
    
    
