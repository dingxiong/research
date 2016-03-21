from personalFunctions import *
from py_ks import *

case = 20

if case == 10:
    """
    test ks.stab() function
    """
    N = 32
    d = 22
    h = 0.1
    ks = pyKS(N, h, d)
    
    a0 = KSreadEq('/usr/local/home/xiong/00git/research/data/ksReqx32.h5', 3)
    print norm(ks.velocity(a0))

    A = ks.stab(a0)
    e, v = eig(A.T)
    idx = argsort(e.real)[::-1]
    e = e[idx]
    v = v[:, idx]
    print e


def rSO(ks, aa):
    m, n = aa.shape
    raa = np.zeros((m, n))
    ths = np.zeros(m)

    phi = np.zeros((m, 4))
    r2 = np.zeros((m, 4))
    for i in range(m):
        phi[i, 0] = np.angle(aa[i, 0] + 1j*aa[i, 1])
        phi[i, 1] = np.angle(aa[i, 2] + 1j*aa[i, 3])
        phi[i, 2] = np.angle(aa[i, 4] + 1j*aa[i, 5])
        phi[i, 3] = np.angle(aa[i, 6] + 1j*aa[i, 7])
        
        r2[i, 0] = aa[i, 0]**2 + aa[i, 1]**2
        r2[i, 1] = aa[i, 2]**2 + aa[i, 3]**2
        r2[i, 2] = aa[i, 4]**2 + aa[i, 5]**2
        r2[i, 3] = aa[i, 6]**2 + aa[i, 7]**2

    for i in range(4):
        phi[:, i] = np.unwrap(phi[:, i])

    for i in range(m):
        theta = - r2[i].dot(phi[i]) / r2[i].dot(np.arange(1, 5))
        ths[i] = theta
        raa[i] = ks.Rotation(aa[i], theta)

    return raa, ths


def rSO2(ks, aa):
    m, n = aa.shape
    raa = np.zeros((m, n))
    ths = np.zeros(m)

    k = 4
    bb = np.zeros((m, k)) + 1j * np.zeros((m, k))
    bb[:, 0] = aa[:, 0] + 1j*aa[:, 1]
    for i in range(k-1):
        bb[:, i+1] = (aa[:, 2*i+2] + 1j*aa[:, 2*i+3]) *  (aa[:, 2*i] - 1j*aa[:, 2*i+1])
    A = np.sum(bb.real, axis=1)
    B = np.sum(bb.imag, axis=1)

    for i in range(m):
        theta = arctan2(-B[i], A[i])
        ths[i] = theta
        raa[i] = ks.Rotation(aa[i], theta)
    
    a1 = np.sqrt(aa[:, 0]**2 + aa[:, 1]**2)
    a2 = np.sqrt(aa[:, 2]**2 + aa[:, 3]**2)
    a3 = np.sqrt(aa[:, 4]**2 + aa[:, 5]**2)
    a4 = np.sqrt(aa[:, 6]**2 + aa[:, 7]**2)
    plot1dfig(a1, yscale='log')
    plot1dfig(a1*a2, yscale='log')
    plot1dfig(a2*a3, yscale='log')
    plot1dfig(a3*a4, yscale='log')
    plot1dfig(A**2+B**2, yscale='log')
    return raa, ths

if case == 20:
    """
    Have a look at the flow in the state space
    """
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)
    
    a0, w, err = KSreadReq('../../data/ks22Reqx64.h5', 1)
    es, vs = KSstabReqEig(ks, a0, w)
    aa = ks.intg(a0 + 1e-1*vs[0].real, 200000, 100)
    aaH = ks.orbitToSlice(aa)[0]
    # plot3dfig(aa[:, 0], aa[:, 3], aa[:, 2])
    plot3dfig(aaH[:, 0], aaH[:, 3], aaH[:, 2])
    raa, ths = rSO2(ks, aa)
    plot3dfig(raa[:, 0], raa[:, 3], raa[:, 2])

if case == 30:
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req = np.zeros((N-2, 2))
    for i in range(2):
        a0, w, err = KSreadReq('../../data/ks22Reqx64.h5', i+1)
        req[i] = rSO(ks, a0)[0]

    eq = np.zeros((N-2, 2))
    for i in range(3):
        a0, err = KSreadEq('../../data/ks22Reqx64.h5', i+1)
        eq[i] = rSO(ks, a0)[0]

    a0 = 0.1 * rand(N-2)
    aa = ks.intg(a0, 100000, 100)
    raa, ths = rSO(ks, aa)
    plot3dfig(raa[:, 0], raa[:, 3], raa[:, 2])

