from personalFunctions import *
from py_ks import *

case = 30

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

    return raa, ths


def rSO3(ks, aa):
    m, n = aa.shape
    raa = np.zeros((m, n))
    ths = np.zeros(m)

    i = 3
    bb = np.zeros(m) + 1j * np.zeros(m)
    bb = aa[:, 0] + 1j*aa[:, 1] + (
        aa[:, 2*i] + 1j*aa[:, 2*i+1]) * (aa[:, 2*i-2] - 1j*aa[:, 2*i-1])
    A = bb.real
    B = bb.imag

    for i in range(m):
        theta = arctan2(-B[i], A[i])
        ths[i] = theta
        raa[i] = ks.Rotation(aa[i], theta)
    
    # plot1dfig(A**2+B**2, yscale='log')
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
    # aa = ks.intg(a0 + 1e-1*vs[0].real, 200000, 100)
    aa = ks.intg(rand(N-2)*0.1, 200000, 100)
    aaH = ks.orbitToSlice(aa)[0]
    # plot3dfig(aa[:, 0], aa[:, 3], aa[:, 2])
    plot3dfig(aaH[:, 0], aaH[:, 3], aaH[:, 2])
    raa, ths = rSO3(ks, aa)
    plot3dfig(raa[:, 0], raa[:, 3], raa[:, 2])

if case == 30:
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)

    req = np.zeros((2, N-2))
    ws = np.zeros(2)
    reqr = np.zeros((2, N-2))
    for i in range(2):
        a0, w, err = KSreadReq('../../data/ks22Reqx64.h5', i+1)
        req[i] = a0
        ws[i] = w
        tmp = ks.redSO2(a0)
        reqr[i] = tmp[0]
        print tmp[1]
        
    eq = np.zeros((3, N-2))
    eqr = np.zeros((3, N-2))
    for i in range(3):
        a0, err = KSreadEq('../../data/ks22Reqx64.h5', i+1)
        eq[i] = a0
        tmp = ks.redSO2(a0)
        eqr[i] = tmp[0]
        print tmp[1]

    k = 0
    es, ev = KSstabEig(ks, eq[k])
    print es[:8]
    aas = []
    nn = 2
    for i in range(nn):
        a0 = eq[k] + 1e-4 * (i+1) * ev[2].real
        aa = ks.intg(a0, 250000, 100)
        raa, ths = ks.redSO2(aa)
        aas.append(raa)

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection='3d')
    i1 = 2
    i2 = 3
    i3 = 6
    c1 = ['r', 'b']
    for i in range(2):
        ax.scatter(reqr[i, i1], reqr[i, i2], reqr[i, i3], c=c1[i], label='TW'+str(i+1))
    c2 = ['c', 'g', 'k']
    for i in range(3):
        ax.scatter(eqr[i, i1], eqr[i, i2], eqr[i, i3], c=c2[i], label='E'+str(i+1))
        
    ns = -100
    for i in range(nn):
        ax.plot(aas[i][:ns, i1], aas[i][:ns, i2], aas[i][:ns, i3])
        # ax.scatter(aas[i][-1, 0], aas[i][-1, 3], aas[i][-1, 4])
    fig.tight_layout(pad=0)
    plt.legend()
    plt.show(block=False)

    
if case == 40:
    N = 64
    d = 22
    h = 0.001
    ks = pyKS(N, h, d)
    
    a0, w, err = KSreadReq('../../data/ks22Reqx64.h5', 1)
    es, vs = KSstabReqEig(ks, a0, w)
    # aa = ks.intg(a0 + 1e-1*vs[0].real, 200000, 100)
    aa = ks.intg(rand(N-2)*0.1, 200000, 100)
    aaH = ks.orbitToSlice(aa)[0]
    # plot3dfig(aa[:, 0], aa[:, 3], aa[:, 2])
    plot3dfig(aaH[:, 0], aaH[:, 3], aaH[:, 2])
    raa, ths = rSO3(ks, aa)
    plot3dfig(raa[:, 0], raa[:, 3], raa[:, 2])
