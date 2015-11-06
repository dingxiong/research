from py_cqcgl1d_threads import pyCqcgl1d
from personalFunctions import *

case = 20

if case == 10:
    """
    calculate the stability exponents of req
    of different di.
    """
    N = 1024
    d = 30
    h = 0.0002
    
    di = -0.0799
    a0, wth0, wphi0, err = cqcglReadReq('req0799.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, di, 4)
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    print eigvalues[:10]

if case == 11:
    """
    calculate the stability exponents and 10 leading vectors
    save them into the database
    """
    dis = []
    
if case == 20:
    """
    Try to locate the Hopf bifurcation limit cycle.
    There is singularity for reducing the discrete symmetry
    """
    N = 1024
    d = 30
    h = 0.0002
    di = -0.0799
    a0, wth0, wphi0, err = cqcglReadReq('req0799.h5', '1')
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, di, 4)

    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    eigvectors = realve(eigvectors)
    eigvectors = Tcopy(eigvectors)
    a0Hat = cgl.orbit2slice(a0)[0]
    a0Tilde = cgl.reduceReflection(a0Hat)
    veHat = cgl.ve2slice(eigvectors, a0)
    # veTilde = cgl.reflectVe(veHat, a0Hat)
    
    nstp = 20000
    a0Erg = a0 + eigvectors[0]*1e-3
    for i in range(10):
        aaErg = cgl.intg(a0Erg, nstp, 1)
        a0Erg = aaErg[-1]
        
    aaErgHat, th, phi = cgl.orbit2slice(aaErg)
    # aaErgTilde = cgl.reduceReflection(aaErgHat)
    # aaErgTilde -= a0Tilde
    aaErgHat -= a0Hat

    # e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])
    e1, e2 = orthAxes(veHat[0], veHat[1])
    # aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)
    aaErgHatProj = np.dot(aaErgHat, np.vstack((e1, e2)).T)
    
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    # ax.plot(aaErgTildeProj[:, 0], aaErgTildeProj[:, 1],
    #         aaErgTildeProj[:, 2], c='r', lw=1)
    ax.plot(aaErgHatProj[:, 0], aaErgHatProj[:, 1],
            c='r', lw=1)
    ax.scatter([0], [0], s=160)
    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    fig.tight_layout(pad=0)
    plt.show(block=False)
