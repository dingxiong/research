import h5py
from pylab import *
import numpy as np
from numpy.linalg import norm, eig
from time import time
from py_cqcgl1d_threads import pyCqcgl1d, pyCqcglRPO
from personalFunctions import *

case = 3

if case == 1:
    """
    use poincare section to find rpos
    """
    N = 512
    d = 50
    h = 0.001
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    Ndim = cgl.Ndim

    a0, wth0, wphi0, err = cqcglReadReq('../../data/cgl/reqN512.h5', '1')
    eigvalues, eigvectors = eigReq(cgl, a0, wth0, wphi0)
    ve = Tcopy(realve(eigvectors))
    a0Tilde = cgl.reduceAllSymmetries(a0)[0]
    veTilde = cgl.reduceVe(ve, a0)
    e1, e2, e3 = orthAxes(veTilde[0], veTilde[1], veTilde[6])

    nstp = 70000
    a0Erg = a0 + ve[0]*1e-2
    totalPoints = np.zeros((0, 2))
    globalIndex = np.zeros((0,), dtype=np.int)
    originalPoints = np.zeros((0, Ndim))
    for i in range(1):
        aaErg = cgl.intg(a0Erg, nstp, 1)
        aaErgTilde = cgl.reduceAllSymmetries(aaErg)[0] - a0Tilde
        aaErgTildeProj = np.dot(aaErgTilde, np.vstack((e1, e2, e3)).T)

        plotConfigSpace(cgl.Fourier2Config(aaErg),
                        [0, d, nstp*h*i, nstp*h*(i+1)])
        points, index = PoincareLinearInterp(aaErgTildeProj, getIndex=True)
        totalPoints = np.vstack((totalPoints, points))
        globalIndex = np.append(globalIndex, index+i*nstp)
        originalPoints = np.vstack((originalPoints, aaErg[index]))
        a0Erg = aaErg[-1]

    scatter2dfig(totalPoints[:, 0], totalPoints[:, 1])

    Nsteps = (globalIndex[180] - globalIndex[0])/10
    Nps = Nsteps * 10
    aa = cgl.intg(originalPoints[0], Nps, 1)
    aaTilde, th, phi = cgl.reduceAllSymmetries(aa)
    print norm(aaTilde[0] - aaTilde[-1]), th[0] - th[-1], phi[0] - phi[-1]

    savez_compressed('rpoGuess', x=aa[:-1:Nsteps], h=h, Nsteps=Nsteps,
                     th=th[0]-th[-1], phi=phi[0]-phi[-1])

if case == 2:
    """
    single shooting
    """
    N = 512
    d = 50
    h = 0.001

    rpoGuess = np.load('rpoGuess.npz')
    nstp = rpoGuess['Nsteps'] * 10
    x0 = rpoGuess['x'][0]
    th0 = rpoGuess['th'].take(0)
    phi0 = rpoGuess['phi'].take(0)
    
    # cglrpo = pyCqcglRPO(nstp, 1, N, d, h,  -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6, 4)
    # rpo = cglrpo.findRPO(x0, nstp*h, th0, phi0, 1e-12, 20, 100, 1e-4, 1e-4, 0.1, 0.5, 30, 100)

if case == 3:
    """
    multi shooting
    """
    N = 512
    d = 50
    h = 0.001

    M = 10
    S = 1
    rpoGuess = np.load('rpoGuess.npz')
    nstp = rpoGuess['Nsteps'].take(0) * S
    x0 = rpoGuess['x'][::S, :].copy()
    th0 = rpoGuess['th'].take(0)
    phi0 = rpoGuess['phi'].take(0)

    cqcglSaveRPO('rpo.h5', '1', x0, nstp*h*M, nstp, th0, phi0, 1000.0)
    # cglrpo = pyCqcglRPO(nstp, M, N, d, h,  -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6, 4)
    # rpo = cglrpo.findRPOM(x0, nstp*h*M, th0, phi0, 1e-12, 20, 100, 1e-4, 1e-4, 0.1, 0.5, 80, 10)

    # cgl = pyCqcgl1d(N, d, h, False, 1,
    #                 -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
    #                 4)
    # Ndim = cgl.Ndim
    # xx = cgl.intg(rpoGuess['x'][0], rpoGuess['Nsteps']*10, 1)
    # plotConfigSpaceFromFourier(cgl, xx, [0, 50, 0, rpoGuess['Nsteps']*10*h])

 
    # cglrpo = pyCqcglRPO(N, d, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    # rpo = cglrpo.findPO(rpoGuess['x'], rpoGuess['h'].take(0),
    #                     rpoGuess['Nsteps'].take(0),
    #                     rpoGuess['th'].take(0), rpoGuess['phi'].take(0),
    #                     100, 1e-12, True, True)
