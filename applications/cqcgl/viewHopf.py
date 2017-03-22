from py_CQCGL1d import *
from cglHelp import *

################################################################################
# IN the parameter domain (Bi, Gi), we have a transition line which seperates
# stable relative equilibria and unstalbe equilibria. Hopf bifurcation happens
# close to this transition line.
#
# This file contains all case studies related to this Hopf bifurcation.
################################################################################

case = 10

if case == 10:
    """
    have a look at how unstable relative equilibrium turns to
    a limit cycle.
    """
    N, d = 1024, 50
    h = 2e-3

    Bi, Gi = 0.6, -3.6
    index = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    req = CQCGLreq(cgl)
    cp = CQCGLplot(cgl)

    a0, wth0, wphi0, err0, e, v = req.readBiGi('../../data/cgl/reqBiGiEV.h5', Bi, Gi, index, flag=2)
    print e[:20]
    
    nstp = 30000
    a0 += 0.1*norm(a0)*v[0].real
    for i in range(5):
        aa = cgl.intg(a0, h, nstp, 10)
        a0 = aa[-1]
        cp.config(aa, [0, d, 0, nstp*h])
        
    
    # save candidate limit cycle
    if False:        
        rpo = CQCGLrpo(cgl)
        # be careful that the skip_rate is 10
        # rpo.saveBiGi('p.h5', Bi, Gi, 1, aa[200], 3000*h, 3000, 0, 0, 0)

if case == 30:
    """
    have a look at the limit cycle
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -3.6
    index = 1

    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo(cgl)
    cp = CQCGLplot(cgl)
    
    x, T, nstp, th, phi, err = rpo.read('../../data/cgl/rpoHopfBiGi.h5', rpo.toStr(Bi, Gi, 1))
    a0 = x[:-3]
    h = T / nstp
    
    aa = cgl.intg(a0, h, nstp*4, 10)
    cp.config(aa, [0, d, 0, nstp*h])

    # check it is not a req
    vel = cgl.velocity(a0)
    velRed = cgl.ve2slice(vel, a0, 1)
    print norm(velRed)

if case == 40:
    """
    move rpo 
    """
    inFile = '../../data/cgl/rpoBiGi2.h5'
    outFile = '../../data/cgl/rpoHopfBiGi.h5'
    rpo = CQCGLrpo()
    gName = rpo.toStr(1.4, -3.9, 1)

    rpo.move(inFile, gName, outFile, gName, flag=0)
    
