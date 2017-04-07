from py_CQCGL1dSub import *
from cglHelp import *

################################################################################
#   view the reqs in the symmetric subspace
################################################################################

case = 20

if case == 10:
    """
    use the new data to calculate tyhe stability exponents of req
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6
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
    Bi, Gi = 0.8, -0.6
    index = 1

    cgl = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, cp = CQCGLreq(cgl), CQCGLplot(cgl)
    
    a0, wth0, wphi0, err0, e, v = req.read('../../data/cgl/reqsubBiGiEV.h5',
                                           req.toStr(Bi, Gi, index),
                                           sub=True,flag=2)
    print e[:10]
    cp.oneConfig(v[0].real * cgl.N)
    cp.oneConfig(v[0].imag * cgl.N)
