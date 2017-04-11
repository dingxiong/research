from py_CQCGL1dSub import *
from py_CQCGL1d import *
from cglHelp import *

################################################################################
#   compare req in subspace and in full space
################################################################################

case = 10

if case == 10:
    """
    compare eigenvalues for symmetric subspace and full space
    """
    N, d = 1024, 50
    Bi, Gi = 2, -5
    index = 1

    cgl = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req = CQCGLreq(cgl)
    
    a0, wth0, wphi0, err0, e0, v0 = req.read('../../data/cgl/reqBiGiEV.h5',
                                             req.toStr(Bi, Gi, index), flag=2)
    e1 = e0

    a0, wth0, wphi0, err0, e0, v0 = req.read('../../data/cgl/reqsubBiGiEV.h5',
                                             req.toStr(Bi, Gi, index), flag=2,
                                             sub=True)
    e2 = e0

    print e1[:10]
    print e2[:10]

if case == 20:
    """
    compare eigenvectors
    """
    N, d = 1024, 50
    Bi, Gi = 2, -2
    index = 1
    
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    cglsub = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, reqsub = CQCGLreq(cgl), CQCGLreq(cglsub), 
    cp, cpsub = CQCGLplot(cgl), CQCGLplot(cglsub)

    a0, wth0, wphi0, err0, e0, v0 = req.read('../../data/cgl/reqBiGiEV.h5',
                                           req.toStr(Bi, Gi, index),
                                           flag=2)
    cp.oneConfig(v0[0].real * cgl.N)
    cp.oneConfig(v0[0].imag * cgl.N)
    cp.oneConfig(v0[1].real * cgl.N)
    cp.oneConfig(v0[1].imag * cgl.N)

    a1, wth1, wphi1, err1, e1, v1 = reqsub.read('../../data/cgl/reqsubBiGiEV.h5',
                                                req.toStr(Bi, Gi, index),
                                                sub=True,flag=2)
    cpsub.oneConfig(v1[0].real * cgl.N)
    cpsub.oneConfig(v1[0].imag * cgl.N)

if case == 30:
    """
    same as case 20, but sperate real/imag part
    """
    N, d = 1024, 50
    Bi, Gi = 2, -2
    index = 1
    
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    cglsub = pyCQCGL1dSub(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0)
    req, reqsub = CQCGLreq(cgl), CQCGLreq(cglsub), 
    cp, cpsub = CQCGLplot(cgl), CQCGLplot(cglsub)

    a0, wth0, wphi0, err0, e0, v0 = req.read('../../data/cgl/reqBiGiEV.h5',
                                           req.toStr(Bi, Gi, index),
                                           flag=2)
    plot1dfig(cgl.Fourier2Config(v0[0].real * cgl.N).real)
    plot1dfig(cgl.Fourier2Config(v0[0].real * cgl.N).imag)
    plot1dfig(cgl.Fourier2Config(v0[0].imag * cgl.N).real)
    plot1dfig(cgl.Fourier2Config(v0[0].imag * cgl.N).imag)
    plot1dfig(cgl.Fourier2Config(v0[2].real * cgl.N).real)
    plot1dfig(cgl.Fourier2Config(v0[2].real * cgl.N).imag)
    plot1dfig(cgl.Fourier2Config(v0[2].imag * cgl.N).real)
    plot1dfig(cgl.Fourier2Config(v0[2].imag * cgl.N).imag)
    

    a1, wth1, wphi1, err1, e1, v1 = reqsub.read('../../data/cgl/reqsubBiGiEV.h5',
                                                req.toStr(Bi, Gi, index),
                                                sub=True,flag=2)
    plot1dfig(cglsub.Fourier2Config(v1[0].real * cglsub.N).real)
    plot1dfig(cglsub.Fourier2Config(v1[0].real * cglsub.N).imag)
    plot1dfig(cglsub.Fourier2Config(v1[0].imag * cglsub.N).real)
    plot1dfig(cglsub.Fourier2Config(v1[0].imag * cglsub.N).imag)

