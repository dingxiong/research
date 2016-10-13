from py_CQCGL1d import *
from personalFunctions import *

case = 10

if case == 10:
    """
    use L = 50 to view the rpo, its FV and the explosion
    """
    N = 1024
    d = 50
    Bi = 2.0
    Gi = -5.6
    
    cgl = pyCQCGL1d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1)
    rpo = CQCGLrpo()
    cp = CQCGLplot(cgl)
    x, T, nstp, th, phi, err, e, v = rpo.readRpoBiGi('../../data/cgl/rpoBiGiEV.h5',
                                                     Bi, Gi, 1, flag=2)
    a0 = x[:cgl.Ndim]
    print e[:10]

    # plot the leading Fv
    isvReal = True
    cp.oneConfig(v[0], d=50, labs=[r'$x$', r'$|\mathbf{e}_0|$' if isvReal else r'$|Re(\mathbf{e}_0)|$'])
    
    # plot 4 period of the orbit
    aa = cgl.intg(a0, T/nstp, 4*nstp, 10)
    cp.config(aa, [0, d, 0, 4*T], axisLabelSize=25, tickSize=15, barTicks=[0.5, 1, 1.5])

    # plot the explosion
    a0 += 0.1*norm(a0)*v[0]
    skipRate = 50
    aa = cgl.intg(a0, T/nstp, 20*nstp, skipRate)
    cellSize = nstp/skipRate
    cp.config(aa[12*cellSize:19*cellSize], [0, d, T*12, T*19])

    
    

    
