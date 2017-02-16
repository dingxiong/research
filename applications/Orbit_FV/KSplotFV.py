from py_ks import *
from personalFunctions import *

case = 10

N, L = 64, 22
eqFile =  '../../data/ks22Reqx64.h5'
poFile = '../../data/ks22h001t120x64EV.h5'

if case == 10:
    """
    plot the FV of ppo1
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    
    fv = ksp.readFV(poFile, 'ppo', 1)
    fvt0 = fv[0].reshape(30, N-2)
    name = 'ksppo1fvt0'
    ksp.oneConfig(fvt0[0], labs=[r'$x$', r'$v$'], axisLabelSize=25, tickSize=18, save=True, name=name+'v1r')
    ksp.oneConfig(fvt0[1], labs=[r'$x$', r'$v$'], axisLabelSize=25, tickSize=18, save=True, name=name+'v1i')
    ksp.oneConfig(fvt0[4], labs=[r'$x$', r'$v$'], axisLabelSize=25, tickSize=18, save=True, name=name+'v5')
    ksp.oneConfig(fvt0[9], labs=[r'$x$', r'$v$'], axisLabelSize=25, tickSize=18, save=True, name=name+'v10')
    ksp.oneConfig(fvt0[-1], labs=[r'$x$', r'$v$'], axisLabelSize=25, tickSize=18, save=True, name=name+'v30')
