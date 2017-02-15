from personalFunctions import *
from py_ks import *

case = 10

N, L = 64, 22
eqFile =  '../../data/ks22Reqx64.h5'
poFile = '../../data/ks22h001t120x64EV.h5'

if case == 10:
    """
    plot the profile of eq and req
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    for i in range(1, 4):
        a, err = ksp.readEq(eqFile, i)
        ksp.oneConfig(a, axisLabelSize=25, tickSize=18)
    
    
