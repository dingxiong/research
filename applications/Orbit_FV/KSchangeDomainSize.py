from py_ks import *
from personalFunctions import *

case = 20

if case == 10:
    """
    use a larger domain size for simulation
    """
    N, L = 128*2, 100
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    T, h = 100, 0.001

    a0 = rand(N-2)
    aa = ks.intg(a0, h, np.int(100/h), 1000000)
    a0 = aa[-1]
    aa = ks.intg(a0, h, np.int(T/h), 5)
    
    ksp.config(aa, [0, L, 0, T], size=[6, 6], axisLabelSize=25, tickSize=16)

if case == 20:
    """
    use a larger domain size for simulation
    """
    N, L = 128*2, 200
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    T, h = 100, 0.001

    a0 = rand(N-2)
    aa = ks.intg(a0, h, np.int(400/h), 1000000)
    a0 = aa[-1]
    aa = ks.intg(a0, h, np.int(T/h), 5)
    
    ksp.config(aa, [0, L, 0, T], size=[6, 6], axisLabelSize=25, tickSize=16)
    

