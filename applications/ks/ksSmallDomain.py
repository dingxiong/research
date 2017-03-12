from personalFunctions import *
from py_ks import *

case = 10

if case == 10:
    N, L = 64, 21
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    
    a0 = rand(N-2)
    for i in range(50):
        aa = ks.intg(a0, 0.01, 10000, 10000)
        a0 = aa[-1]
    aa = ks.intg(a0, 0.01, 10000, 10)
    raa, ths = ks.redSO2(aa, 1, False)
    
    # plot3dfig(aa[:, 0], aa[:,1], aa[:,2])
    # plot3dfig(raa[:, 0], raa[:,2], raa[:,3])
    ksp.config(aa, [0, L, 0, 100])
    
