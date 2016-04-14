from time import time
from py_ks import *
import numpy as np

case = 2
if case == 1:
    ks = pyKS(32, 0.1, 22)
    t_init = time()
    for i in range(1000):
        aa = ks.intg(np.ones(30)*0.1, 2000, 1)
    print time() - t_init
    t_init = time()
    for i in range(100):
        aa, daa = ks.intgj(np.ones(30)*0.1, 2000, 1, 1)
    print time() - t_init

    ks = pyKSM1(32, 0.1, 22)
    t_init = time()
    for i in range(1000):
        a0 = np.ones(30)*0.1
        a0[1] = 0
        aa, tt = ks.intg(a0, 2000, 1)
    print time() - t_init

    t_init = time()
    for i in range(10):
        a0 = np.ones(30)*0.1
        a0[1] = 0
        aa, tt = ks.intg2(a0, 20, 1)
    print time() - t_init

if case == 2:
    ks = pyKS(32, 0.1, 22)
    aa = ks.intg(np.ones(30)*0.1, 20, 1)
    aaHat, ang = ks.orbitToSlice(aa)
    aaTilde = ks.reduceReflection(aaHat)

    # check p2
    print np.sqrt(aaHat[:, 2]**2 + aaHat[:, 5]**2)
    print aaTilde[:, 2]
    # check the last transformed term
    print aaHat[:, 26] * aaHat[:, 29] / np.sqrt(aaHat[:, 26]**2 + aaHat[:, 29]**2)
    print aaTilde[:, 29]
    # check the unchanged terms
    print aaHat[:, 3]
    print aaTilde[:, 3]

    x = 0.1*np.arange(30)
    ve = np.sin(range(30))
    veTilde = ks.reflectVe(ve, x)