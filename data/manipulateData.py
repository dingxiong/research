from personalFunctions import *
import numpy as np

case = 2

if case == 1:
    """
    test the vadility of rpo in N = 64
    """
    fileName = 'ks22h001t120x64EV.h5'
    bad = []
    for i in range(1, 835):
        a, T, nstp, r, s = KSreadPO(fileName, 'rpo', i)
        Fv = KSreadFV(fileName, 'rpo', i)
        if nstp != 5 * np.double(Fv.shape[0]):
            bad.append(i)
            print i

if case == 2:
    
