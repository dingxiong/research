from ksHelp import *

case = 10

if case == 10:
    ksp = KSplot()
    fileName = '../../data/Ruslan/ks22h1t120.h5'
    a, T, nstp, theta, err = ksp.readPO(fileName, ksp.toStr('ppo', 1), False, hasNstp=False)
    
    
