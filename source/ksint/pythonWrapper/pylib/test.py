from time import time
from py_ks import *
import numpy as np

ks = pyKS(32, 0.1, 22);
t_init = time();
for i in range(1000) :
    aa = ks.intg(np.ones(30)*0.1, 2000, 1);
print time() - t_init;
t_init = time();
for i in range(100) :
    aa, daa = ks.intgj(np.ones(30)*0.1, 2000, 1, 1);
print time() - t_init;


ks = pyKSM1(32, 0.1, 22);
t_init = time();
for i in range(1000) :
    a0 = np.ones(30)*0.1; a0[1] = 0;
    aa, tt = ks.intg(a0, 2000, 1);
print time() - t_init;

t_init = time();
for i in range(10) :
    a0 = np.ones(30)*0.1; a0[1] = 0;
    aa, tt = ks.intg2(a0, 20, 1);
print time() - t_init;


