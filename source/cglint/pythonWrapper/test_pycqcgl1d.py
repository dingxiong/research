import numpy as np
from numpy.random import rand
from py_cqcgl1d import pyCqcgl1d
cgl = pyCqcgl1d(256, 50, 0.01, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)

a0 = rand(512)
aa = cgl.intg(a0, 10, 1)

aa, daa = cgl.intgj(a0, 10, 1, 1)

