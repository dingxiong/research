from personalFunctions import *
from py_cqcgl1d import pyCgl1d

case = 1

if case == 1:
    N = 256
    d = 100
    h = 0.02
    cgl = pyCgl1d(N, d, h, False, 0, 1.5, -1.4, 4)

    A0 = 3*centerRand(2*N, 0.2)
    a0 = cgl.Config2Fourier(A0)
    nstp = 10000
    aa = cgl.intg(a0, nstp, 1)
    AA = cgl.Fourier2Config(aa)
    AAabs = cgl.Fourier2ConfigMag(aa)
    plotConfigSpaceFromFourier(cgl, aa, [0, d, 0, nstp*h])
