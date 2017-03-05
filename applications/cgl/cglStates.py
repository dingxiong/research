from personalFunctions import *
from py_CQCGL1d import *

case = 10

# complex Ginzburg Landau equation
# A_t = A + (1 + alpha*i) A_{xx} - (1 + beta*i) |A|^2 A 
# The constructor siginature is
# A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A

if case == 10:
    """
    alpha = 1.5, beta = -1.4
    L = 100
    """
    N, d = 256, 100
    alpha, beta = 2, -2
    
    cgl = pyCQCGL1d(N, d, 1, 1, alpha, -1, -beta, 0, 0, -1)
    cgl.IsQintic = False
    cp = CQCGLplot(cgl)
    h = 0.02
    T = 100.0
    nstp = np.int(T/h)
    # cgl = pyCgl1d(N, d, h, False, 0, 1.5, -1.4, 4)

    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    aa = cgl.intg(a0, T/nstp, 1000, 1000000)
    a0 = aa[-1]
    aa = cgl.intg(a0, T/nstp, nstp, 10)
    
    cp.config(aa, [0, d, 0, T], size=[3.6, 6], percent='3%', barTicks=[0, 0.5, 1],
              axisLabelSize=25, tickSize=15)

if case == 20:
    """
    complex Ginzburg Landau equation
    A_t = A + (1 + alpha*i) A_{xx} - (1 + beta*i) |A|^2 A 
    alpha = 1.5, beta = -1.4
    The constructor siginature is
    A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    """
    N, d = 256, 200
    alpha, beta = 2, -2
    
    cgl = pyCQCGL1d(N, d, 1, 1, alpha, -1, -beta, 0, 0, -1)
    cgl.IsQintic = False
    cp = CQCGLplot(cgl)
    h = 0.02
    T = 100.0
    nstp = np.int(T/h)
    # cgl = pyCgl1d(N, d, h, False, 0, 1.5, -1.4, 4)

    A0 = 3*centerRand(N, 0.2, True)
    a0 = cgl.Config2Fourier(A0)
    aa = cgl.intg(a0, T/nstp, 1000, 1000000)
    a0 = aa[-1]
    aa = cgl.intg(a0, T/nstp, nstp, 10)
    
    cp.config(aa, [0, d, 0, T], size=[6, 6], percent='3%', barTicks=[0, 0.5, 1],
              axisLabelSize=25, tickSize=15)
