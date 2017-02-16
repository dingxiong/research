from personalFunctions import *
from py_ks import *

case = 10

N, L = 64, 22
eqFile =  '../../data/ks22Reqx64.h5'
poFile = '../../data/ks22h001t120x64EV.h5'

if case == 10:
    """
    print the exponents of ppo/rpo
    """
    ks = pyKS(N, L)
    ksp = KSplot(ks)
    fe = ksp.readFE(poFile, 'rpo', 3)
    np.set_printoptions(formatter={'float': '{: 15.6f}'.format})
    print fe[:, np.r_[0:10, -4:0]].T  # print formated 10 first and 4 last
    np.set_printoptions()
    print 
    print fe[0, :4].T                   # print the marginal ones

