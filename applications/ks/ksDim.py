from personalFunctions import *
from py_ks import *


def getEnoughN(x, nb):
    ma = max(x)
    mi = min(x)
    l = ma - mi
    id = np.floor(nb * (x - mi) / l)
    s = 0
    for i range(len(id)):
        
case = 10

if case == 10:
    N = 64
    d = 22
    h = 0.01

    ks = pyKS(N, d)
    fileName = '../../data/ks22h001t120x64EV.h5'
    poType = 'rpo'
    poId = 1
    
    a0, T, nstp, r, s = KSreadPO(fileName, poType, poId)
    a0H = ks.orbitToSlice(a0)[0]
    FE, FV = KSreadFEFV(fileName, poType, poId)
    FV = FV[0].reshape((30, N-2))
    FVH = ks.veToSlice(FV, a0)
    fv = FVH[[0, 1] + range(3, 30)]  # get rid of group tengent

    aa = ks.aintg(rand(N-2), h, T, 1000000)
    vd = np.zeros((0, N-2))
    for i in range(100):
        if i % 10 == 0:
            print i, vd.shape
        a0E = aa[-1]
        aa = ks.aintg(a0E, h, 500, 1)
        aaH = ks.orbitToSlice(aa)[0]
        div = aaH-a0H
        dis = norm(div, axis=1)
        ix = dis < 0.1
        vd = np.vstack((vd, div[ix]))
        
    x = fv[0] / norm(fv[0])
    x = vd.dot(x)
