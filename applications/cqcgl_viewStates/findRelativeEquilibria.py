import h5py
from pylab import *
import numpy as np
from time import time
from py_cqcgl1d import pyCqcgl1d
from personalFunctions import *


def cqcglFindReq(fileName, frac=0.3, MaxIter=300, Ntrial=1000):
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)

    ReqNum = 1
    for i in range(Ntrial):
        A0 = 3*centerRand(2*N, frac)
        a0 = cgl.Config2Fourier(A0).squeeze()
        wth0 = rand()
        wphi0 = rand()
        a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, MaxIter,
                                        1e-12, True, False)
        if err < 1e-12:
            print "found : " + str(ReqNum)
            cqcglSaveReq(fileName, str(ReqNum), a, wth, wphi, err)
            ReqNum += 1


def cqcglRemoveReq(inputFile, outputFile, Num, groups):
    """
    remove some groups in relative equilibria file
    Num: the total number of groups in original file
         The group names are: 
         1, 2, 3, 4, ..., Num
    groups: the group names that need to be removed
    """
    ix = 1
    for i in range(1, Num+1):
        if i not in groups:
            a, wth, wphi, err = cqcglReadReq(inputFile, str(i))
            cqcglSaveReq(outputFile, str(ix), a, wth, wphi, err)
            ix += 1


def cqcglExtractReq(inputFile, outputFile, groups, startId=1):
    """
    Extract a subset of relative equibiria from input file
    The inverse of cacglRemoveReq
    groups: the indice of gropus that are going to be extracted from
            input file
    startId : the start group index in the output file
    """
    ix = startId
    n = np.size(groups)
    for i in range(n):
        a, wth, wphi, err = cqcglReadReq(inputFile, str(groups[i]))
        cqcglSaveReq(outputFile, str(ix), a, wth, wphi, err)
        ix += 1

case = 5

if case == 1:
    """
    refine the old relative equilibria
    """
    N = 256
    d = 50
    h = 0.005

    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    f = h5py.File('../../data/cgl/reqOld.h5', 'r')
    req = '/req1/'
    a0 = f[req+'a0'].value
    wth0 = f[req+'w1'].value[0]
    wphi0 = f[req+'w2'].value[0]

    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 100, 1e-12, True, True)
    cqcglSaveReq("req.hdf5", '1', a, wth, wphi, err)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])

if case == 2:
    """
    try different guesses to find relative equilibria
    Just for test purpose
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    
    A0 = 3*centerRand(2*N, 0.25)
    a0 = cgl.Config2Fourier(A0).squeeze()
    wth0 = rand()
    wphi0 = rand()
    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 500, 1e-12, True, True)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])
    plot1dfig(a)
    plotOneConfigFromFourier(cgl, a)
    print wth, wphi
    # cqcglSaveReq("req2.hdf5", '16', a, wth, wphi, err)

if case == 3:
    """
    run a long time to collect relative equilibria
    """
    cqcglFindReq("req3.hdf5", 0.3, 300, 1000)


if case == 4:
    """
    have a look at these relative equilibrium and remove the duplicates.
    All kinds of manipulations to build the final data base.
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)

    # Num = 851
    # wthAll = np.zeros(Num)
    # wphiAll = np.zeros(Num)
    # errAll = np.zeros(Num)
    # aAll = np.zeros((Num, 2*N))
    # for i in range(Num):
    #     a, wth, wphi, err = cqcglReadReq("req3.hdf5", str(i+1))
    #     aAll[i] = a
    #     wthAll[i] = wth
    #     wphiAll[i] = wphi
    #     errAll[i] = err

    # dup = []
    # groups = []
    # for i in range(Num):
    #     for j in range(i):
    #         if (np.abs(np.abs(wthAll[i]) - np.abs(wthAll[j])) < 1e-6 and
    #             np.abs(wphiAll[i] - wphiAll[j]) < 1e-6):
    #             dup.append([i, j])
    #             groups.append(i+1)
    #             break

    # cqcglRemoveReq("req3.hdf5", "req4.hdf5", Num, groups)

    # Num2 = 567
    # for i in range(Num2):
    #     a, wth, wphi, err = cqcglReadReq("req4.hdf5", str(i+1))
    #     plotOneConfigFromFourier(cgl, a, save=True,
    #                              name=str(i+1)+'.png')
    #     print wth, wphi, err
    #     wait = raw_input("press enter to continue")

    # groups = [1, 2, 3, 10, 12, 18, 62, 138, 141, 212, 214, 221, 265,
    #           270, 276, 292, 330, 351, 399, 414, 425]
    # cqcglExtractReq('req4.hdf5', 'req5.h5', groups, 2)

    # for i in range(21):
    #     a, wth, wphi, err = cqcglReadReq("req5.h5", str(i+2))
    #     print wth, wphi, err
    #     plotOneConfigFromFourier(cgl, a, save=True,
    #                              name=str(i+2)+'.png')


if case == 5:
    """
    plot the interesting relative equilibria that I have found
    """
    N = 256
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)

    Num = 22
    nstp = 18000
    for i in range(Num):
        a, wth, wphi, err = cqcglReadReq('../../data/cgl/req.h5', str(i+1))
        aa = cgl.intg(a, nstp, 1)
        plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h],
                        save=True, name='cqcglReq'+str(i+1)+'T90'+'.png')
        plotOneConfigFromFourier(cgl, a, save=True,
                                 name='cqcglReq'+str(i+1)+'.png')
