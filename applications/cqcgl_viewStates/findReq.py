from time import time
from py_cqcgl1d_threads import pyCqcgl1d, pyCqcglRPO
from personalFunctions import *


def cqcglFindReq(fileName, frac=0.3, MaxIter=300, Ntrial=1000):
    N = 512
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


def cqcglConvertReq(N2, inputFile, outputFile, indices,
                    MaxIter=300, tol=1e-12, doesPrint=False):
    """
    convert the relative equilibria found for some truncation number N to
    a finer resolution N2 = 2*N.

    parameters:
    N2          the dimension of the finer system
    inputFile   file storing reqs with N
    outputFile  file is about to create to store reqs with 2*N
    indices     the indices of reqs needed to be converted
    """
    # N = 512
    d = 50
    h = 0.005                   # time step is not used acturally
    cgl = pyCqcgl1d(N2, d, h, True, 0,
                    -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
                    4)

    for i in indices:
        a0, wth0, wphi0, err0 = cqcglReadReq(inputFile, str(i))
        a0unpad = 2 * cgl.generalPadding(a0)
        a, wth, wphi, err = cgl.findReq(a0unpad, wth0, wphi0,
                                        MaxIter, tol, True, doesPrint)
        print err
        cqcglSaveReq(outputFile, str(i), a, wth, wphi, err)


case = 11

if case == 1:
    """
    find the relative equlibria after changing dimension
    """
    N = 512
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6)
    Ndim = cgl.Ndim

    a0, wth0, wphi0, err0 = cqcglReadReq('../../data/cgl/req.h5', '10')

    a0unpad = 2*cgl.generalPadding(a0)
    a, wth, wphi, err = cgl.findReq(a0unpad, 0, wphi0,
                                    100, 1e-12, True, True)
    # cqcglSaveReq("req.hdf5", '1', a, wth, wphi, err)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])

if case == 2:
    """
    try different guesses to find relative equilibria
    Just for test purpose
    """
    N = 1024
    d = 30
    h = 0.0002
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, -0.01, -0.08, 4)

    A0 = 5*centerRand(2*N, 0.2)
    # A0 = 5*centerOne(2*N, 0.25)
    a0 = cgl.Config2Fourier(A0)
    wth0 = rand()
    wphi0 = rand()
    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 100, 1e-12, True, True)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])
    plot1dfig(a)
    plotOneConfigFromFourier(cgl, a)
    print wth, wphi
    # cqcglSaveReq("req2.h5", '1', a, wth, wphi, err)

if case == 10:
    """
    use existing req to find the req with slightly different di
    """
    N = 1024
    d = 30
    h = 0.0002
    diOld = 0.41
    di = diOld + 0.001
    cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)
    
    a0, wth0, wphi0, err = cqcglReadReqdi('../../data/cgl/reqDi.h5',
                                          diOld, 1)
    a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 40, 1e-12, True, True)
    nstp = 10000
    aa = cgl.intg(a, nstp, 1)
    plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h])
    plot1dfig(a)
    plotOneConfigFromFourier(cgl, a)
    print wth, wphi
    # cqcglSaveReq("req98.h5", '1', a, wth, wphi, err)
    eigvalues, eigvectors = eigReq(cgl, a, wth, wphi)
    print eigvalues[:10]

if case == 11:
    """
    use existing req to find the req with slightly different di
    and do this for a sequence of di s.
    """
    N = 1024
    d = 30
    h = 0.0002

    diOld = 0.94
    diInc = 0.02
    for i in range(1):
        di = diOld + diInc
        print di

        cgl = pyCqcgl1d(N, d, h, True, 0, 4.0, 0.8, 0.01, di, 4)

        fileName = '../../data/cgl/reqDi.h5'
        a0, wth0, wphi0, err0 = cqcglReadReqdi(fileName, diOld, 2)
        a, wth, wphi, err = cgl.findReq(a0, wth0, wphi0, 20, 1e-12, True, True)

        plotOneConfigFromFourier(cgl, a, save=True, name=str(di)+'.png')
        print wth, wphi, err
        e, v = eigReq(cgl, a, wth, wphi)
        print e[:8]
        cqcglSaveReqEVdi(fileName, di, 2, a, wth, wphi, err,
                         e.real, e.imag, v[:, :10].real, v[:, :10].imag)

        diOld += diInc


if case == 3:
    """
    run a long time to collect relative equilibria
    """
    cqcglFindReq("req3.h5", 0.3, 300, 1000)


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
    change the existing reqs to different dimension
    """
    # cqcglConvertReq('../../data/cgl/req.h5', 'req2.h5', range(1, 23))
    # cqcglConvertReq(2048, '../../data/cgl/reqN1024.h5', 'req2.h5', range(1, 23))
    cqcglConvertReq(4096, '../../data/cgl/reqN2048.h5', 'req2.h5',
                    range(1, 23), tol=1.5e-11, doesPrint=True)

if case == 6:
    """
    plot the interesting relative equilibria that I have found
    """
    N = 1024
    d = 50
    h = 0.005
    cgl = pyCqcgl1d(N, d, h, True, 0,
                    -0.1, 1.0, 0.8, 0.125, 0.5, -0.1, -0.6,
                    4)

    Num = 22
    nstp = 18000
    for i in range(Num):
        a, wth, wphi, err = cqcglReadReq('../../data/cgl/reqN1024.h5', str(i+1))
        aa = cgl.intg(a, nstp, 1)
        plotConfigSpace(cgl.Fourier2Config(aa), [0, d, 0, nstp*h],
                        save=True, name='cqcglReq'+str(i+1)+'T90'+'.png')
        plotOneConfigFromFourier(cgl, a, save=True,
                                 name='cqcglReq'+str(i+1)+'.png')
