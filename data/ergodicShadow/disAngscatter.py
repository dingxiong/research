import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from personalFunctions import *
from py_ks import *


def statisAverage(dis, ang, cell):
    dis = np.floor(np.log10(dis)/cell)
    N = len(dis)
    minD = np.min(dis)
    maxD = np.max(dis)
    sumAng = np.zeros((maxD-minD+1, ang.shape[1]))
    sumNum = np.zeros(maxD - minD + 1)
    for i in range(N):
        ix = dis[i] - minD
        sumNum[ix] += 1
        sumAng[ix, :] += ang[i, :]

    # calcuate the mean value
    average = np.zeros([maxD-minD+1, ang.shape[1]])
    for i in range(len(sumNum)):
        if sumNum[i] != 0:
            average[i, :] = sumAng[i, :] / sumNum[i]

    # form x coordinate
    x = np.arange(minD, maxD+1) + 0.5

    return 10**(x*cell), average


def calSpacing(fileName, ppType, ppId, nqr):
    """
    calculate the spacing of points in a rpo/ppo
    """
    a, T, nstp, r, s = KSreadPO(fileName, ppType, ppId)
    h = T / nstp
    ks = pyKS(64, h, 22)
    aa = ks.intg(a, np.int(nstp), nqr)
    aa = aa[:-1, :]
    if ppType == 'ppo':
        aaWhole = ks.half2whole(aa)
    else:
        aaWhole = aa
    aaHat = ks.orbitToSlice(aaWhole)[0]
    M = aaHat.shape[0]
    dis = np.zeros(M)
    for i in range(M):
        dis[i] = np.linalg.norm(aaHat[(i+1) % M] - aaHat[i])
    return dis


def setAxis(ax, xl, xr, yl, yr, px, py):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([yl, yr])
    ax.set_xlim([xl, xr])
    # ax.legend(loc='upper left')
    # ax.set_title('(a)')

    ax.text(px, py, r'$\sin(\theta)$', fontsize=18)
    # ax.set_ylabel(r'$\sin(\theta)$', size='large')
    # ax.set_xlabel(r'$||\Delta \hat{u}||_2$', fontsize=15, labelpad=0)


def setAxis2(ax):
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.text(px, py, r'$\langle \sin(\theta) \rangle$', fontsize=15)
    # ax.set_ylabel(r'$< \sin(\theta) >$', size='large')
    # ax.set_xlabel(r'$||\Delta \hat{u}||_2$', size='large')

    
if __name__ == "__main__":

    case = 10

    if case == 1:
        """
        compare the truncated and untruncated
        """
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        folder = 'cases32modes/case4rpo8x10sT40/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')
        sps = np.loadtxt('cases32modes/spacing/spacing_rpo8.txt')

        ax.scatter(dis, ang[:, 4], s=7, c='b', marker='o', edgecolor='none')
        ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]
        ax.scatter(dis[ii], ang[ii, 4], s=7, c='r', marker='s', edgecolor='none')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-4, 1e-1])
        ax.set_xlim([1e-3, 1e-1])

        ax.set_ylabel(r'$\sin(\theta)$', size='large')
        ax.set_xlabel(r'$||\Delta x||_2$', size='large')

        fig.tight_layout(pad=0)
        plt.show()

    if case == 2:
        """
        plot one good truncated statistic figure
        """
        folder = 'cases32modes/case4rpo4x10sT30/'
        sps = np.loadtxt('cases32modes/spacing/spacing_rpo4.txt')
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')

        ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]

        cix = [3, 4, 5]             # column index
        spix = [6, 7, 8, 15]        # subspace index
        colors = ['r', 'b', 'c', 'm']
        markers = ['o', 's', 'v', '*']
        Num = np.size(cix)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        for i in range(Num):
            ax.scatter(dis[ii], ang[ii, cix[i]], s=7, c=colors[i], marker=markers[i],
                       edgecolor='none', label='1-'+str(spix[i]))

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-4, 1e-0])
        ax.set_xlim([1e-3, 1e-1])
        # ax.legend(loc='upper left')
        # ax.set_title('(a)')

        ax.text(1.2e-3, 2e-1, r'$\sin(\theta)$', fontsize=18)
        # ax.set_ylabel(r'$\sin(\theta)$', size='large')
        # ax.set_xlabel(r'$||\Delta x||_2$', size='large')
        fig.tight_layout(pad=0)
        plt.show()

    if case == 3:
        """
        plot only one shadowing incidence
        """
        folder = 'cases32modes/case4ppo3x10sT30/'
        sps = np.loadtxt('cases32modes/spacing/spacing_ppo3.txt')
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')
        No = np.loadtxt(folder + 'No')

        midx = np.argmax(No)
        i1 = np.int(np.sum(No[:midx]))
        i2 = np.int(np.sum(No[:midx+1]))
        ii = [i for i in range(i1, i2) if dis[i] > 4*sps[idx[i]]]

        cix = [3, 4, 5]             # column index
        spix = [6, 7, 8, 15]        # subspace index
        colors = ['r', 'b', 'c', 'm']
        markers = ['o', 's', 'v', '*']
        Num = np.size(cix)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        for i in range(Num):
            ax.scatter(dis[ii], ang[ii, cix[i]], s=10, c=colors[i], marker=markers[i],
                       edgecolor='none', label='1-'+str(spix[i]))

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([5e-4, 2e-1])
        ax.set_xlim([5e-3, 1e-1])
        # ax.legend(loc='upper left')
        # ax.set_title('(a)')

        ax.text(6e-3, 9e-2, r'$\sin(\theta)$', fontsize=18)
        # ax.set_ylabel(r'$\sin(\theta)$', size='large')
        # ax.set_xlabel(r'$||\Delta \hat{u}||_2$', fontsize=15, labelpad=0)
        fig.tight_layout(pad=0)
        plt.show()

    if case == 4:
        """
        plot spacing along periodic orbit
        """
        sps = np.loadtxt('spacing/spacing_rpo6.txt')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.semilogy(sps)

        ax.set_xlabel('index of points along the orbit')
        ax.set_ylabel('spacing')
        ax.set_title('rpo6')
        # ax.set_xlim([0, 2100])
        ax.set_ylim([4e-4, 1e-1])

        plt.tight_layout(pad=0)
        plt.show()

    if case == 5:
        """
        plot the distance structure
        """
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        dis = np.loadtxt('./cases32modes/case4rpo6x10sT30/dis')
        No = np.loadtxt('./cases32modes/case4rpo6x10sT30/No', dtype=int)
        ix = 13                 # shadowing incidence.
        h = 0.1

        start = No[:ix].sum()   # number of point before what we want
        x = h*np.arange(No[ix])
        y = dis[start:start+No[ix]]

        ax.plot(x, y, c='b', lw=1)
        # ax.quiver(22, 2e-3, 0, 0.1)
        # ax.quiver(18, 2e-3, 0, 0.1)
        ax.text(8.5, 6e-3, 'B')
        ax.text(16.5, 1.6e-2, 'C')

        ax.set_ylim([4e-3, 1e-1])
        ax.set_yscale('log')

        ax.set_xlabel('t')
        ax.set_ylabel(r'$||\Delta x||_2$')

        fig.tight_layout(pad=0)
        plt.show()

    ############################################################
    #                 The new set of data : x5                 #
    ############################################################
    if case == 6:
        """
        plot one good truncated statistic figure
        """
        ppType = 'rpo'
        ppId = 4
        sps = calSpacing('../ks22h001t120x64EV.h5', ppType, ppId, 5)

        folder = 'cases32modes_x5/case4rpo4x5sT30/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')

        ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]

        cix = [3, 4, 5]             # column index
        spix = [6, 7, 8, 15]        # subspace index
        colors = ['r', 'b', 'c', 'm']
        markers = ['o', 's', 'v', '*']
        Num = np.size(cix)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        for i in range(Num):
            ax.scatter(dis[ii], ang[ii, cix[i]], s=7, c=colors[i], marker=markers[i],
                       edgecolor='none', label='1-'+str(spix[i]))

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-4, 1e-0])
        ax.set_xlim([1e-3, 1e-1])
        # ax.legend(loc='upper left')
        # ax.set_title('(a)')

        ax.text(1.2e-3, 2e-1, r'$\sin(\theta)$', fontsize=18)
        # ax.set_ylabel(r'$\sin(\theta)$', size='large')
        # ax.set_xlabel(r'$||\Delta x||_2$', size='large')
        fig.tight_layout(pad=0)
        plt.show()

    if case == 7:
        """
        plot only one shadowing incidence
        """
        ppType = 'ppo'
        ppId = 4
        sps = calSpacing('../ks22h001t120x64EV.h5', ppType, ppId, 5)

        folder = 'cases32modes_x5/case4ppo4x5sT30/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')
        No = np.loadtxt(folder + 'No')

        # midx = 1 for rpo4; midx = 6 for ppo4;
        # midx = np.argmax(No)
        midx = 6
        i1 = np.int(np.sum(No[:midx]))
        i2 = np.int(np.sum(No[:midx+1]))
        ii = [i for i in range(i1, i2) if dis[i] > 4*sps[idx[i]]]

        cix = [3, 4, 5]             # column index
        spix = [6, 7, 8, 15]        # subspace index
        colors = ['r', 'b', 'c', 'm']
        markers = ['o', 's', 'v', '*']
        Num = np.size(cix)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        for i in range(Num):
            ax.scatter(dis[ii], ang[ii, cix[i]], s=10, c=colors[i], marker=markers[i],
                       edgecolor='none', label='1-'+str(spix[i]))

        # setAxis(ax, 1e-3, 1e-1, 2e-4, 5e-1, 3.5e-3, 1.5e-1)  # for rpo4
        setAxis(ax, 7e-3, 1e-1, 5e-4, 8e-2, 8e-3, 4e-2)  # for ppo4

        fig.tight_layout(pad=0)
        plt.show()

    if case == 8:
        """
        plot the statistic avaerage
        """
        ppType = 'rpo'
        ppId = 4
        sps = calSpacing('../ks22h001t120x64EV.h5', ppType, ppId, 5)

        folder = 'cases32modes_x5/case4rpo4x5sT30/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')

        ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]

        cell = 0.2
        x, aver = statisAverage(dis[ii], ang[ii], 0.2)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        
        for i in range(aver.shape[1]):
            ax.plot(x, aver[:, i], '-o')

        setAxis2(ax)
        fig.tight_layout(pad=0)
        plt.show()

    if case == 9:
        """
        calcuate and save the full angle points.
        """
        ppType = 'rpo'
        ppId = 4
        gTpos = 3
        sps = calSpacing('../ks22h001t120x64EV.h5', ppType, ppId, 5)

        a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', ppType, ppId)
        h = T / nstp
        veAll = KSreadFV('../ks22h001t120x64EV.h5', ppType, ppId)
        ks = pyKS(64, h, 22)
        aaHat, veHat = ks.orbitAndFvWholeSlice(a, veAll, nstp, ppType, gTpos)

        folder = 'cases32modes_x5/case4rpo4x5sT30/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')
        difv = np.loadtxt(folder + 'difv')

        M = len(idx)
        NFV = 29
        ang2 = np.zeros((M, NFV))
        for i in range(M):
            if i % 100 == 0:
                print "i = ", i
            d = idx[i]
            fvs = veHat[d*NFV:(d+1)*NFV, :]
            for j in range(NFV):
                ang2[i, j] = pAngle(difv[i], fvs[:j+1].T)

        ang3 = np.cos(ang2)
        np.savetxt('ang2.dat', ang3)

    if case == 10:
        """
        plot the statistic avarage also use more subspaces
        """
        ppType = 'rpo'
        ppId = 4
        sps = calSpacing('../ks22h001t120x64EV.h5', ppType, ppId, 5)

        folder = 'cases32modes_x5/case4rpo4x5sT30/'
        ang = np.sin(np.arccos(np.loadtxt(folder + 'angle2')))
        dis = np.loadtxt(folder + 'dis')
        idx = np.loadtxt(folder + 'indexPo')

        ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]

        cell = 0.2
        x, aver = statisAverage(dis[ii], ang[ii], 0.2)

        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        
        for i in range(3, 6) + range(6, 11, 2)+range(16, 25, 4):
            ax.plot(x, aver[:, i], '-o')

        setAxis2(ax)
        ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
        fig.tight_layout(pad=0)
        plt.show()
