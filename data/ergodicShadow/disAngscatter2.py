import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

case = 2

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

    # ######     1st      #################
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
    plot one good truncated figure
    """
    folder = 'cases32modes/case4ppo2x10sT30/'
    sps = np.loadtxt('cases32modes/spacing/spacing_ppo2.txt')
    ang = np.sin(np.arccos(np.loadtxt(folder + 'angle')))
    dis = np.loadtxt(folder + 'dis')
    idx = np.loadtxt(folder + 'indexPo')

    ii = [i for i in range(np.size(dis)) if dis[i] > 4*sps[idx[i]]]

    cix = [3, 4, 5]             # column index
    spix = [6, 7, 8, 15]        # subspace index
    colors = ['r', 'b', 'c', 'm']
    markers = ['o', 's', 'v', '*']
    Num = np.size(cix)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    for i in range(Num):
        ax.scatter(dis[ii], ang[ii, cix[i]], s=7, c=colors[i], marker=markers[i],
                   edgecolor='none', label='1-'+str(spix[i]))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-4, 1e-0])
    ax.set_xlim([1e-3, 1e-1])
    ax.legend(loc='upper left')
    # ax.set_title('(a)')

    ax.set_ylabel(r'$\sin(\theta)$', size='large')
    ax.set_xlabel(r'$||\Delta x||_2$', size='large')
    fig.tight_layout(pad=0)
    plt.show()
