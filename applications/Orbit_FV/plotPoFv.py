import h5py
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
# from IPython.display import display
# from IPython.html.widgets import interact
from py_ks import *
from personalFunctions import *

##################################################
# load data
f = h5py.File('../../data/myN32/ks22h02t100EV.h5', 'r')

dataName1 = '/ppo/1/'
a1 = f[dataName1 + 'a'].value
T1 = f[dataName1 + 'T'].value[0]
nstp1 = np.int(f[dataName1 + 'nstp'].value[0])
h1 = T1 / nstp1
e1 = f[dataName1 + 'e'].value
ve1 = f[dataName1 + 've'].value

dataName2 = '/rpo/2/'
a2 = f[dataName2 + 'a'].value
T2 = f[dataName2 + 'T'].value[0]
nstp2 = np.int(f[dataName2 + 'nstp'].value[0])
h2 = T2 / nstp2
e2 = f[dataName2 + 'e'].value
ve2 = f[dataName2 + 've'].value

##################################################
# different experimental cases

case = 3

if case == 1:
    """
    visualize the orbit after reduce reflection symmetry
    projected into Fourier space
    """
    ks1 = pyKS(32, h1, 22)
    aa1 = ks1.intg(a1, nstp1, 1)
    aa1 = aa1[:-1]
    # aaWhole1 = ks1.half2whole(aa1)
    aaWhole1 = aa1
    aaHat1, ang1 = ks1.orbitToSlice(aaWhole1)
    aaTilde1 = ks1.reduceReflection(aaHat1)
    plot3dfig(aaHat1[:, 0], aaHat1[:, 2], aaHat1[:, 3])
    plot3dfig(aaTilde1[:, 0], aaTilde1[:, 2], aaTilde1[:, 3])

if case == 2:
    """
    see how ppo2 is shadowed by rpo1 after reducing O2 symmetry.
    Note: only integrate one period
    """
    ks1 = pyKS(32, h1, 22)
    aa1 = ks1.intg(a1, nstp1, 1)[:-1]
    aaWhole1 = aa1
    aaHat1, ang1 = ks1.orbitToSlice(aaWhole1)
    aaTilde1 = ks1.reduceReflection(aaHat1)

    ks2 = pyKS(32, h2, 22)
    aa2 = ks2.intg(a2, nstp2, 1)
    aa2 = aa2[:-1]
    aaHat2, ang2 = ks2.orbitToSlice(aa2)
    aaTilde2 = ks2.reduceReflection(aaHat2)

    plot3dfig2lines(aaTilde1[:, 0], aaTilde1[:, 2], aaTilde1[:, 3],
                    aaTilde2[:, 0], aaTilde2[:, 2], aaTilde2[:, 3])
    
if case == 3:
    """
    visualize how Floquet vectors are aligned along ppo/rpo
    """
    ks1 = pyKS(32, h1, 22)
    aa1 = ks1.intg(a1, nstp1, 1)
    aa1 = aa1[:-1]
    aaWhole1 = ks1.half2whole(aa1)
    veWhole1 = ks1.half2whole(ve1)
    aaHat1, ang1 = ks1.orbitToSlice(aaWhole1)
    vep1 = ks1.veToSliceAll(veWhole1, aaWhole1, 0)

    ks2 = pyKS(32, h2, 22)
    aa2 = ks2.intg(a2, nstp2, 1)
    aa2 = aa2[:-1]
    aaHat2, ang2 = ks2.orbitToSlice(aa2)
    vep2 = ks2.veToSliceAll(ve2, aa2, 0)

    i1 = 0
    i2 = 2
    i3 = 3

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aaHat1[:, i1], aaHat1[:, i2], aaHat1[:, i3], 'r')
    ax.plot(aaHat2[:, i1], aaHat2[:, i2], aaHat2[:, i3], 'g')

    spa = 70

    ratio1 = 5
    veIndex1 = 1
    ah1 = aaHat1
    nor1 = norm(vep1[veIndex1::30], axis=1)
    nor1.resize((nor1.size, 1))
    ae1 = aaHat1 + vep1[veIndex1::30] / nor1 / ratio1

    ratio2 = 5
    veIndex2 = 1
    ah2 = aaHat2
    nor2 = norm(vep2[veIndex2::30], axis=1)
    nor2.resize((nor2.size, 1))
    ae2 = aaHat2 + vep2[veIndex2::30] / nor2 / ratio2

    shift = 900
    ah1 = np.vstack((ah1[shift:], ah1[:shift]))
    ae1 = np.vstack((ae1[shift:], ae1[:shift]))

    for i in np.arange(0, aaHat1.shape[0]/2, spa):
        a1 = Arrow3D([ah1[i, i1], ae1[i, i1]], [ah1[i, i2], ae1[i, i2]],
                     [ah1[i, i3], ae1[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="m")
        ax.add_artist(a1)

    for i in np.arange(0, aaHat2.shape[0], spa):
        a2 = Arrow3D([ah2[i, i1], ae2[i, i1]], [ah2[i, i2], ae2[i, i2]],
                     [ah2[i, i3], ae2[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="k")
        ax.add_artist(a2)

    ax.set_xlabel(r'$b_1$')
    ax.set_ylabel(r'$b_2$')
    ax.set_zlabel(r'$c_2$')
    plt.tight_layout(pad=0)
    plt.show(block=False)

if case == 4:
    """
    visualize how Floquet vectors are aligned along ppo/rpo
    afeter reducing both rotation and reflection
    """
    ks1 = pyKS(32, h1, 22)
    aa1 = ks1.intg(a1, nstp1, 1)[:-1]
    aaHat1, ang1 = ks1.orbitToSlice(aa1)
    vep1 = ks1.veToSliceAll(ve1, aa1, 0)
    aaTilde1 = ks1.reduceReflection(aaHat1)
    veTilde1 = ks1.reflectVeAll(vep1, aaHat1, 0)

    ks2 = pyKS(32, h2, 22)
    aa2 = ks2.intg(a2, nstp2, 1)[:-1]
    aaHat2, ang2 = ks2.orbitToSlice(aa2)
    vep2 = ks2.veToSliceAll(ve2, aa2, 0)
    aaTilde2 = ks2.reduceReflection(aaHat2)
    veTilde2 = ks2.reflectVeAll(vep2, aaHat2, 0)

    # load Burak's axes.
    # comment out this part if using Fourier modes projection
    ee = sio.loadmat('ee.mat')['ee'][:3].T
    aaTilde1 = np.dot(aaTilde1, ee)
    veTilde1 = np.dot(veTilde1, ee)
    aaTilde2 = np.dot(aaTilde2, ee)
    veTilde2 = np.dot(veTilde2, ee)
    i1 = 0
    i2 = 1
    i3 = 2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aaTilde1[:, i1], aaTilde1[:, i2], aaTilde1[:, i3], 'r')
    ax.plot(aaTilde2[:, i1], aaTilde2[:, i2], aaTilde2[:, i3], 'g')

    spa = 60

    ratio1 = 5
    ah1 = aaTilde1
    veIndex1 = 0
    nor1 = norm(veTilde1[veIndex1::30], axis=1)
    nor1.resize((nor1.size, 1))
    ae1r = ah1 + veTilde1[veIndex1::30] / nor1 / ratio1
    veIndex1 = 1
    nor1 = norm(veTilde1[veIndex1::30], axis=1)
    nor1.resize((nor1.size, 1))
    ae1i = ah1 + veTilde1[veIndex1::30] / nor1 / ratio1

    ratio2 = 5
    ah2 = aaTilde2
    veIndex2 = 0
    nor2 = norm(veTilde2[veIndex2::30], axis=1)
    nor2.resize((nor2.size, 1))
    ae2r = ah2 + veTilde2[veIndex2::30] / nor2 / ratio2
    veIndex2 = 1
    nor2 = norm(veTilde2[veIndex2::30], axis=1)
    nor2.resize((nor2.size, 1))
    ae2i = ah2 + veTilde2[veIndex2::30] / nor2 / ratio2

    for i in np.arange(0, aaTilde1.shape[0], spa):
        a1 = Arrow3D([ah1[i, i1], ae1r[i, i1]], [ah1[i, i2], ae1r[i, i2]],
                     [ah1[i, i3], ae1r[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="m")
        ax.add_artist(a1)
        a1 = Arrow3D([ah1[i, i1], ae1i[i, i1]], [ah1[i, i2], ae1i[i, i2]],
                     [ah1[i, i3], ae1i[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="b")
        ax.add_artist(a1)

    for i in np.arange(0, aaTilde2.shape[0], spa):
        a2 = Arrow3D([ah2[i, i1], ae2r[i, i1]], [ah2[i, i2], ae2r[i, i2]],
                     [ah2[i, i3], ae2r[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="k")
        ax.add_artist(a2)
        a2 = Arrow3D([ah2[i, i1], ae2i[i, i1]], [ah2[i, i2], ae2i[i, i2]],
                     [ah2[i, i3], ae2i[i, i3]],
                     mutation_scale=20, lw=1.0, arrowstyle="-|>", color="c")
        ax.add_artist(a2)

    ax.set_xlabel(r'$e_1$', fontsize=25)
    ax.set_ylabel(r'$e_2$', fontsize=25)
    ax.set_zlabel(r'$e_3$', fontsize=25)
    plt.tight_layout(pad=0)
    plt.show(block=False)

