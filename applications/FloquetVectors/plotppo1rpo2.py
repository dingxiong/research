import h5py
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import display
# from IPython.html.widgets import interact
from py_ks import *
from personalFunctions import *

##################################################
# load data
f = h5py.File('../../data/myN32/ks22h02t100EV.h5', 'r')

dataName1 = '/ppo/2/'
a1 = f[dataName1 + 'a'].value
T1 = f[dataName1 + 'T'].value[0]
nstp1 = np.int(f[dataName1 + 'nstp'].value[0])
h1 = T1 / nstp1
e1 = f[dataName1 + 'e'].value
ve1 = f[dataName1 + 've'].value

dataName2 = '/rpo/1/'
a2 = f[dataName2 + 'a'].value
T2 = f[dataName2 + 'T'].value[0]
nstp2 = np.int(f[dataName2 + 'nstp'].value[0])
h2 = T2 / nstp2
e2 = f[dataName2 + 'e'].value
ve2 = f[dataName2 + 've'].value


##################################################
# plot orbits

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
ax.plot(aaHat1[:, 0], aaHat1[:, 2], aaHat1[:, 3], 'r')
ax.plot(aaHat2[:, 0], aaHat2[:, 2], aaHat2[:, 3], 'g')

spa = 70

ratio1 = 5
ah1 = aaHat1
nor1 = norm(vep1[0::30], axis=1)
nor1.resize((nor1.size, 1))
ae1 = aaHat1 + vep1[0::30] / nor1 / ratio1

ratio2 = 5
ah2 = aaHat2
nor2 = norm(vep2[0::30], axis=1)
nor2.resize((nor2.size, 1))
ae2 = aaHat2 + vep2[0::30] / nor2 / ratio2

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
