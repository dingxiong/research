import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """
    The 3d arrow class
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def setvalue(self, x, y, z):
        self._verts3d = x, y, z
##################################################

data = np.load('data.npz')

aaTilde1 = data['aaTilde1']
veTilde1 = data['veTilde1']
aaTilde2 = data['aaTilde2']
veTilde2 = data['veTilde2']
ee = data['ee']

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
