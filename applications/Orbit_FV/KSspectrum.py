import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter

f = h5py.File('../../data/ks22h001t120x64EV.h5')
pp = '/ppo/1/'
e = np.array(f[pp + 'e'])[0]

Nh = 16
x1 = np.kron(np.arange(1, Nh), np.array([1, 1]))
x2 = np.kron(np.arange(1, 2*Nh), np.array([1, 1]))
x3 = np.arange(1, 2*Nh, 0.01)

d = 22
k = 2*np.pi/d*x3
k = k**2-k**4

# create figure
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)

ax.plot(x3, k, 'g--', linewidth=1.8, label=r'$q_k^2 - q_k^4$')
ax.scatter(np.arange(1, 32), e[::2], c=(1, 0, 0), marker='o',
           s=22, edgecolors='none', label=r'$\mu_{2k-1}$')
ax.scatter(np.arange(1, 32), e[1::2], marker='s',
           s=30, facecolors='none', edgecolors='k', label=r'$\mu_{2k}$')
yfmt = ScalarFormatter()
yfmt.set_powerlimits((0, 1))
ax.yaxis.set_major_formatter(yfmt)

ax.set_xlabel('k')
ax.set_yticks((-7e3, -3e3, 0, 1e3))
ax.set_xlim((0, 35))
ax.set_ylim((-7e3, 1e3))
ax.grid('on')

axin = inset_axes(ax, width="45%", height="50%", loc=3)
axin.scatter(np.arange(1, 32), e[::2], c=(1, 0, 0), marker='o',
             s=22, edgecolors='none')
axin.scatter(np.arange(1, 32), e[1::2], c=(1, 0, 0), marker='s',
             s=30, facecolors='none', edgecolors='k')
axin.set_xlim(0.5, 4.5)
axin.set_ylim(-0.4, 0.1)
axin.yaxis.set_ticks_position('right')
axin.xaxis.set_ticks_position('top')
axin.set_xticks((1, 2, 3, 4))
# axin.set_yticks((-0.3,-0.2,-0.1,0,0.1))
axin.grid('on')

mark_inset(ax, axin, loc1=1, loc2=2, fc="none")

ax.legend()

fig.tight_layout(pad=0)
plt.show()
