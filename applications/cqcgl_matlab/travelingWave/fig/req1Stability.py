import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

N = 256;

f = h5py.File('/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5', 'r')
valr = f['/req1/valr']; n1= valr.shape[0]
vali = f['/req1/vali']
vr = np.zeros((n1,), np.double); 
vi = np.zeros((n1,), np.double);
valr.read_direct(vr); vali.read_direct(vi);
ve = vr+1j*vi

ia = ve

fig = plt.figure(figsize=(8,5));
ax = fig.add_subplot(111)
#ax.plot(range(1,2*N+1), ia.real, 'r.')
ax.scatter(range(1,2*N+1), ia.real, s=4, c = 'r', marker='o', edgecolor='none')
ax.locator_params(nbins=5)
ax.set_xlim((0,520))
ax.set_ylim((-35,1))
ax.grid('on')

ax2in=inset_axes(ax,width="45%",height="50%",loc=3)
ax2in.scatter(np.arange(1,11), ia[:10].real, c=(1,0,0), marker='o',
              s=22,edgecolors='none')
ax2in.set_xlim(0.5,10.5)
ax2in.set_ylim(-0.2,0.4)
ax2in.yaxis.set_ticks_position('right')
ax2in.xaxis.set_ticks_position('top')
ax2in.set_xticks((1,3,6,9))
#ax2in.set_yticks((-0.3,-0.2,-0.1,0,0.1))
ax2in.grid('on')

mark_inset(ax,ax2in,loc1=1,loc2=2,fc="none")


plt.tight_layout(pad=0)
plt.show()
f.close()
