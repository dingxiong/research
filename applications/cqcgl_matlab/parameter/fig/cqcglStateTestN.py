import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

aa = sio.loadmat('../mat/cqcglStateTestN.mat')
a1 = aa['uu128'];
a2 = aa['uu256'];
a3 = aa['uu512'];

h = 0.01;

fig = plt.figure(figsize=(8,4));
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

########## fig 1 ##########
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.imshow(abs(np.fft.ifft(a1, axis = 0)).T, cmap=plt.get_cmap('jet'),
          extent=(0,128,0, h*a1.shape[1]), aspect='auto', origin ='lower');
ax.grid('on')

########## fig 2 ##########
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.imshow(abs(np.fft.ifft(a2, axis = 0)).T, cmap=plt.get_cmap('jet'),
          extent=(0,256,0, h*a1.shape[1]), aspect='auto', origin ='lower');
ax2.grid('on')

########## fig 3 ##########
ax3.set_xlabel('x')
ax3.set_ylabel('t')
im = ax3.imshow(abs(np.fft.ifft(a3, axis = 0)).T, cmap=plt.get_cmap('jet'),
          extent=(0,512,0, h*a1.shape[1]), aspect='auto', origin ='lower');
ax3.grid('on')
dr3 = make_axes_locatable(ax3)
cax3 = dr3.append_axes('right', size = '10%', pad =0.05)
bar = plt.colorbar(im, cax=cax3, ticks=[0, 2.5])

##############################
fig.tight_layout(pad=0)
plt.show()
