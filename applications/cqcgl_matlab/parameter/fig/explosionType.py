import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable

aa = sio.loadmat('../mat/explosionType.mat')
S = aa['aaS'] ; 
As1 = aa['aaAs1']
As2 = aa['aaAs2']

h = 0.01

taa = As2; taa = taa[::2,:]+1j*taa[1::2,:];# change taa to plot differnt types.
#################### fig 1 ##########
fig = plt.figure(figsize=(4,5));
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('t')
im = ax.imshow(abs(np.fft.ifft(taa, axis=0)).T, cmap=plt.get_cmap('get'),
                extent=(0,50,0, h*taa.shape[1]), aspect='auto',
                origin ='lower')
ax.grid('on')
dr = make_axes_locatable(ax)
cax = dr.append_axes('right', size = '5%', pad =0.05)
bar = plt.colorbar(im, cax=cax, ticks=[0, 3])
########################################
fig.tight_layout(pad=0)
plt.show()
