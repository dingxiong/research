import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)

##############################
# prepare the data
"""
pp1 = sio.loadmat('./ppo4/disAng.mat'); 
ang1 = pp1['Sang']; dis1 = pp1['Sdis'];
pp2 = sio.loadmat('./ppo4_old2/disAng.mat')
ang2 = pp2['Sang']; dis2 = pp2['Sdis'];
ang = np.vstack((ang1, ang2)); dis = np.vstack((dis1, dis2));
"""
pp = sio.loadmat('MdisAng_rpo1.mat'); 
dis = pp['Mdis']; ang = np.sin(np.arccos(pp['Mang'])) + 1e-8;
 
Num = 4;
cix = [3, 4, 5, 8]; # column index
spix = [6, 7, 9, 15]; # subspace index
colors = ['r', 'b', 'c', 'm'];
markers =['o', 's', 'v', '*'];

#set up the guide line y=x*10^b
b = -0.7
#######     1st      #################
for i in range(Num):
    ax.scatter(dis, ang[:,cix[i]], s=7, c=colors[i], marker=markers[i], 
               edgecolor='none', label='1-'+str(spix[i]))
#ax.plot([10**-2.5,10**(-1)],[10**(b-2.5), 10**(b-1)], c='k', lw=2)

ax.set_yscale('log')    
ax.set_xscale('log')
ax.set_ylim([1e-5,1])
ax.set_xlim([1e-4,1e-1])

ax.legend(loc='upper left')
#ax.set_title('(a)')

ax.set_ylabel(r'$\sin(\theta)$',size='large')
ax.set_xlabel(r'$||\Delta x||_2$',size='large')

 
##############################################

fig.tight_layout(pad=0)
plt.show()
