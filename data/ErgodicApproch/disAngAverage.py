import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d;

fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot(111)

pp = sio.loadmat('./ppo4/angAver.mat');
dis = pp['x'][:,0]; ang = pp['angleAverage'];
logDis = np.linspace(np.log10(dis[0]), np.log10(dis[-1]), 100)
Ncol = 9 # ang.shape[1];
cix =  [0, 1, 2, 3, 4, 6, 8, 9, 10]
spix = [3, 4, 5, 6, 7, 9, 13, 15, 21]
colors = ['b','g', 'm','c', 'r', 'DarkGreen', 'Brown', 'Purple', 'Navy']

##############################################
for i in range(Ncol):
    f = interp1d(np.log10(dis), np.log10(ang[:,cix[i]]))
    ax.plot(10**logDis, 10**(f(logDis)), c = colors[i], ls = '--' );
    ax.scatter(dis, ang[:,cix[i]], s=30,  c = colors[i],
               edgecolor='none', label='1-'+str(spix[i]))
    if i < 6:
        ax.text(dis[0]-0.0005, ang[0, cix[i]], str(spix[i]))
    else :
        ax.text(dis[7], ang[7, cix[i]], str(spix[i]))
    
ax.set_yscale('log')    
ax.set_xscale('log')
#ax.legend(loc='upper left')
#ax.set_title('(a)')
ax.set_ylabel(r'$< \sin(\theta) >$',size='large')
ax.set_xlabel(r'$||\Delta x||_2$',size='large')

fig.tight_layout(pad=0)
plt.show()
