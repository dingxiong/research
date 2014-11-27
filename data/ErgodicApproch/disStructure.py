import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)

dis = np.loadtxt('./ppo4/dis_ppo4')
No = np.loadtxt('./ppo4/No_ppo4', dtype =int);
ix = 25; # shadowing incidence.
h = 0.1;

start = No[:ix].sum() # number of point before what we want
x = h*np.arange(No[ix]);
y = dis[start:start+No[ix]];

ax.plot(x, y, c='b', lw = 1);
ax.quiver(10, 0.008, 0, 0.1)
ax.quiver(18, 0.03, 0, 0.1)
ax.set_yscale('log')

ax.set_xlabel('t')
ax.set_ylabel(r'$||\Delta x||_2$') 
##############################################

fig.tight_layout(pad=0)
plt.show()

    



