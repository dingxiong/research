import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)

dis = np.loadtxt('./cases32modes/case4rpo6x10sT30/dis')
No = np.loadtxt('./cases32modes/case4rpo6x10sT30//No', dtype =int);
ix = 13; # shadowing incidence.
h = 0.1;

start = No[:ix].sum() # number of point before what we want
x = h*np.arange(No[ix]);
y = dis[start:start+No[ix]];

ax.plot(x, y, c='b', lw = 1);
#ax.quiver(22, 2e-3, 0, 0.1)
#ax.quiver(18, 2e-3, 0, 0.1)
ax.text(8.5, 6e-3, 'B')
ax.text(16.5, 1.6e-2, 'C')

ax.set_ylim([4e-3, 1e-1])
ax.set_yscale('log')

ax.set_xlabel('t')
ax.set_ylabel(r'$||\Delta x||_2$') 
##############################################

fig.tight_layout(pad=0)
plt.show()

    



