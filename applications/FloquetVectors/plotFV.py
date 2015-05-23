
# coding: utf-8

# In[51]:

import h5py
import numpy as np
import matplotlib.pyplot as plt

def magvalue(x):
	return np.abs(x[0::2] + 1j * x[1::2]);
	

# In[52]:

f = h5py.File('../../data/myN32/ks22h02t100EV.h5');
ve = np.array(f['/ppo/1/ve']).T;
N, M = ve.shape; N = np.int(np.sqrt(N));
ve = ve[:,0]; ve.resize(N, N);


### Plot the physical Floquet vectors

# In[53]:

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(range(1, N/2+1), magvalue(ve[:,0]), '-o', c='r',label=r'$V_{1r}$')
ax.plot(range(1, N/2+1), magvalue(ve[:,1]), '-s', c='b',label=r'$V_{1i}$')
ax.plot(range(1, N/2+1), magvalue(ve[:,7]), '-^', c='g',label=r'$V_{8}$')
ax.plot([4.5, 4.5], [0,0.7], '--', c='k',lw=3)
ax.grid('on')
ax.set_xlabel('Fourier mode index')
ax.set_ylabel('amplitute')
ax.legend()
plt.tight_layout(pad=0)
plt.show()

# The black dashed line separate the physical (first 8) and transient modes ( 9 to 30). 
# since 
# $$
# u(x_{n},t)=\!\!\sum_{k=-N/2+1}^{N/2}\!\!a_{k}(t)e^{iq_{k}x_{n}}
# $$
# Denote $a_k = b_k + ic_k$, so we has state space $[b_1, c_1, b_2, c_2, \cdots]$.  X-axis is the exponent index of this state space

### Plot the unphysical Floquet vectors

# In[63]:

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(range(1, N/2+1), magvalue(ve[:,8]), '-o', c='r',label=r'$V_{9}$')
ax.plot(range(1, N/2+1), magvalue(ve[:,17]), '-s', c='b',label=r'$V_{18}$')
ax.plot(range(1, N/2+1), magvalue(ve[:,24]), '-v', c='c',label=r'$V_{25}$')
ax.plot(range(1, N/2+1), magvalue(ve[:,29]), '-^', c='g',label=r'$V_{30}$')
ax.plot([4.5, 4.5], [0,1], '--', c='k',lw=3)
ax.grid('on')
ax.set_xlabel('Fourier mode index')
ax.set_ylabel('amplitute')
ax.legend(loc='best')
plt.tight_layout(pad=0)
plt.show()


# The unphysical modes look like pure Fourier modes.

# In[ ]:



