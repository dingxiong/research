import numpy as np
import matplotlib.pyplot as plt
import h5py

N = 256;

f = h5py.File('/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5', 'r')
ver = f['/req1/vecr']; n1, n2 = ver.shape
vei = f['/req1/veci']
vr = np.zeros((n1,n2), np.double); 
vi = np.zeros((n1,n2), np.double);
ver.read_direct(vr); vei.read_direct(vi);
ve = vr+1j*vi

taa = ve.T
ia = np.fft.ifft(taa, axis=0)

fig = plt.figure(figsize=(5,5));
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 50, 2*N), abs(ia[:,0]), 'r-', lw = 2.0,
        label=r"$|v_1|$")
ax.plot(np.linspace(0, 50, 2*N), abs(ia[:,1]), 'b--', lw = 2.0,
        label=r"$|v_2|$")
ax.locator_params(nbins=5)

plt.tight_layout(pad=0)
plt.legend()
plt.show()
f.close()
