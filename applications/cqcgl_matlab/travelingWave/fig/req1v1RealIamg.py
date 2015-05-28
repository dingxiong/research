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
ax.plot(np.linspace(0, 50, 2*N), ia[:,0].real, 'r-', lw = 2.0,
        label='real')
ax.plot(np.linspace(0, 50, 2*N), ia[:,0].imag, 'b--', lw = 2.0,
        label='imag')
ax.locator_params(nbins=5)

ax.set_ylim((-0.008, 0.008))

plt.tight_layout(pad=0)
plt.legend()
plt.show()
f.close()
