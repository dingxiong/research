import numpy as np
import matplotlib.pyplot as plt
import h5py

N = 256;

f = h5py.File('/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5', 'r')
a0 = f['/req1/a0']


taa = a0; taa = taa[::2]+1j*taa[1::2];
ia = np.fft.ifft(taa, axis=0)

fig = plt.figure(figsize=(5,5));
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 50, 256), abs(ia), 'c-', lw = 2.0)
ax.plot(np.linspace(0, 50, 256), ia.real,'r-',lw = 1.0)
ax.plot(np.linspace(0, 50, 256), ia.imag,'b-',lw = 1.0)
ax.locator_params(nbins=5)

plt.tight_layout(pad=0)
plt.show()
f.close()
