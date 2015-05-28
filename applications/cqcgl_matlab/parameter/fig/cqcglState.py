import numpy as np
import matplotlib.pyplot as plt

N = 1024; M = 10000;
AA = np.fromfile('aa.bin', np.double, 2*N*M).reshape(M,2*N).T;
Ar = AA[0::2,:]; Ai = AA[1::2, :]; Ama = abs(Ar+1j*Ai); Ama=Ama.T;
Maxf = 7000 ;
mag = 2;
h = 0.005;

fig = plt.figure(figsize=(6,8));
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_yticks([0, h*mag*Maxf])
ax.set_yticklabels(["$0$", "35"])
#ax.set_ylim((-0.5, 3.5))


ax.imshow(Ama[:,0:Maxf], cmap=plt.get_cmap('seismic'), extent=(0,50,0,h*mag*Maxf));
ax.grid('on')
fig.tight_layout(pad=0)
plt.show()
