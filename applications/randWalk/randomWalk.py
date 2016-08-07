from personalFunctions import *
from scipy.spatial.distance import pdist

if False:
    Np = 10000                      # number of points
    Ns = 1000                       # number of steps

    z = np.zeros(Np, dtype=np.complex)
    s = np.zeros(Ns)
    for i in range(Ns):
        z += np.exp(rand(Np)*2j*np.pi)
        s[i] = np.mean(np.abs(z)**2)

    a2 = np.zeros(Np)
    z2 = np.zeros(Np, dtype=np.complex)
    s2 = np.zeros(Ns)
    for i in range(Ns):
        for j in range(Np):
            t = rand()
            d = min([abs(t-a2[j]), abs(t-a2[j]+1), abs(t-a2[j]-1)])
            while d < 0.1:
                t = rand()
                d = [abs(t - a2[j]), abs(t-a2[j]+1), abs(t-a2[j]-1)]
            a2[j] = t
            z2[j] += np.exp(t*2j*np.pi)
        s2[i] = np.mean(np.abs(z2)**2)


ss = np.loadtxt('ss.dat')
fig, ax = pl2d(size=[8, 6], labs=[r'$N$', r'$<r^2>$'], axisLabelSize=25,
               xlim=[0, 1000], ylim=[0, 1000], tickSize=None)
for i in range(ss.shape[1]):
    ax.plot(ss[:, i], lw=2)
ax2d(fig, ax)
