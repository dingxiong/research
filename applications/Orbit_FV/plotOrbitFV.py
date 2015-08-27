import h5py
import numpy as np
import matplotlib.pyplot as plt
from py_ks import *
from personalFunctions import *

def magvalue(x):
	return np.abs(x[0::2] + 1j * x[1::2]);

case = 5

if case == 1:
        f = h5py.File('../../data/myN32/ks22h02t100EV.h5')
        ve = np.array(f['/ppo/1/ve']).T
        N, M = ve.shape
        N = np.int(np.sqrt(N))
        ve = ve[:, 0]
        ve.resize(N, N)
        
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(range(1, N/2+1), magvalue(ve[:, 0]), '-o', c='r', label=r'$V_{1r}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 1]), '-s', c='b', label=r'$V_{1i}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 7]), '-^', c='g', label=r'$V_{8}$')
        ax.plot([4.5, 4.5], [0, 0.7], '--', c='k', lw=3)
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
        ax.plot(range(1, N/2+1), magvalue(ve[:, 8]), '-o', c='r', label=r'$V_{9}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 17]), '-s', c='b', label=r'$V_{18}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 24]), '-v', c='c', label=r'$V_{25}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 29]), '-^', c='g', label=r'$V_{30}$')
        ax.plot([4.5, 4.5], [0,1], '--', c='k', lw=3)
        ax.grid('on')
        ax.set_xlabel('Fourier mode index')
        ax.set_ylabel('amplitute')
        ax.legend(loc='best')
        plt.tight_layout(pad=0)
        plt.show()


if case == 2:
        """
        plot Fvs for N = 64
        """        
        f = h5py.File('../../data/myN32/ks22h02t100EV.h5')
        ve = np.array(f['/ppo/1/ve']).T
        N, M = ve.shape
        N = np.int(np.sqrt(N))
        ve = ve[:, 0]
        ve.resize(N, N)
        
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(range(1, N/2+1), magvalue(ve[:, 0]), '-o', c='r', label=r'$V_{1r}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 1]), '-s', c='b', label=r'$V_{1i}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 7]), '-^', c='g', label=r'$V_{8}$')
        ax.plot([4.5, 4.5], [0, 0.7], '--', c='k', lw=3)
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
        ax.plot(range(1, N/2+1), magvalue(ve[:, 8]), '-o', c='r', label=r'$V_{9}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 17]), '-s', c='b', label=r'$V_{18}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 24]), '-v', c='c', label=r'$V_{25}$')
        ax.plot(range(1, N/2+1), magvalue(ve[:, 29]), '-^', c='g', label=r'$V_{30}$')
        ax.plot([4.5, 4.5], [0,1], '--', c='k', lw=3)
        ax.grid('on')
        ax.set_xlabel('Fourier mode index')
        ax.set_ylabel('amplitute')
        ax.legend(loc='best')
        plt.tight_layout(pad=0)
        plt.show()


if case == 3:
        """
        plot the orbit ppo1 for N = 64
        """
        N = 64
        f = h5py.File('../../data/ks22h001t120x64EV.h5')
        pp = '/ppo/1/'
        T = np.array(f[pp + 'T'])[0]
        nstp = np.int(np.array(f[pp + 'nstp'])[0])
        a0 = np.array(f[pp + 'a'])
        h = T / nstp

        ks = pyKS(N, h, 22)
        aa = ks.intg(a0, nstp*4, 10)
        KSplotColorMapOrbit(aa, [0, 22, 0, 10.25*4])


if case == 4:
        """
        plot the orbit rpo1 for N = 64
        """
        N = 64
        f = h5py.File('../../data/ks22h001t120x64EV.h5')
        pp = '/rpo/1/'
        T = np.array(f[pp + 'T'])[0]
        nstp = np.int(np.array(f[pp + 'nstp'])[0])
        a0 = np.array(f[pp + 'a'])
        h = T / nstp

        ks = pyKS(N, h, 22)
        aa = ks.intg(a0, nstp*2, 10)
        KSplotColorMapOrbit(aa, [0, 22, 0, 16.31*2])

if case == 5:
        """
        plot the Fv of ppo1 for N = 64
        """
        N = 64
        Ndim = N - 2
        f = h5py.File('../../data/ks22h001t120x64EV.h5')
        pp = '/ppo/1/'
        T = np.array(f[pp + 'T'])[0]
        nstp = np.int(np.array(f[pp + 'nstp'])[0])
        a0 = np.array(f[pp + 'a'])
        h = T / nstp

        veAll = np.array(f[pp + 've'])
        Nve = 30
        veid = 30
        ve = veAll[:, (veid-1)*Ndim:veid*Ndim]

        KSplotColorMapOrbit(ve, [0, 22, 0, 10.25],
                            size=[2.5, 6],
                            save=True,
                            name='ppo1Fv30_64', axisOn=False, barOn=False)
        
if case == 6:
        """
        plot the Fv of rpo1 for N = 64
        """
        N = 64
        Ndim = N - 2
        f = h5py.File('../../data/ks22h001t120x64EV.h5')
        pp = '/rpo/1/'
        T = np.array(f[pp + 'T'])[0]
        nstp = np.int(np.array(f[pp + 'nstp'])[0])
        a0 = np.array(f[pp + 'a'])
        h = T / nstp

        veAll = np.array(f[pp + 've'])
        Nve = 30
        veid = 30
        ve = veAll[:, (veid-1)*Ndim:veid*Ndim]

        KSplotColorMapOrbit(ve, [0, 22, 0, 16.31],
                            size=[2.5, 6],
                            save=True,
                            name='rpo1Fv30_64', axisOn=False, barOn=False)

if case == 7:
        """
        plot the Fv power graph of ppo1/rpo1 for N = 64
        """
        N = 64
        Ndim = N - 2
        f = h5py.File('../../data/ks22h001t120x64EV.h5')
        pp = '/ppo/1/'
        T = np.array(f[pp + 'T'])[0]
        nstp = np.int(np.array(f[pp + 'nstp'])[0])
        a0 = np.array(f[pp + 'a'])
        h = T / nstp

        veAll = np.array(f[pp + 've'])
        Nve = 30
        ve = veAll[0]
        
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.set_xlabel('k')
        ax.set_ylabel('power')
        for i in range(Nve):
                x = range(1, Nve/2+1)
                y = ve[Ndim*i:Ndim*(i+1)]
                y = y[::2]**2 + y[1::2]**2
                y = y[:Nve/2]
                if i < 8:
                        ax.plot(x, y, 'r-o', lw=1)
                if i >= 8:
                        ax.plot(x, y, 'b-o', lw=1)

        # ax.set_yscale('log')
        fig.tight_layout(pad=0)
        plt.show()
