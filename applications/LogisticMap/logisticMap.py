###################################################
# This file contains all the related functions to
# investigate the escape rate in Logistic map using
# cycle expansion.
# please complete the experimental part
###################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
from personalFunctions import *

class Logistic:
    def __init__(self, A):
        self.A = A

    def oneIter(self, x):
        return self.A * (1.0 - x) * x

    def multiIters(self, x, n):
        y = np.zeros(n+1)
        y[0] = x
        tmpx = x
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            y[i+1]=tmpx
            
        return y

    def df(self, x):
        return self.A*(1-2*x)
    
    def dfn(self, x):
        n = np.size(x)
        multiplier = 1.0
        for i in range(n):
            multiplier *= self.df(x[i])

        return multiplier

    def plotIni(self):
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$x_n$', fontsize=30)
        ax.set_ylabel(r'$x_{n+1}$', fontsize=30)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.gca().set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        x = np.linspace(0, 1, 200)
        y = self.oneIter(x)
        ax.plot(x, y, lw=2, c='b')
        ax.plot(x, x, lw=2, c='k', ls='--')
        
        return fig, ax
        

    def plotEnd(self, fig, ax):
        ax2d(fig, ax)

    def plotIter(self, ax, x0, n):
        x, y = x0, self.oneIter(x0)
        for i in range(n):
            print i
            newy = self.oneIter(y)
            ax.scatter(x, y, c='g', edgecolors='none', s=100)
            ax.plot([x, y], [y, y], lw=1.5, c='r')
            ax.plot([y, y], [y, newy], lw=1.5, c='r')
            x, y = y, newy

if __name__ == '__main__':
    """
    experiment
    """
    case = 10

    if case == 10:
        A = 5.0
        order = 4
        lm = Logistic(A)
        f = lambda x : lm.multiIters(x, order)[-1] - x
        x = fsolve(f, 0.2)
        xn = lm.multiIters(x, order-1)
        mp = lm.dfn(xn)
        np.set_printoptions(precision=16)
        print lm.multiIters(x, order), mp

    if case == 20:
        # Floquet multipliers for the periodic orbits up to length 4
        # { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111} }
        mp = [[-4., 6.0], 
              [-20.0],    
              [-114.954535015, 82.9545350148],
              [ -684.424135353, 485.09371391, -328.669578465]
              ]
        lm = Logistic(6.0)
        z = sp.Symbol('z')
        zeta = 1 
        order = 4 # cycle expansion order
        for i in range(order):
            for j in range(np.size(mp[i])):
                zeta = zeta * (1 - z**(i+1)/np.abs(mp[i][j]))
                zeta = ( zeta.expand() + sp.O(z**(order+1)) ).removeO() # remove higher orders
        print "zeta function at order: ", order
        print zeta
        
        # for efficicy, we choose to use np.roots() instead sp.solve() to find the zero points
        coe = sp.Poly(zeta, z).coeffs() # get the coefficients => a_n, a_{n-1}, ..., a_0
        zps = np.roots(coe) # find the zeros points of zeta function
        leig = np.max(1.0 / zps),  # get the leading eigenvalues
        print leig, np.log(leig)

    if case == 30:
        # Floquet multipliers for the periodic orbits up to length 4
        # { 0, 1, 01, 001, 011, 0001, 0011, 0111 }
        
        # multipliers for A = 6.0
        mp = [[-4., 6.0],
              [-20.0],
              [-114.954535015, 82.9545350148],
              [ -684.424135353, 485.09371391, -328.669578465]
              ]
     
  
        lm = Logistic(6.0)
        z = sp.Symbol('z')
        trace = 0
        order = 4
        for i in range(order):
            for j in range(np.size(mp[i])):
                for r in range(order/(i+1)):
                    trace += (i+1) * z**((r+1)*(i+1))/(np.abs(mp[i][j]**(r+1)-1))
                
        
        C = sp.Poly(trace, z).coeffs() # obtain coefficients of trace: C_n, C_{n-1}, C_1
        C = C[::-1] # reverse the order => C_1, C_2,... C_n
        Q = np.zeros(np.size(C))
        Q[0] = C[0]
        for i in range(order):
            Q[i] = C[i]
            for j in range(i):
                Q[i] -= (C[j]*Q[i-1-j])  
            Q[i] = Q[i] / np.double(i+1) 
        
        det = np.append(1.0, -Q) # obtain the coefficients of spectral determinant
        print det
        zps = np.roots(det[::-1]) # find the zeros points of zeta function
        leig = np.max(1.0 / zps),  # get the leading eigenvalues
        print leig, np.log(leig)


    if case == 35:
        """
        periodic orbits for A = 5.0
        """
        

    if case == 40:
        # Floquet multipliers for the periodic orbits up to length 5
        # { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111},
        # {00001, 00011, 00101, 00111, 01011, 01111} }     
        # multipliers for A = 5.0   
        mp = [[-3.0, 5.0],
              [-11.0],
              [-49.4264068712, 35.4264068712],
              [ -240.961389769, 167.507590647, -103.546200878],
              [-1198.49148617, 827.861204429, 547.72002265, -481.824783312 ,-386.519433432, 313.254476004]
              ]
        lm = Logistic(5.0)
        z = sp.Symbol('z')
        trace = 0
        order = 3
        for i in range(order):
            for j in range(np.size(mp[i])):
                for r in range(order/(i+1)):
                    trace += (i+1) * z**((r+1)*(i+1))/(np.abs(mp[i][j]**(r+1)-1))
                
        
        C = sp.Poly(trace, z).coeffs() # obtain coefficients of trace: C_n, C_{n-1}, C_1
        C = C[::-1] # reverse the order => C_1, C_2,... C_n
        Q = np.zeros(np.size(C))
        Q[0] = C[0]
        for i in range(order):
            Q[i] = C[i]
            for j in range(i):
                Q[i] -= (C[j]*Q[i-1-j])  
            Q[i] = Q[i] / np.double(i+1) 
        
        det = np.append(1.0, -Q) # obtain the coefficients of spectral determinant
        print det
        zps = np.roots(det[::-1]) # find the zeros points of zeta function
        leig = np.max(1.0 / zps),  # get the leading eigenvalues
        print leig, np.log(leig)
