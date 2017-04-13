Author: Xiong Ding
Fri Dec 12 11:34:10 EST 2014
==============================
The format of Flqouet exponents:
[mu, w]: Lambda = exp(mu*T + 1i * w)

==============================
converge: 
	  contains the information about convergence of PED on 
	  upos.
120 : angle distriubtion for T < 120

============================
Mon Jan  5 12:46:44 EST 2015

a
T
nstp 
theta
err

Ruslan : Ruslan's original data
myN32 : my data for truncation N = 32
ks22h02t120x64.h5 : the refined the data with h = 0.02
ks22h005t120x64.h5 : the refined the data with h = 0.005
ks22h001t120x64.h5 : the refined the data with h = 0.001

ks22h001t120x64E.h5
ks22h001t120x64EV.h5 : only the FE/FVs of the first 200 rpo/ppo are calculated

ks22Reqx64.h5 : Equilibria and relative equilibria for L = 22, N = 64
	      a : state variable 62 x 1
	      w : phase velocity = 2*pi/L * c
	      err : absolute error of the solution
	      e : real and imaginary part of the stability exponents
	      
