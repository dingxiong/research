Author: Xiong Ding
Fri Dec 12 11:34:10 EST 2014
==============================
The format of Flqouet exponents:
[mu, w]: Lambda = exp(mu*T + 1i * w)

==============================
copyh5.sh : script to combine data for T < 100 and  100 < T < 120
mergeh5.sh : script to merge two h5 files by some selection rule
readksh5.m : Matlab script to export h5 contents to Matlab structure.

converge: 
	  contains the information about convergence of PED on 
	  upos.
120 : angle distriubtion for T < 120

Note: in the H5 files, nstp is double.

============================
Mon Jan  5 12:46:44 EST 2015

ks22h1t120.h5, ks22h1t120x64.h5 : Ruslan's original data
ks22h02t120x64.h5 : the refined the data with h = 0.02
ks22h005t120x64.h5 : the refined the data with h = 0.005
ks22h001t120x64.h5 : the refined the data with h = 0.001

