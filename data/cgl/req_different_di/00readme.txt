############################################################
Author : Xiong Ding
Date : Fri Nov  6 17:12:21 EST 2015

############################################################
This folder contains the initial conditions for relative equilibria
of cqcgl with fixed paramters:
   
   N  = 1024
   d  = 30
   b  = 4
   c  = 0.8
   dr = -0.01
    
di can be read from the file name: req0799.h5 => di = -0.0799

data structure:
     a     : initial condition
     wth   : translational velocity
     wphi  : phase velocity
     err   : error of these req
     er    : real part of stability exponents
     ei    : imaginary part of stability exponents
     vr    : real part of stability vectors (if not full, then the leading few)
     vi    : imaginary part of stability vectors


classification:
	-0.081              : stable solition
	-0.08 -- -0.0795    : one unstable conjugate pair
	-0.079 -- 	
