Thu Oct 23 14:53:38 EDT 2014
###################################################
rpoT2X1.h5: Hopf cycles for different di. Group name is di

	 fixed paramters:
	 N = 1024
	 L = 30
	 b = 4.0
	 c = 0.8
	 dr = 0.01

	Each group contains a few indices. Each index contains	  
	      x    : initial condition
	      T    : period
	      nstp : integration steps
	      th   : shift of translation (in angle)
	      phi  : shift of complex phase
	      err  : absolute err

rpoT2X1EV15.h5 : save with rpoT2X1.h5, but has Floquet exponents and
                 Floquet vectors.
		 e: [mu, omega] => multiplier = exp(T*mu + 1i*omega)
		 v: vectors. For complex pairs, real and imaginary
		    parts are separated.

rpoT2X1EV30.h5 : save with rpoT2X1EV15.h5.
	       But this one has 30 leading exponents/vectors.

Similar with
rpoT2X1EV16.h5
rpoT2X1EV31.h5

############################################################
	      
reqDi.h5 : store soliton solutions for different di. Group name is di	
	 
	 fixed paramters:
	 N = 1024
	 L = 30
	 b = 4.0
	 c = 0.8
	 dr = 0.01

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
	0.36 -- 0.96       : one unstable conjugate pair.
	0.081 --0.35       : stable solition.
	0.0795 -- -0.08    : one unstable conjugate pair
	0.026  -- 0.079    : more than one unstable pair. 
	       	  	     Some may have 20 pairs.

############################################################
req_N: previous soliton solutions for different N. for L = 50


############################################################

reqBiGi.h5 / reqBiGiEV.h5 : relative equilibria in the parameter space (Bi, Gi)
	   Group structure is /Bi/Gi/index/

############################################################
rpoBiGi.h5 / rpoBiGiEV.h5 : relative periodic orbits in the parameter space (Bi, Gi)
	   Group structure is /Bi/Gi/index/
	   
	   fixed paramters:
	    N = 1024, L = 50	
	    mu = -0.1
	    Dr = 0.125, Di = 0.5
	    Br = 1, Gr = -0.1

	   data structure :

	   T     :    [double]       period
	   x     :    [1365 x 1]     state [1362 x 1], T, theta, phi
	   th    :    [double]       translation velocity
	   phi   :    [double]       phase rotation velocity
	   nstp  :    [int]          integration steps
	   err   :    [double]       error of this orbit
	   er    :    ?              real part of Floquet multiplier
	   ei    :    ?              imaginary part of Floquet multiplier
	   v     :    ?              Floquet vectors in real format

	   The eigenvalues and eigenvectors are obtained by Arpack package.

	   Range (Bi, Gi) = [1.9, 5.7] x [-5.6, -4.0] as a rectangle area

rpoHopfBiGi.h5 : rpos close to the Hopf bifurcation line.

############################################################

req2dBiGi.h5 : two dimensional relative equilibria in the parameter space (Bi, Gi)
	   
