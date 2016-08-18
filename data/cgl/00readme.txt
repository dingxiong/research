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
reqBiGi.h5: req in the Bi -- Gi parameter plane

reqBiGi_bak1.h5 : only one req with index 1 and 2

reqBiGi_bak2.h5 : index 1 is extended

reqBiGi_bak3.h5 : index 2 is extended

reqBiGi_bak4.h5 : index 1 is further extended but failed at 
		Bi = -2.1, Gi = -5.4

reqBiGi_bak4.h5 : index 2 is further extended but failed at 
		Bi = -2.1, Gi = -5.4
