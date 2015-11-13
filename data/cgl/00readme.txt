Thu Oct 23 14:53:38 EDT 2014
###################################################
req.h5: relative equilibria

	Each group contains:	  
	      a0   : initial condition
	      wth  : shift of translation (in angle)
	      wphi : shift of complex phase
	      err  : absolute err

reqOrigin.h5: the origin data which contains a lot of plan waves.

##################################################
folder rpo:

      rpo0799T2X1.h5 : periodic orbits for di = -0.0799, h = 0.0002, 1 column of a.
      
############################################################
files:
	      
reqDi.h5 : store soliton solutions for different di	
	 
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
