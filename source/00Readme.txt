Xiong Ding
Tue Nov 25 19:35:09 EST 2014
==============================

'kssolve': 
	   the old version of KS integrator, which is written in
	    plain C++.
'ksint': 
	 new verison of KS integrator, developed with template 
	 libarary Eigen.
	 integrator on the 1st mode slice with Jacobian is not
	 finished yet.
'symm': 
	symmetry functions of KS system. ==> will be absorbed into
	'ksint'.

'poinc.cc': Test program for integrator onto Poincare section in
	    KS, not succeed.

'ksDimension.cc' : the main file of conducting experiments of 
		 dimension staff in KS. 
'performanceDimension.png': the gprof performance analysis of 
			    'ksDimension.cc'.
