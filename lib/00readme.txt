Wed Jul  9 15:29:09 EDT 2014
Xiong Ding
-----------------------------

KS solver:

In order to use the shared library and the Matlab interface, you
need to set shell enviorment LD_LIBRARY_PATH including the 
"libkssolve.so" when link you C/C++ code or start you Matlab.

To use the python interface, "kssolve.py" mush be in the same folder as
"libkssolve.so". 

The shared library "libkssolve.so" is tested on my PC and the Mint machine.
It works on Matlab 2013a/b, but not Matlab 2009b. If it does not work 
in your platform, please compile the library from the source code.


Caution:

No guarantee that the C++ code is correct, you need to varify the result of mex 
file with the original Matlab code in the backup folder.

------------------------------
Thu Dec 18 11:05:46 EST 2014

file details:
     
libksint.so/.a : new KS integrator based on library Eigen3. Both KS and KSM1 
	       classes are included. 
libped.so/.a : Periodic eigendecompostion.
libreaks.a : read h5 file

======================================================================
orbitToSlice.mexa64
>> [aaReduced, theta] = orbitToSlice(aa);
