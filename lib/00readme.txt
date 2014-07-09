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
