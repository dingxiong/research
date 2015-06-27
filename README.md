# My research C++ code -- no reflection symmetry branch

This branch contains the old implementation of cqcgl1d class,
It does not dealiase the FFT. Also it keeps the `N/2` mode, 
so it has problem with reflection symmetry

## folder structure 
* **source**   C++ files and their Matlab/Python binding files
* **include**  header files
* **lib**      The shared/static library, compiled Matlab/Python binding library

## Requirement
* G++ 4.7 above (supporting C++11)
* Eigen 2.1 or above
* FFTW3 library
* HDF5 

If you need build python binding, you also need
* [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/)
* [Boost.NumPy](https://github.com/ndarray/Boost.NumPy)
