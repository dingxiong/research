# My research C++ code -- old FFT branch

This branch contains the old implementation of cqcgl1d class,
in which FFT serves as subclass. Also, it lacks the ability to
evaluate `Jacobian * vector`.  

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
