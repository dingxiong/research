# My research C++ code
This repository contains all my code for research at
[Center for Nonliear Science](http://www.cns.gatech.edu/)
at School of Physics in Georgia Institute of Technology.

It is mainly C++ code and a few Python/Matlab bindings. Large datasets are not committed.
## folder structure 
* **source**   C++ files and their Matlab/Python binding files
* **include**  header files
* **lib**      The shared/static library, compiled Matlab/Python binding library
* **PythonWrapper** The Boost.Python bindings for C++ classes
* **MatlabWrapper** The Matlab bindings for C++ functions

## Requirement
* G++ 4.7 above (supporting C++11)
* Eigen 3.1 or above
* FFTW3 library
* HDF5 

If you need build python binding, you also need
* [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/)
* [Boost.NumPy](https://github.com/ndarray/Boost.NumPy)
