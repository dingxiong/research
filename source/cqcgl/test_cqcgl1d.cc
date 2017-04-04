/* to comiple:
 * h5c++ -O3 test_cqcgl1d.cc  -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++0x -lCQCGL1d -lsparseRoutines -ldenseRoutines -lmyH5 -lped -lmyfft -lfftw3 -lm 
 */
#include "CQCGL1d.hpp"
#include "myH5.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace MyH5;
using namespace denseRoutines;

typedef std::complex<double> dcp;


#define N10

int main(){

#ifdef N10
    //======================================================================
    /* test the new constructor */
    const int N = 1024;
    const int L = 50;
    // CQCGL1d cgl(N, L, 4.0, 0.8, 0.01, 0.06, -1);
    CQCGL1d cgl(N, L, -1, 1, 4.0, 1, 0.8, -0.01, -0.06, -1);
	
    int nstp = 1;
	
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl.Config2Fourier(A0);
	
    ArrayXXd aa = cgl.intg(a0, 0.0001, nstp, 1);
	
    cout << aa.rows() << 'x' << aa.cols() << endl;
    cout << aa << endl;
#endif

    
    return 0;
}
