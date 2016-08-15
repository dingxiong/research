/* to comiple:
 * h5c++ -O3 test_CQCGL1dReq.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dReq -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm 
 */
#include "CQCGL1dReq.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

typedef std::complex<double> dcp;


#define N10

int main(){

#ifdef N10
    //======================================================================
    /* test the new constructor */
    const int N = 1024;
    const int L = 30;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, 0.8, -0.1, -0.6, 0);
	
    string file = "/usr/local/home/xiong/00git/research/data/cgl/reqDi.h5";
    ArrayXd a0;
    double wth, wphi, err;
    std::tie(a0, wth, wphi, err) = cgl.readReq(file, "0.060000/1");

    a0 *= 0.316;
    ArrayXXd aa = cgl.intg(a0, 2e-3, 100000, 100);
	
    cout << aa.rows() << 'x' << aa.cols() << endl;
    savetxt("aa.dat", aa);

#endif

    
    return 0;
}
