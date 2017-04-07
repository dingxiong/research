/* to comiple:
 * h5c++ -O3 test_CQCGL1dSubReq.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dSubReq -lCQCGL1dSub -lCQCGL1dReq -lCQCGL1d -ldenseRoutines -literMethod -lmyH5 -lfftw3 -lm
 */
#include "CQCGL1dSubReq.hpp"			
#include "CQCGL1dReq.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;
using namespace MyH5;

typedef std::complex<double> dcp;


#define N10

int main(int argc, char **argv){

#ifdef N10
    //======================================================================
    // use the full-space req as initial condition to find req in the
    // symmetric subspace.
    const int N = 1024, L = 50;
    double Bi = 0.8, Gi = -0.6;
    CQCGL1dReq cgl0(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    CQCGL1dSubReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    string fileName = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = cgl0.read(file, cgl.toStr(Bi, Gi, 1));
    a0 = a0.head(cgl.Ndim);

    ArrayXd a;
    double wphi, err;
    int flag;
    std::tie(a, wphi, err, flag) = cgl.findReq_LM(a0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g\n", wphi, err);
    H5File fout("sub.h5", H5F_ACC_TRUNC);
    if (flag == 0) cgl.write(fout, cgl.toStr(Bi, Gi, 1), a, wphi, err);    
#endif

    return 0;
}
