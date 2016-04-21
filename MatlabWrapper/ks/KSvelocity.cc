/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' KSvelocity.cc ../ksint.cc  -I../../../include -I$XDAPPS/eigen/include/eigen3 -lfftw3
 *
 * usage :
 * >> v = ksVelocity(aa(:,i) );
 * */
#include "ksint.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;

static MatrixXd
KSvelocity(double *x, const int N){
    KS ks(N+2);
    Map<VectorXd> x2(x, N);
    return ks.velocity(x2);
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // get the pointer and the size of input 
    double *x = mxGetPr(prhs[0]);
    mwSize N = mxGetM(prhs[0]);
    mwSize M = mxGetN(prhs[0]);
    assert(M2 = 1);
    
    plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);

    MatrixXd v = KSvelocity(x, N);
    memcpy(mxGetPr(plhs[0]), &v(0), N*M*sizeof(double));
}
