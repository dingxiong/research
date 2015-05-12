/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' veToSlice.cc ../ksint.cc  -I../../../include -I$XDAPPS/eigen/include/eigen3 -lfftw3
 *
 * usage :
 * >> vep = veToSlice(ve, aa(:,i) );
 * */
#include "ksint.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;

/* transform trajectory into 1st mode slice */
static MatrixXd
veToSlice(double *ve, double *x, const int N, const int M){
    KS ks(N+2);
    Map<MatrixXd> aa2(ve, N, M);
    Map<VectorXd> x2(x, N);
    return ks.veToSlice(aa2, x2);
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // get the pointer and the size of input 
    double *aa = mxGetPr(prhs[0]);
    double *x = mxGetPr(prhs[1]);
    mwSize N = mxGetM(prhs[0]);
    mwSize M = mxGetN(prhs[0]);
    mwSize N2 = mxGetM(prhs[1]);
    mwSize M2 = mxGetN(prhs[1]);
    assert(N == N2 && M2 = 1);
    
    plhs[0] = mxCreateDoubleMatrix(N, M, mxREAL);

    MatrixXd vep = veToSlice(aa, x, N, M);
    memcpy(mxGetPr(plhs[0]), &vep(0,0), N*M*sizeof(double));
}
