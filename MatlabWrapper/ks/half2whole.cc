/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' half2whole.cc ../ksint.cc  -I../../../include -I$XDAPPS/eigen/include/eigen3 -lfftw3
 * */
#include "ksint.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>

using namespace Eigen;

/* transform trajectory into 1st mode slice */
static MatrixXd
orbitToSlice(double *aa, const int N, const int M){
    KS ks(N+2);
    Map<MatrixXd> aa2(aa, N, M);
    return ks.half2whole(aa2);
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // get the pointer and the size of input 
    double *aa = mxGetPr(prhs[0]);
    mwSize N = mxGetM(prhs[0]);
    mwSize M = mxGetN(prhs[0]);
  
    MatrixXd tmp = orbitToSlice(aa, N, M);
    mwSize N2 = tmp.rows();
    mwSize M2 = tmp.cols();
    plhs[0] = mxCreateDoubleMatrix(N2, M2, mxREAL);
    memcpy(mxGetPr(plhs[0]), &tmp(0,0), N2*M2*sizeof(double));
}
