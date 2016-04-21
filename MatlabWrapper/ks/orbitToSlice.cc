/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' orbitToSlice.cc ../ksint.cc  -I../../../include -I$XDAPPS/eigen/include/eigen3 -lfftw3
 * */
#include "ksint.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Map;

/* transform trajectory into 1st mode slice */
static std::pair<MatrixXd, VectorXd>
orbitToSlice(double *aa, const int N, const int M){
    KS ks(N+2);
    Map<MatrixXd> aa2(aa, N, M);
    return ks.orbitToSlice(aa2);
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // get the pointer and the size of input 
    double *aa = mxGetPr(prhs[0]);
    mwSize N = mxGetM(prhs[0]);
    mwSize M = mxGetN(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(N, M, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(M, 1, mxREAL);

    std::pair<MatrixXd, VectorXd> tmp = orbitToSlice(aa, N, M);
    memcpy(mxGetPr(plhs[0]), &tmp.first(0,0), N*M*sizeof(double));
    memcpy(mxGetPr(plhs[1]), &tmp.second(0), M*sizeof(double));
}
