/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3 -march=corei7 -msse4.2' MEXcqcgl1d.cpp cqcgl1d.cc -I../../include -I/usr/include/eigen3 -lm -lfftw3
 * */

#include "cqcgl1d.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>

using namespace Eigen;


static ArrayXXd intgv(int N, double d, double h,
		      int Njacv,
		      double b, double c,
		      double dr, double di,
		      int threadNum,
		      double *a0, double *v, int nstp){

    bool enableJacv = true;
    Cqcgl1d cgl(N, d, h, enableJacv, Njacv, b, c, dr, di, threadNum);
    Map<ArrayXd> tmpa(a0, cgl.Ndim);
    Map<ArrayXXd> tmpv(v, cgl.Ndim, cgl.trueNjacv);
    ArrayXXd av = cgl.intgv(tmpa, tmpv, nstp);
    return av;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    int N = mxGetScalar(prhs[0]);
    double d = mxGetScalar(prhs[1]);
    double h = mxGetScalar(prhs[2]);
    int Njacv = mxGetScalar(prhs[3]);
    double b = mxGetScalar(prhs[4]);
    double c = mxGetScalar(prhs[5]);
    double dr = mxGetScalar(prhs[6]);
    double di = mxGetScalar(prhs[7]);
    int threadNum = mxGetScalar(prhs[8]);

    double *a0 = mxGetPr(prhs[9]);
    double *v = mxGetPr(prhs[10]);
    int nstp = mxGetScalar(prhs[11]);
    // mwSize isJ = mxGetScalar(prhs[14]);
    
    ArrayXXd av = intgv(N, d, h, Njacv, b, c, dr, di, threadNum, a0, v, nstp);
    plhs[0] = mxCreateDoubleMatrix(av.rows(), av.cols(), mxREAL);
    memcpy(mxGetPr(plhs[0]), av.data(), av.cols()*av.rows()*sizeof(double));
}
