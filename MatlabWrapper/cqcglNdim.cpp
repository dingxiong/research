/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3 -march=corei7 -msse4.2' MEXcqcgl1d.cpp cqcgl1d.cc -I../../include -I/usr/include/eigen3 -lm -lfftw3
 * */

#include "cqcgl1d.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>

using namespace Eigen;


static int getNdim(int N){
    double d = 1;
    double h = 1;
    bool enableJacv = false;
    int Njacv = 0;
    double b = 1;
    double c = 1;
    double dr = 1;
    double di = 1;
    int threadNum = 4;

    Cqcgl1d cgl(N, d, h, enableJacv, Njacv, b, c, dr, di, threadNum);
   
    return cgl.Ndim;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    int N = mxGetScalar(prhs[0]);

    double Ndim = (double) getNdim(N); // to simply the process, transform it to double

    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), &Ndim, 1*sizeof(double));
}
