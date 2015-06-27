/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3 -march=corei7 -msse4.2' MEXcqcgl1d.cpp cqcgl1d.cc
 * -I../../include -I/usr/include/eigen3 -lm -lfftw3
 * */
#include "cqcgl1d.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>
using Eigen::ArrayXXd; using Eigen::ArrayXXcd;
using Eigen::ArrayXd; using Eigen::ArrayXcd;
using Eigen::Map;

static ArrayXXd intg(double *a0, int N, double d,  double h, int nstp, int np,
		     double Mu, double Br, double Bi, double Dr, double Di, 
		     double Gr, double Gi){
  Cqcgl1d cgl(N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi);
  Map<ArrayXd> v0(a0, 2*N);
  ArrayXXd aa = cgl.intg(v0, nstp, np);
  return aa;
}

static Cqcgl1d::CGLaj intgj(double *a0, int N, double d, double h, int nstp,
			    int np, int nqr, double Mu, double Br, double Bi,
			    double Dr, double Di, double Gr, double Gi){
  Cqcgl1d cgl(N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi);
  Map<ArrayXd> v0(a0, 2*N);
  Cqcgl1d::CGLaj aj = cgl.intgj(v0, nstp, np, nqr);
  
  return aj;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  double *a0 = mxGetPr(prhs[0]);
  int N = mxGetScalar(prhs[1]);
  double d = mxGetScalar(prhs[2]);
  double h = mxGetScalar(prhs[3]);
  int nstp = mxGetScalar(prhs[4]);
  int np = mxGetScalar(prhs[5]);
  int nqr = mxGetScalar(prhs[6]);
  double Mu = mxGetScalar(prhs[7]);
  double Br = mxGetScalar(prhs[8]);
  double Bi = mxGetScalar(prhs[9]);
  double Dr = mxGetScalar(prhs[10]);
  double Di = mxGetScalar(prhs[11]);
  double Gr = mxGetScalar(prhs[12]);
  double Gi = mxGetScalar(prhs[13]);
  mwSize isJ = mxGetScalar(prhs[14]);

  
  if( 0 == isJ){
    plhs[0] = mxCreateDoubleMatrix(2*N, nstp/np+1, mxREAL);
    ArrayXXd aa = intg(a0, N, d, h, nstp, np, Mu, Br, Bi, Dr, Di, Gr, Gi);
    memcpy(mxGetPr(plhs[0]), &aa(0,0), (nstp/np+1)*(2*N)*sizeof(double));
  } else {
    plhs[0] = mxCreateDoubleMatrix(2*N, nstp/np+1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(2*N*2*N, nstp/nqr, mxREAL);
    Cqcgl1d::CGLaj aj = intgj(a0, N, d, h, nstp, np, nqr, Mu, Br, Bi, Dr, Di, Gr, Gi);
    memcpy(mxGetPr(plhs[0]), &(aj.aa(0,0)), (nstp/np+1)*(2*N)*sizeof(double));
    memcpy(mxGetPr(plhs[1]), &(aj.daa(0,0)), (nstp/nqr)*(2*N)*(2*N)*sizeof(double));
  }
}
