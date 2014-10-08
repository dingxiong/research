/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' MEXksint.cpp ksint.cc ksintM1.cc
 * -I../../include -L../../lib -lksintm1
 * -I/usr/include/eigen3 
 * */
#include "ksintM1.hpp"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <Eigen/Dense>
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Map;

const int N = 32; 

static ArrayXXd ksf(double *a0, double d,  double h, int nstp, int np){
  KS ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  //ArrayXd v0(N-2); for(int i = 0 ; i < N-2; i++) v0(i) = a0[i];
  ArrayXXd aa = ks.intg(v0, nstp, np);
  return aa;
}

static KSM1::KSat ksfM1(double *a0, double d,  double h, int nstp, int np){
  KSM1 ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  //ArrayXd v0(N-2); for(int i = 0 ; i < N-2; i++) v0(i) = a0[i];
  KSM1::KSat aa = ks.intg(v0, nstp, np);
  return aa;
}

static KSM1::KSat ksf2M1(double *a0, double d,  double h, double T, int np){
  KSM1 ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  //ArrayXd v0(N-2); for(int i = 0 ; i < N-2; i++) v0(i) = a0[i];
  KSM1::KSat aa = ks.intg2(v0, T, np);
  return aa;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  double *a0 = mxGetPr(prhs[0]);
  double d = mxGetScalar(prhs[1]);
  double h = mxGetScalar(prhs[2]);
  int nstp = mxGetScalar(prhs[3]);
  int np = mxGetScalar(prhs[4]);
  int nqr = mxGetScalar(prhs[5]);
  mwSize isM1 = mxGetScalar(prhs[6]); /* isM1 =1: integration on the 1st
					 mode slice. */
  mwSize isT = mxGetScalar(prhs[7]); /* isT = 1: integration time on full
					state space. */
  double T = mxGetScalar(prhs[8]);

  if(isM1 == 0){
    plhs[0] = mxCreateDoubleMatrix(N-2, nstp/np + 1, mxREAL);
    //Map<ArrayXXd> aa(mxGetPr(plhs[0]), N-2, nstp/np + 1);
    //aa = ksf(a0, d, h, nstp, np);
    ArrayXXd aa = ksf(a0, d, h, nstp, np);
    memcpy(mxGetPr(plhs[0]), &aa(0,0), (nstp/np+1)*(N-2)*sizeof(double));
  }
  else{
    if(isT == 0){
      plhs[0] = mxCreateDoubleMatrix( nstp/np + 1 , 1, mxREAL);
      plhs[1] = mxCreateDoubleMatrix(N-2, nstp/np + 1, mxREAL);
      KSM1::KSat at = ksfM1(a0, d, h, nstp, np);
      memcpy(mxGetPr(plhs[0]), &(at.tt(0)), (nstp/np+1)*sizeof(double));
      memcpy(mxGetPr(plhs[1]), &(at.aa(0,0)), (nstp/np+1)*(N-2)*sizeof(double));
      //Map<ArrayXXd> (at.aa)(mxGetPr(plhs[1]), N-2, nstp/np+1);
      //Map<ArrayXd> (at.tt)(mxGetPr(plhs[0]), nstp/np+1);
    }
    else{
      KSM1::KSat at = ksf2M1(a0, d, h, T, np);
      int m = at.aa.cols();
      plhs[0] = mxCreateDoubleMatrix( m , 1, mxREAL);
      plhs[1] = mxCreateDoubleMatrix(N-2, m, mxREAL);
      memcpy(mxGetPr(plhs[0]), &(at.tt(0)), m*sizeof(double));
      memcpy(mxGetPr(plhs[1]), &(at.aa(0,0)), m*(N-2)*sizeof(double));
    }
  }
}
