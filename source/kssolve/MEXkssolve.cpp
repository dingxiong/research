/* compile command: 
 * mex CXXFLAGS='-std=c++11 -fPIC' MEXkssolve.cpp -I../../include -L../../lib -lkssolve
 * */
#include "kssolveM1.hpp"
#include "mex.h"
const int N = 32; 

static void ksf(double *a0, double d,  double h, int nstp, int np, double *aa){
  Ks ks(32, d, h);
  ks.kssolve(a0, nstp, np, aa);
}

static void ksfM1(double *a0, double d,  double h, int nstp, int np, double *aa, 
		  double *tt){
  KsM1 ks(32, d, h);
  ks.kssolve(a0, nstp, np, aa, tt);
}

static void ksfj(double *a0, double d,  double h, int nstp, int np, int nqr, double *aa,
		 double *daa){
  Ks ks(32, d, h);
  ks.kssolve(a0, nstp, np, nqr, aa, daa);
}

static void ksfjM1(double *a0, double d,  double h, int nstp, int np, int nqr, double *aa,
		   double *daa, double *tt){
  KsM1 ks(32, d, h);
  ks.kssolve(a0, nstp, np, nqr, aa, daa, tt);
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  double *a0 = mxGetPr(prhs[0]);
  double d = mxGetScalar(prhs[1]);
  double h = mxGetScalar(prhs[2]);
  mwSize nstp = mxGetScalar(prhs[3]);
  mwSize np = mxGetScalar(prhs[4]);
  mwSize nqr = mxGetScalar(prhs[5]);
  mwSize isJ = mxGetScalar(prhs[6]); // isJ = 1: calculate Jacobian
  mwSize isM1 = mxGetScalar(prhs[7]); /* isM1 =1: integration on the 1st
					 mode slice. */
  if(isM1 == 0){
    plhs[0] = mxCreateDoubleMatrix(N-2, nstp/np, mxREAL);
    double *aa = mxGetPr(plhs[0]);
  
    if(isJ == 1){
      plhs[1] = mxCreateDoubleMatrix((N-2)*(N-2), nstp/nqr, mxREAL);
      double *daa = mxGetPr(plhs[1]);
      ksfj(a0, d, h, nstp, np, nqr, aa, daa);  
    }
    else
      ksf(a0, d, h, nstp, np, aa);
  }
  else{
    plhs[0] = mxCreateDoubleMatrix(nstp/np, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(N-3, nstp/np, mxREAL);
    double *tt = mxGetPr(plhs[0]);
    double *aa = mxGetPr(plhs[1]);

    if(isJ == 1){
      plhs[2] = mxCreateDoubleMatrix((N-3)*(N-3), nstp/nqr, mxREAL);
      double *daa = mxGetPr(plhs[2]);
      ksfjM1(a0, d, h, nstp, np, nqr, aa, daa, tt);
    }
    else
      ksfM1(a0, d, h, nstp, np, aa, tt);
  }
}
