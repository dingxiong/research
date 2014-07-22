#include "kssolve.hpp"
#include "mex.h"

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
    ksfM1(a0, d, h, nstp, np, aa, tt);
  }
}
