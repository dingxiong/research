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
  
  plhs[0] = mxCreateDoubleMatrix(N-2, nstp/np, mxREAL);
  plhs[1] = mxCreateDoubleMatrix((N-2)*(N-2), nstp/nqr, mxREAL);
  double *aa = mxGetPr(plhs[0]);
  double *daa = mxGetPr(plhs[1]);

  ksfj(a0, d, h, nstp, np, nqr, aa, daa);  
}
