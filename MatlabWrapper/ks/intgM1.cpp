/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' intgM1.cpp ../ksint.cc ../ksintM1.cc -I../../../include  -I$XDAPPS/eigen/include/eigen3 -lfftw3
 * */
#include "ksintM1.hpp"
#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;

/* ks 1st mode slice integrator without Jacobian */
static std::pair<ArrayXXd, ArrayXd>
intgM1(double *a0, int N, double h, int nstp, int np, double d){
  KSM1 ks(N+2, h, d);
  Map<ArrayXd> v0(a0, N);
  return ks.intg(v0, nstp, np);
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    int np = 1;
    double d = 22;
    
    switch (nrhs) 
	{
	case 5 :
	    d =  mxGetScalar(prhs[4]); 
	case 4 :
	    np = mxGetScalar(prhs[3]); 
	case 3 : {
	    // get the pointer and the size of input 
	    double *a0 = mxGetPr(prhs[0]);
	    mwSize N = mxGetM(prhs[0]);
	    mwSize M = mxGetN(prhs[0]);
	    assert( N % 2 == 0 && M = 1 );	
	    double h = mxGetScalar(prhs[1]);
	    int nstp = mxGetScalar(prhs[2]);

	    plhs[0] = mxCreateDoubleMatrix(N, nstp/np + 1, mxREAL);
	    plhs[1] = mxCreateDoubleMatrix(nstp/np + 1, 1, mxREAL);
	    std::pair<ArrayXXd, ArrayXd> tmp = intgM1(a0, N, h, nstp, np, d);
	    memcpy(mxGetPr(plhs[0]), &tmp.first(0,0), (nstp/np+1)*N*sizeof(double));
	    memcpy(mxGetPr(plhs[1]), &tmp.second(0), (nstp/np+1)*sizeof(double));

	    break; // very import
	}
	default:
	    mexErrMsgIdAndTxt( "KS_integrator:inputMismatch", 
			       " 3 =< #input <=5. Example: intg(a0, h, nstp, np, d) \n");
	}
   
}
