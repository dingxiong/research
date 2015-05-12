/* compile command: 
 * mex CXXFLAGS='-std=c++0x -fPIC -O3' intg2M1.cpp ../ksint.cc ../ksintM1.cc -I../../../include  -I$XDAPPS/eigen/include/eigen3 -lfftw3
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
intg2M1(double *a0, int N, double h, double T, int np, double d){
  KSM1 ks(N+2, h, d);
  Map<ArrayXd> v0(a0, N);
  return ks.intg2(v0, T, np);
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
	    double T = mxGetScalar(prhs[2]);

	    std::pair<ArrayXXd, ArrayXd> tmp = intg2M1(a0, N, h, T, np, d);
	    int m = tmp.first.cols();
	    plhs[0] = mxCreateDoubleMatrix(N, m, mxREAL);
	    plhs[1] = mxCreateDoubleMatrix(m, 1, mxREAL);
	    
	    memcpy(mxGetPr(plhs[0]), &tmp.first(0,0), m*N*sizeof(double));
	    memcpy(mxGetPr(plhs[1]), &tmp.second(0), m*sizeof(double));

	    break; // very import
	}
	default:
	    mexErrMsgIdAndTxt( "KS_integrator:inputMismatch", 
			       " 3 =< #input <=5. Example: intg(a0, h, nstp, np, d) \n");
	}
   
}
