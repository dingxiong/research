/* to comiple:
 * h5c++ -O3 test_CQCGL2dDislin.cc  -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++11 -lCQCGL2d -lsparseRoutines -ldenseRoutines -lmyH5 -lmyfft -lfftw3 -lm -lCQCGL2dDislin -ldiscpp
 */
#include "CQCGL2dDislin.hpp"
#include "myH5.hpp"
#include "denseRoutines.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

#define cee(x) (cout << (x) << endl << endl)

using namespace std;
using namespace Eigen;
using namespace MyH5;
using namespace denseRoutines;

typedef std::complex<double> dcp;

int main(){

    switch(3){
	
    case 1 : {			/* test integrator */
	int N = 1024; 
	double L = 100;
	double di = 0.05;
	CQCGL2dDislin cgl(N, L, 4.0, 0.8, 0.01, di, 4);
	
	VectorXd xs(2); xs << N/4, 3*N/4;
	VectorXd as(2); as << 0.1, 0.1;
	ArrayXXcd A0 = dcp(3, 3)*solitons(N, N, xs, xs, as, as);
	// ArrayXXcd A0 = dcp(3, 3)*soliton(N, N, N/2, N/2, 0.1, 0.1);
	//ArrayXXcd A0 = dcp(3, 0) * center2d(N, N, 0.2, 0.2);
	ArrayXXcd a0 = cgl.Config2Fourier(A0);
	
	cgl.constAnim(a0, 0.001, 10);
	
	break;
    }

    case 2 : { 
	int N = 1024; 
	double L = 100;
	double di = 0.05;
	CQCGL2dDislin cgl(N, L, 4.0, 0.8, 0.01, di, 4);
	
	VectorXd xs(4); xs << N/4, N/4, 3*N/4, 3*N/4;
	VectorXd ys(4); ys << N/4, 3*N/4, N/4, 3*N/4;
	VectorXd as(4); as << 0.1, 0.1, 0.1, 0.1;
	ArrayXXcd A0 = dcp(3, 3)*solitons(N, N, xs, ys, as, as);
	ArrayXXcd a0 = cgl.Config2Fourier(A0);
	
	cgl.constAnim(a0, 0.001, 10);
	break;
    }

	
    case 3 : { 
	int N = 1024; 
	double L = 100;
	double di = 0.05;
	CQCGL2dDislin cgl(N, L, 4.0, 0.8, 0.01, di, 4);
	
	ArrayXXcd A0 = dcp(3, 3)*solitonMesh(N, N, 3, 3, 0.05);
	ArrayXXcd a0 = cgl.Config2Fourier(A0);
	
	cgl.constAnim(a0, 0.001, 10);
	break;
    }

    default: {
	fprintf(stderr, "please indicate a valid case number \n");
	}
    }
    
    return 0;
}
