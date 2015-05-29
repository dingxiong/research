/* to comiple:
 * g++ -O3 test_cqcgl1d.cc  -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++0x -lcqcgl1d -lsparseRoutines -lm -lfftw3 
 */
#include "cqcgl1d.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>

using namespace std;
using namespace Eigen;
typedef std::complex<double> dcp;

int main(){

    switch(2){
	
    case 1: {
	const int N = 256; 
	const int L = 50;
	int nstp = 2;
	int nqr = 1;
	double h = 0.01; 
	
	ArrayXd A0(2*N) ;
	// prepare Gaussian curve initial condition
	for(int i = 0; i < N; i++) {
	    double x = (double)i/N*L - L/2.0; 
	    A0(2*i) =  exp(-x*x/8.0);
	}
	Cqcgl1d cgl(N, L, h);  
	std::pair<ArrayXXd, ArrayXXd> tmp = cgl.intgj(A0, nstp, nqr, nqr);
	ArrayXXd &AA = tmp.first;
	
	cout << AA.rows() << 'x' << AA.cols() << endl << "--------------" << endl;
	//cout << AA.col(2) << endl;
	//cout << A0 << endl;
	break;
    }
	
    case 2:{ 			/* test multishoot */
	const int N = 256; 
	const int L = 50;
	int nstp = 2;
	int nqr = 1;
	double h = 0.01; 
	
	ArrayXd A0(2*N) ;
	// prepare Gaussian curve initial condition
	for(int i = 0; i < N; i++) {
	    double x = (double)i/N*L - L/2.0; 
	    A0(2*i) =  exp(-x*x/8.0);
	}
	ArrayXXd aa(2*N, 3);
	aa << A0, A0, A0;
	Cqcgl1d cgl(N, L, h);
	pair<Cqcgl1d::SpMat, VectorXd> tmp = cgl.multishoot(aa, nstp, 0.1, 0.2, true);
	Cqcgl1d::SpMat &AA = tmp.first;
	
	cout << AA.rows() << 'x' << AA.cols() << endl << "--------------" << endl;
	//cout << AA.col(2) << endl;
	//cout << A0 << endl;
	break;
    }
    default: {
	fprintf(stderr, "please indicate a valid case number \n");
	
	}
    }

    return 0;
}
