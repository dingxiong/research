/* to comiple:
 * g++ -O3 test_cqcgl1d.cc -lcqcgl1d -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++0x -lfftw3
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

    switch(1){
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
    case 2:{
	const int N = 256; 
	const int L = 50;
	ArrayXd x = ArrayXd::LinSpaced(N, 1, N) / N * L - L/2.0;
	ArrayXcd a0(N); a0.real() = (-x*x/8.0).exp(); a0.imag() = ArrayXd::Zero(N);
	//cout << a0 << endl;
	Cqcgl1d cgl; 
	ArrayXd v0 = cgl.C2R(a0);
	//ArrayXXd aa = cgl.intg(v0, 10, 1);
	//cout << aa.rows() << 'x' << aa.cols() << endl;
	//cout << cgl.R2C(aa.col(1)) << endl;
	std::pair<ArrayXXd, ArrayXXd> tmp= cgl.intgj(v0, 100, 1, 20);
	cout << tmp.second.rows() << 'x' << tmp.second.cols() << endl;

	break;
    }
    default: {
	fprintf(stderr, "please indicate a valid case number \n");
	
	}
    }

    return 0;
}
