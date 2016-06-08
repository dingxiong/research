/* to comiple:
 * h5c++ -O3 test_CQCGL2d.cc  -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++11 -lCQCGL2d -lsparseRoutines -ldenseRoutines -lmyH5 -lmyfft -lfftw3 -lm 
 */
#include "CQCGL2d.hpp"
#include "myH5.hpp"
#include "denseRoutines.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

#define CE(x) (cout << (x) << endl << endl)

using namespace std;
using namespace Eigen;
using namespace MyH5;
using namespace denseRoutines;

typedef std::complex<double> dcp;

int main(){

    switch(1){
	
    case 1: {			/* test integrator */
	int N = 1024; 
	double L = 30;
	double di = 0.3;
	CQCGL2d cgl(N, L, 4.0, 0.8, 0.05, di, 4);
	
	ArrayXXcd A0 = 5*centerRand2d<dcp>(N, N, 0.2, 0.2);
	ArrayXXcd a0 = cgl.Config2Fourier(A0);
	ArrayXXcd A2 = cgl.Fourier2Config(a0);
	
	ArrayXXcd aa = cgl.intg(a0, 0.001, 10, 1, true, "ex.h5");
	CE(a0.rows()); CE(a0.cols()); CE(aa.rows()); CE(aa.cols());
	
	break;
    }
#if 0
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

		
    case 3 :{ 			/* test Fourier2Config */
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
	ArrayXXd aa = cgl.intg(A0, nstp, nqr);
	ArrayXXd AA = cgl.Fourier2Config(aa);
	
	cout << AA.rows() << 'x' << AA.cols() << endl << "--------------" << endl;
	//cout << AA.col(2) << endl;
	//cout << A0 << endl;
	break;
    }

    case 4:{			/* test the new constructor */
	const int N = 512;
	const int L = 50;
	double h = 0.001;
	Cqcgl1d cgl(N, L, h, false, 0, 0.1, 0.1, 0.1, 0.1, 4);
	const int Ndim = cgl.Ndim;
	
	int nstp = 1000;
	
	ArrayXd A0(2*N) ; 
	// prepare Gaussian curve initial condition
	for(int i = 0; i < N; i++) {
	    double x = (double)i/N*L - L/2.0; 
	    A0(2*i) =  exp(-x*x/8.0);
	} 
	ArrayXd a0 = cgl.Config2Fourier(A0).col(0);

	ArrayXXd AA = cgl.intg(a0, nstp, 1);
	
	cout << AA.rows() << 'x' << AA.cols() << endl << "--------------" << endl;
	cout << cgl.trueNjacv << endl;
	
	break;
    }

    case 5:{			/* test the cgl constructor */
	const int N = 512;
	const int L = 50;
	double h = 0.001;
	Cgl1d cgl(N, L, h, false, 0, 0.1, 0.2, 4);
	const int Ndim = cgl.Ndim;

	cout << cgl.Mu << endl;
	cout << cgl.Br << endl;
	cout << cgl.Bi << endl;
	cout << cgl.Dr << endl;
	cout << cgl.Di << endl;
	cout << cgl.Gr << endl;
	cout << cgl.Gi << endl;
	cout << cgl.trueNjacv << endl;
	
	break;
    }

    case 6: {			
	/* test reflectVe()
	 * Find the reason why discrete symmetry reduction will produce NAN elements.
	 * I find the answer is that my transformation is singular at point (0, 0).
	 */
	const int N = 1024;
	const int L = 30;
	const double h = 0.0002;
	const double di = 0.0799;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/req_different_di/req0799.h5");
	VectorXd a0;
	double wth, wphi, err;
	CqcglReadReq(file, "1", a0, wth, wphi, err);
	    
	Cqcgl1d cgl(N, L, h, true, 0, 4.0, 0.8, 0.01, di, 4);
	
	auto tmp = cgl.evReq(a0, wth, wphi);
	VectorXcd &e = tmp.first;
	MatrixXd v = realv(tmp.second);
	cout << e.head(10) << endl;
	
	ArrayXd a0Hat = cgl.orbit2sliceSimple(a0);
	ArrayXd a0Tilde = cgl.reduceReflection(a0Hat);
	MatrixXd vHat = cgl.ve2slice(v, a0);
	MatrixXd vTilde = cgl.reflectVe(vHat, a0Hat);
	
	MatrixXd G = cgl.refGradMat(a0Hat);
	MatrixXd G1 = cgl.refGrad1();
	ArrayXd step1 = cgl.reduceRef1(a0Hat);
	ArrayXd step2 = cgl.reduceRef2(step1);
	MatrixXd G2 = cgl.refGrad2(step1);
	MatrixXd G3 = cgl.refGrad3(step2);

	cout << G.maxCoeff() << endl;
	cout << G.minCoeff() << endl << endl;
	MatrixXd tmp2(step1.size(), 4);
	tmp2 << a0Hat, step1, step2, a0Tilde;
	cout << tmp2 << endl << endl;
	// cout << G3.row(42) << endl << endl;
	// cout << (G * vHat.col(0)).head(50) << endl;
	// cout << vTilde.col(0).head(60) << endl;
	break;

    }
    case 7:{
	/* see the spacing along a orbit
	 *   I find h = 0.0002 seems too large
	 */
	const int N = 1024;
	const int L = 30;
	const double h = 0.0002;
	const double di = 0.4;
	
	Cqcgl1d cgl(N, L, h, true, 0, 4.0, 0.8, 0.01, di, 4);

	std::string file("rpot.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	printf("T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);

	ArrayXXd aa = cgl.intg(x.col(0), nstp*10, 1);
	VectorXd sp = spacing(aa);
	
	cout << sp.maxCoeff() << endl;
	cout << aa.rightCols(1).matrix().norm() << endl;
	break;
	
    }
#endif
    default: {
	fprintf(stderr, "please indicate a valid case number \n");
	}
    }
    
    return 0;
}
