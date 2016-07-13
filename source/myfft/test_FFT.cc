/*
 * g++ test_FFT.cc -std=c++11 -I$EIGEN -DEIGEN_FFTW_DEFAULT -lfftw3 -O3 && ./a.out
 */
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <ctime>
#define cee(x) (cout << (x) << endl << endl )

using namespace std;
using namespace Eigen;


int main(){
    
    switch (10){
	
    case 1 : {
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	FFT<double> fft2;
	fft2.SetFlag(fft2.HalfSpectrum);

	int N = 16;
	VectorXd A(VectorXd::LinSpaced(N, 0, 1));
	VectorXcd B(VectorXd::LinSpaced(N/2+1, 0, 1).cast<std::complex<double>>());
	cee(A.data()); cee(B.data());
    
	clock_t t = clock();
	for(int i = 0; i < 1000; i++){
	    fft.fwd(B, A);
	    fft.inv(A, B);
	}
	t = clock()-t;
	cee( (double)t / CLOCKS_PER_SEC);

	t = clock();
	for(int i = 0; i < 1000; i++){
	    fft.fwd(B, A);
	    fft2.inv(A, B);
	}
	t = clock()-t;
	cee( (double)t / CLOCKS_PER_SEC);

	cee(A); cee(B); 
	cee(A.data()); cee(B.data());
	break;
    }

    case 10: {			/* does not work */
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	int N = 16;
	VectorXcd A(N);
	VectorXcd B(N);
	A.setRandom();
	
	fft.fwd(B, A);
	cee(A); cee(B);
	
	break;

    }

	////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef N20
    case 20: {			/* does not work */
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	int N = 16;
	MatrixXd A(16, 2);
	MatrixXcd B(16, 2);
	
	A.col(1) = VectorXd::LinSpaced(N, 0, 1);
	fft.fwd(B.col(0), A.col(1));
	cee(A); cee(B);
	
	break;

    }
#endif

    case 2: {			/* simple test 2d */
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	int N0 = 8;
	int N1 = 4;
	MatrixXcd A(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	MatrixXcd B(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	clock_t t = clock();
	fft.inv2(B, A);
	t = clock()-t;
	cee( (double)t / CLOCKS_PER_SEC);

	cee(A); cee(B); 
	//cee(A.real());

	break;
    }

    case 3: {			/* 2d performance */
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	int N0 = 1024;
	int N1 = 1024;
	MatrixXcd A(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	MatrixXcd B(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	cee(A.data()); cee(B.data());

	clock_t t = clock();
	for (int i = 0; i < 200; i++){
	    fft.fwd2(B, A);
	}
	t = clock()-t;
	cee( (double)t / CLOCKS_PER_SEC);

	cee(A.data()); cee(B.data());
	//cee(A); cee(B); 
	//cee(A.real());

	break;
    }
    }
    return 0;
}
