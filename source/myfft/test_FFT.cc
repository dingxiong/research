#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <ctime>
#define cee(x) (cout << (x) << endl << endl )

using namespace std;
using namespace Eigen;


int main(){
    
    switch (2){
	
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

    case 2: {
	FFT<double> fft;
	fft.SetFlag(fft.HalfSpectrum);
	int N0 = 8;
	int N1 = 4;
	MatrixXcd A(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	MatrixXcd B(MatrixXd::Random(N0,N1).cast<std::complex<double>>());
	clock_t t = clock();
	fft.fwd2(B, A);
	t = clock()-t;
	cee( (double)t / CLOCKS_PER_SEC);

	cee(A); cee(B); 
	//cee(A.real());

	break;
    }
    }
    return 0;
}
