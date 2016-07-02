/* g++ -std=c++11 -I$EIGEN test_EigenFFT.cc */
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <iostream>

#define EIGEN_FFTW_DEFAULT

#define CE(x) cout << (x) << endl << endl

using namespace Eigen;
using namespace std;

int main(){
    
    VectorXcd A(4);
    A.real() << 2, 8, 3, 6;
    A.imag() << 3, 5, 7, 19;

    VectorXcd B(4);
    VectorXcd C(4);

    FFT<double> fft;
    fft.fwd(B, A);
    fft.inv(C, B);

    CE(A); CE(B); CE(C);
    
    return 0;
}

