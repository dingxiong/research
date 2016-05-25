/* g++ test_myfft.cc -lmyfft -lfftw3 -lfftw3_threads -ldenseRoutines -std=c++11  -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3
 */
#include "myfft.hpp"
#include "denseRoutines.hpp"
#include <iostream>

#define CE(x) cout << (x) << endl << endl

using namespace std;
using namespace MyFFT;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    switch (2) {

    case 1 :{
	RFFT F(64, 1);
	ArrayXcd x(ArrayXcd::Random(33));

	F.vc1 = x;
	F.ifft();

	break;
    }
       
    case 2 : {			/* test FFT2d */
	FFT2d F(4, 4);
	ArrayXXcd x(4, 4);
	x.real()= loadtxt("re.dat");
	x.imag() = loadtxt("im.dat");
	
	CE(x);
	F.v1 = x;
	F.ifft();

	CE(F.v2);

	F.fft();
	CE(F.v3);

	break;
    }

    default : {
	cout << "please indicate the correct number" << cout;
	break;
    }
	
    }

    return 0;

}
