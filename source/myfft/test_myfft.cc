/* g++ test_myfft.cc -lmyfft -lfftw3 -lfftw3_threads -ldenseRoutines -std=c++11  -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3
 */
#include "myfft.hpp"
#include "denseRoutines.hpp"
#include <iostream>
#include <time.h>

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
	/* for large 2d matrix
	   The c++ version is 2 times slower than
	   Matlab FFT
	*/
	
	clock_t t;
	FFT2d F(8, 4);
	ArrayXXcd x = loadComplex("re.dat", "im.dat");	
	CE(x);
	F.v1 = x;
	
	t = clock();
	for (int i = 0; i < 100; i++) F.ifft();
	t = clock() - t;
	cout << "ifft time: " << ((float)t)/CLOCKS_PER_SEC << endl;
	
	savetxt("f1.dat", F.v2.real());
	savetxt("f2.dat", F.v2.imag());
	CE(F.v2);

	F.fft();
	// CE(F.v3);

	break;
    }

    default : {
	cout << "please indicate the correct number" << cout;
	break;
    }
	
    }

    return 0;

}
