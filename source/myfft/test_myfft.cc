/* g++ test_myfft.cc -lmyfft -lfftw3 -std=c++11  -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3
 */
#include "myfft.hpp"

using namespace std;
using namespace MyFFT;
using namespace Eigen;

int main(){

    switch (1) {

    case 1 :{
	RFFT F(64, 1);
	ArrayXcd x(ArrayXcd::Random(33));

	F.vc1 = x;
	F.ifft();

	break;
    }
       
    default : {

	
    }
	
    }

    return 0;

}
