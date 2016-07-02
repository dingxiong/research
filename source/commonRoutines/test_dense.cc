/* g++ test_dense.cc -O3 -std=c++11 -I$EIGEN -I${RESH}/include -L${RESH}/lib -ldenseRoutines */

#include <iostream>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

#define CE(x) (cout << x << endl << endl)
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){
    
    switch (1)
	{
	case 1 : {		/* test savetxt */
	    MatrixXcd A(2, 3);
	    A.real() << 1, 2, 3, 4, 5, 6;
	    A.imag() = A.real()/2;
	    CE(A);

	    savetxt("f1.dat", A.real());
	    savetxt("f2.dat", A.imag());
	    break;
	}
	    
	default :{
	    fprintf(stderr, "indicate a case \n");
	}
	    
    }

    return 0;

}
