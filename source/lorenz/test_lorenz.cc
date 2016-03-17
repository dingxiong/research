/* to comiple:
 * g++ -O3 test_lorenz.cc  -L../../lib -I../../include -I $XDAPPS/eigen/include/eigen3 -std=c++11 -llorenz -ldenseRoutines -lm && ./a.out
 */
#include "lorenz.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

typedef std::complex<double> dcp;

int main(){

    switch(1){
	
    case 1: {			/* test integrator */
	
	Vector3d x0(VectorXd::Random(3));
	Lorenz loz = Lorenz();
	// auto tmp = loz.intgj(x0, 0.01, 1000, 1, 1);
	// savetxt("xx.dat", tmp.first);
	loz.Rho = 487.115277;
	loz.B = 0.25;
	MatrixXd xx = loz.equilibriaIntg(1, 0, 1e-3, 0.001, 8000, 1);
	savetxt("xx.dat", xx);

	cout << loz.equilibria() << endl;
	break;
    }
		

    default: {
	fprintf(stderr, "please indicate a valid case number \n");
	
	}
    }

    return 0;
}
