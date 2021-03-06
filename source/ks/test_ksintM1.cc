/* to compile:
 * g++ test_ksintM1.cc -std=c++0x -lksint -lfftw3 -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3
 */

#include "ksintM1.hpp"
#include <Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;

int main(){
    /* ------------------------------------------------------- */
    switch (2) {
    case 1: {
	ArrayXd a0 = ArrayXd::Ones(30) * 0.1; a0(1) = 0.0;
	KSM1 ks(32, 0.1, 22);
	std::pair<ArrayXXd, ArrayXd> at = ks.intg(a0, 2000, 2000);
	cout << at.first << endl;
	cout << at.second << endl;

	break;
    }

    case 2: {
	ArrayXd a0 = ArrayXd::Ones(30) * 0.1; a0(1) = 0.0;
	KSM1 ks(32, 0.1, 22);
	std::pair<ArrayXXd, ArrayXd> at = ks.intg2(a0, 2, 1);
	cout << at.first.cols()<< endl;
	cout << at.first.rows() << endl;
	cout << at.second.tail(1) << endl;

	break;
    }
    
    default : {
	cout << "please indicate the correct index." << endl;
    }
	
    }

    return 0;
}
