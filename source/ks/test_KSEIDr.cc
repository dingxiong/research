/* h5c++ test_KSEIDr.cc -std=c++11 -lmyH5 -ldenseRoutines -literMethod -lfftw3 -lKSEIDr -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3 && ./a.out 
 */
#include <iostream>
#include "KSEIDr.hpp"
#include "myH5.hpp"
#include "denseRoutines.hpp"

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){
    switch (1) {

    case 1 : {
	std::string file = "../../data/ks22h001t120x64.h5";
	std::string poType = "ppo";

	MatrixXd a0;
	double T, r, s;
	int nstp;
	std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);
	
	KSEIDr ks(64, 22);
	ArrayXXd aa = ks.intgC(a0, 2*T, T/nstp, 1, 2, true);
	savetxt("f.dat", aa);
	break;
    }
	
    }

    return 0;
}
