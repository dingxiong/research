/* h5c++ test_KSEIDr.cc -std=c++11 -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3 && ./a.out 
 */
#include <iostream>
#include "KSEIDr.hpp"

int main(){
    switch (1) {

    case 1 : {
	std::string file = "../../data/ks22h001t120x64.h5";
	std::string poType = "rpo";

	MatrixXd a0;
	double T, r, s;
	int nstp;
	std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);
	
	KSEIDr ks(64, 22);
	ks.etd(a0, T, 0.01, 1, 2, true);
	
	break;
    }
	
    }

    return 0;
}
