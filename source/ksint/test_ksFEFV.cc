/* to comiple:
 * (Note : libksdim.a is static library, so the following order is important)
 *
 * h5c++ test_ksFEFV.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lksFEFV -lksint -lfftw3 -lmyH5 -lped
 * 
 */

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include "ksFEFV.hpp"


using namespace std;
using namespace Eigen;

int main(){
    cout.precision(16);

    switch (1) {
	
    case 1: {
	const double L = 22;
	const int nqr = 5; // spacing 	
	const int MaxN = 2000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED   
	const int trunc = 30; // number of vectors	
	const size_t ppId = 33;
	const string inputfileName = "../../data/ks22h001t120x64EV.h5";
	const string outputfileName = "ex.h5";
	const string ppType = "rpo";
	// KScalWriteFEFVInit(inputfileName, outputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	auto tmp = KScalFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	cout << tmp.second.col(0) << endl;
	
	break;
    }
	
    default: {
	cout << "Please indicate the right #" << endl;
    }
	
    }


    return 0;
}
