/* to comiple:
 * (Note : libksdim.a is static library, so the following order is important)
 *
 * h5c++ test_ksdim.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lksdim -lksint -lfftw3 -ldenseRoutines -lmyH5
 * 
 */

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include "ksdim.hpp"

using namespace std;
using namespace Eigen;

int main(){
    cout.precision(16);

    switch (1) {

    case 1: {
	MatrixXi subspDim(4,3); 
	subspDim << 
	  0, 0, 0,
	  0, 7, 27,
	  1, 8, 28,
	  1, 29, 29; // (0-6,7-29), (0-7,8-29), (0-8,9-29)
	cout << subspDim << endl;
	string fileName("../../data/ks22h001t120x64EV.h5");
	string ppType("rpo");	
	int ppId = 33;
	auto ang = anglePO(fileName, ppType, ppId, subspDim);
	
	cout << ang.second << endl;
	
	ofstream file;
	file.precision(16);
	file.open("good.dat", ios::trunc);
	file << ang.first << endl;
	file.close();

	break;
	
    }
	
    case 2: {			/* test the angle tangency */
	string fileName("../../data/ks22h001t120x64EV.h5");
	string ppType("rpo");
	anglePOs(fileName, ppType, 30, 200, "./anglePOs64/rpo/vector/", "vector", 29);
	break;
    }

    case 3: {			/* test partialHyperb */
	string fileName("../../data/ks22h001t120x64EV.h5");
	string ppType("rpo");
	MatrixXd expand  = partialHyperb(fileName, ppType, 3);
	// cout << expand << endl;
	
	break;
    }

    case 4: {	       /* calculate the local expansion rate of FVs */
	string fileName("../../data/ks22h001t120x64EV.h5");
	partialHyperbOneType(fileName, "rpo", 200, "./FVexpand/rpo/");
	// partialHyperbAll(fileName, 200, 200, "./FVexpand/");
	break;
    }


    }
}
