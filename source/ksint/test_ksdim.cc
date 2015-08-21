/* to comiple:
 * (Note : libksdim.a is static library, so the following order is important)
 *
 * h5c++ test_ksdim.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lksdim -ldenseRoutines -lmyH5
 * 
 */

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include "ksdim.hpp"

using namespace std;
using namespace Eigen;

int main(){
    cout.precision(16);

    switch (1) {

    case 1: {			/* test the angle tangency */
	string fileName("../../data/ks22h001t120x64EV.h5");
	string ppType("ppo");
	anglePOs(fileName, ppType, 30, 200, "./anglePOs64/ppo/space/", "space", 29);
	break;
    }
	


    }
}
