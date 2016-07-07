/* h5c++ test_KSEIDr.cc -std=c++11 -lksint -lmyH5 -ldenseRoutines -literMethod -lfftw3 -lKSEIDr -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3 && ./a.out 
 */
#include <iostream>
#include <ctime>
#include "KSEIDr.hpp"
#include "myH5.hpp"
#include "ksint.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;

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
	clock_t t = clock();
	ArrayXXd aa;
	for(int i = 0; i < 10; i++) aa = ks.intgC(a0, 2*T, T/nstp, 1, 2, true);
	t = clock() - t;
	cee((double)t / CLOCKS_PER_SEC);

	KS ks2(64, 22);
	t = clock();
	ArrayXXd aa2;
	for(int i = 0; i < 10; i++) aa2 = ks2.intg(a0, T/nstp, 2*nstp, 1);
	t = clock() - t;
	cee((double)t / CLOCKS_PER_SEC);

	// savetxt("f.dat", aa);
	break;
    }
	
    }

    return 0;
}
