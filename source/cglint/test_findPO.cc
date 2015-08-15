/* to comiple:
 * (Note : libreadks.a is static library, so the following order is important)
 *
 * h5c++ test_findPO.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lcqcglRPO_print -lcqcgl1d -lmyfft_threads -lfftw3_threads -lfftw3 -lm -lpthread -lsparseRoutines -literMethod -lmyH5
 *
 * or
 * h5c++ test_findPO.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lcqcglRPO_omp -lcqcgl1d -lmyfft -lfftw3 -lm -fopenmp -lsparseRoutines -literMethod -lmy5H
 * 
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>
#include <H5Cpp.h>

#include "cqcgl1d.hpp"
#include "cqcglRPO.hpp"
#include "myH5.hpp"

using namespace std; 
using namespace Eigen;
using namespace MyH5;

int main(){
    
    cout.precision(15);
    
    switch (1){
	
    case 1:{
	const int N = 512; 
	const double d = 50;
	const double h = 0.001;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo.h5");
	auto tmp = CqcglReadRPO(file, "1");
	MatrixXd &x = std::get<0>(tmp);
	double T = std::get<1>(tmp);
	int nstp = std::get<2>(tmp);
	double th = std::get<3>(tmp);
	double phi = std::get<4>(tmp);
	double err = std::get<5>(tmp);

	
	int M = x.cols();
	int S = 10;
	M /= S;
	nstp *= S;

	MatrixXd xp(x.rows(), M);
	for(int i = 0; i < M; i++){
	    xp.col(i) = x.col(S*i);
	}
	
	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);	
	
	CqcglRPO cglrpo(nstp, M, N, d, h);
	auto result = cglrpo.findRPOM(xp, T, th, phi, 1e-12, 20, 100, 1e-7, 1e-2, 0.1, 0.5, 6000, 10);
	
	break;
    }
	
    case 2: {
	
    }
	
    default: {
	cout << "please choose a case" << endl;
    }
	
    }
    
    return 0;
    

}
