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
    
    switch (50){
	
    case 1:{
	/* try to find periodic orbit with the old form of cqcgl
	 * space resolution N = 512 is small  
	 */
	const int N = 512; 
	const double d = 50;
	const double h = 0.001;

	// read initial condition
	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	int S = 1;
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
	/* try to find periodic orbit with the new form of cqcgl
	 * space resolution is large
	 */
	const int N = 512*8;
	const double d = 30;
	const double h = 0.0001;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	int S = 2;
	M /= S;
	nstp *= S;
 
	MatrixXd xp(x.rows(), M);
	for(int i = 0; i < M; i++){
	    xp.col(i) = x.col(S*i);
	}

	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, -0.01, -0.04, 4);
	auto result = cglrpo.findRPOM(xp, T, th, phi, 1e-12, 20, 100, 1e-2, 1e-2, 0.1, 0.5, 6000, 10);
	
	break;
    }


    case 30: {
	/* try to find periodic orbit with the new form of cqcgl
	 * using GMRES Hook method with multishooting method
	 * space resolution is large
	 */
	const int N = 512*2;
	const double d = 30;
	const double h = 0.0002;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	int S = 1;
	M /= S;
	nstp *= S;
 
	MatrixXd xp(x.rows(), M);
	for(int i = 0; i < M; i++){
	    xp.col(i) = x.col(S*i);
	}

	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, -0.01, -0.04, 4);
	auto result = cglrpo.findRPOM_hook(xp, T, th, phi, 1e-12, 100, 8, 1e-2, 500, 10);
	CqcglWriteRPO("rpo3.h5", "2",
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );
	
	break;
    }

    case 31: {
	/* same as case = 3, but here use single shooting method */
	const int N = 512*2;
	const double d = 30;
	const double h = 0.0002;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	nstp *= M;
	printf("T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, -0.01, -0.04, 4);
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 30, 8, 1e-6, 500, 10);
	CqcglWriteRPO("rpoX1.h5", "1",
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );
	
	break;
	
    }

    case 32:{
	/* use the inexact new to refine rpo 
	 * which is obtain from GMRES HOOK method
	 *
	 * ==> this trial almost fails.
	 */
	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpoT2x2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);

	int M = x.cols();
	const int N = 1024;
	const double d = 30;
	const double h = T / (M * nstp);

	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, -0.01, -0.04, 4);
	auto result = cglrpo.findRPOM(x, T, th, phi, 1e-12, 20, 100, 1e-2, 1e-2, 0.1, 0.5, 1000, 10);
	CqcglWriteRPO("rpo2x2.h5", "2",
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );
	
	break;
    }
	
    case 40: {			/* test the strength factor a1, a2, a3 */

	CqcglRPO cglrpo(2000, 10, 512, 30, 0.0001, 4.0, 0.8, -0.01, -0.04, 4);

	printf("%g, %g, %g\n", cglrpo.alpha1, cglrpo.alpha2, cglrpo.alpha3);
	cglrpo.alpha1 = 0.01;
	cglrpo.alpha2 = 0.02;
	cglrpo.alpha3 = 0.03;
	printf("%g, %g, %g\n", cglrpo.alpha1, cglrpo.alpha2, cglrpo.alpha3);
	
	break;
    }

    case 50: {
	/*  Similar to case 31. But we try to use different di such that
	 *  the base relative equilbirum has only pair of expanding
	 *  direction.
	 */
	const int N = 1024;
	const double d = 30;
	const double h = 0.0002;
	const double di = -0.0799;

	// std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo3.h5");
	std::string file("rpoT2X1.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	nstp *= M;
	printf("T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, -0.01, di, 4);
	cglrpo.alpha1 = 0.1;
	cglrpo.alpha2 = 0.1;
	cglrpo.alpha3 = 0.0;
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 10, 8, 1e-6, 500, 10);
	CqcglWriteRPO("rpoT2X1.h5", "1",
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );

	break;
	
	
    }
	
    case 60: {
	/*  same as 50 but use smaller time step
	 */
	const int N = 1024;
	const double d = 30;
	const double h = 0.0001;
	const double di = -0.0799;
	
	// std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo3.h5");
	std::string file("rpoT2X1.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "3", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	nstp *= 2 * M;
	printf("T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, -0.01, di, 4);
	cglrpo.alpha1 = 0.0;
	cglrpo.alpha2 = 0.0;
	cglrpo.alpha3 = 0.0;
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 10, 8, 1e-5, 500, 10);
	CqcglWriteRPO("rpoT2X1.h5", "3",
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );

	break;
	
	
    }

    default: {
	cout << "please choose a case" << endl;
    }
	
    }
    
    return 0;
    

}
