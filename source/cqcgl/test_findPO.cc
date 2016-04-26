/* to comiple:
 * (Note : libreadks.a is static library, so the following order is important)
 *
 * h5c++ test_findPO.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lCQCGLRPO -lCQCGL -lmyfft_threads -lfftw3_threads -lfftw3 -lm -lsparseRoutines -ldenseRoutines -literMethod -lped -lmyH5
 *
 * or
 * h5c++ test_findPO.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lcqcglRPO_omp -lcqcgl1d -lmyfft -lfftw3 -lm -fopenmp -lsparseRoutines -ldenseRoutines -literMethod -lped -lmy5H
 * 
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>
#include <H5Cpp.h>

#include "CQCGL.hpp"
#include "CQCGLRPO.hpp"
#include "myH5.hpp"
#include "iterMethod.hpp"

using namespace std; 
using namespace Eigen;
using namespace MyH5;
using namespace iterMethod;

int main(){
    
    cout.precision(15);
    
    switch (28){
#if 0
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

    case 25: {
	/* try to find periodic orbit with the new form of cqcgl
	 * using GMRES Hook v2 method with multishooting method
	 * space resolution is large
	 */
	GMRES_IN_PRINT_FREQUENCE = 10;

	const int N = 512;
	const double d = 30;
	const double h = 1e-4;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo3.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	int S = 1;
	M /= S;
	nstp *= S;
	
	int Ndim = x.rows();
	MatrixXd xp(Ndim+3, M);
	for(int i = 0; i < M; i++){
	    xp.col(i).head(Ndim) = x.col(S*i);
	    xp(Ndim, i) = T / M;
	    if (i == M-1) {
		xp(Ndim+1, i) = th;
		xp(Ndim+2, i) = phi;
	    }
	    else{
		xp(Ndim+1, i) = 0;
		xp(Ndim+2, i) = 0;
	    }
	}

	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, 0.01, 0.04, 4);
	auto result = cglrpo.findRPOM_hook2(xp, 1e-12, 1e-3, 10, 20, 2e-1, 100, 1);

	CqcglWriteRPO2("rpo2.h5", "1", 
		       std::get<0>(result),
		       nstp,
		       std::get<1>(result)
		       );
	
	break;
    }

    case 26: {
	/* try to find periodic orbit with the new form of cqcgl
	 * using Lervenberg-Marquardt method with multishooting method
	 * space resolution is large
	 */
	CG_PRINT_FREQUENCE = 1;
	
	const int N = 512;
	const double d = 30;
	const double h = 1e-5;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo5.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	
	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, 0.01, 0.04, 4);
	auto result = cglrpo.findRPOM_LM(x, 1e-12, 100, 20);

	CqcglWriteRPO2("rpo5.h5", "2", 
		       std::get<0>(result),
		       nstp,
		       std::get<1>(result)
		       );
	
	break;
    }

#endif

    case 28: {
	/* try to find periodic orbit with the new form of cqcgl
	 * using GMRES Hook v2 method with multishooting method
	 * space resolution is large
	 */
	GMRES_IN_PRINT_FREQUENCE = 1;

	const int N = 1024;
	const double d = 30;
	const double di = 0.06;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "1", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	
	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CQCGLRPO cglrpo(M, N, d, 4.0, 0.8, 0.01, di, 4);
	cglrpo.changeOmega(-176.67504941219335);
	cglrpo.cgl1.rtol = 1e-10;
	cglrpo.cgl2.rtol = 1e-10;
	cglrpo.cgl3.rtol = 1e-10;
	auto result = cglrpo.findRPOM_hook2(x, 1e-12, 1e-3, 10, 20, 3e-1, 300, 1);

	CqcglWriteRPO2("rpo5.h5", "2", 
		       std::get<0>(result),
		       nstp,
		       std::get<1>(result)
		       );
	
	break;
    }

#if 0
    case 30: {
	/* try to find periodic orbit with the new form of cqcgl
	 * using GMRES Hook method with multishooting method
	 * space resolution is large
	 */
	const int N = 1024;
	const double d = 30;
	const double h = 1e-5;

	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo2.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, "2", x, T, nstp, th, phi, err);
	
	int M = x.cols();
	int S = 1;
	M /= S;
	nstp *= S;
 
	MatrixXd xp(x.rows(), M);
	for(int i = 0; i < M; i++){
	    xp.col(i) = x.col(S*i);
	}

	printf("T %g, nstp %d, M %d, th %g, phi %g, err %g\n", T, nstp, M, th, phi, err);
	CqcglRPO cglrpo(nstp, M, N, d, h, 4.0, 0.8, 0.01, 0.04, 4);
	auto result = cglrpo.findRPOM_hook(xp, T, th, phi, 1e-12, 1e-3, 10, 20, 1e-1, 140, 1);
	CqcglWriteRPO("rpo2.h5", "1",
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
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 1e-3, 30, 8, 1e-6, 500, 10);
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
	printf("\n T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, -0.01, di, 4);
	cglrpo.alpha1 = 0.1;
	cglrpo.alpha2 = 0.1;
	cglrpo.alpha3 = 0.0;
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 1e-3, 10, 8, 1e-6, 500, 10);
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
	printf("\n T %g, nstp %d, th %g, phi %g, err %g\n", T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, -0.01, di, 4);
	cglrpo.alpha1 = 0.0;
	cglrpo.alpha2 = 0.0;
	cglrpo.alpha3 = 0.0;
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 1e-12, 1e-3, 10, 8, 1e-5, 500, 10);
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

    case 70: {
	/*  Similar to case 31. But we are using di such that there exist a 
	 *  limit Hopf cycle 
	 */
	const int N = 1024;
	const double d = 30;
	const double h = 0.0002;
	const double di = 0.422;

	// std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpo3.h5");
	std::string file("rpot.h5");
	int nstp;
	double T, th, phi, err;
	MatrixXd x;
	CqcglReadRPO(file, di, 1, x, T, nstp, th, phi, err);
	
	nstp *= 2;
	printf("\n di %g, T %g, nstp %d, th %g, phi %g, err %g\n", di, T, nstp, th, phi, err);
	CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, 0.01, di, 4);
	/* cglrpo.alpha1 = 0; */
	/* cglrpo.alpha2 = 0; */
	/* cglrpo.alpha3 = 0; */
	auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 5e-11, 1e-3, 30, 30, 1e-6, 500, 10);
	CqcglWriteRPO("rpot2.h5", di, 1,
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      nstp,		   /* nstp */
		      std::get<2>(result), /* th */
		      std::get<3>(result), /* phi */
		      std::get<4>(result)  /* err */
		      );
	
	break;
    }

    case 71: {
	/* Move one rpo
	 */
	std::string infile("/usr/local/home/xiong/00git/research/data/cgl/rpot2.h5");
	std::string outfile("/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1.h5");
	// CqcglMoveRPO(infile, outfile, 0.422, 1);

	break;
    }

    case 72:{
	/* move several rpos */
	std::string infile("/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1t.h5");
	std::string outfile("/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1.h5");
	double dis[] = {0.32, 0.36, 0.38, 0.42};
	for(int i = 0; i < 4; i++) CqcglMoveRPO(infile, outfile, dis[i], 1);

	break;
    }

    case 80: {
	/* After we find the Hopf limit cycle for di = 0.39, we need to find
	 * how it evoles with different di
	 */
	const int N = 1024;
	const double d = 30;
	const double di = 0.414;
	const double diInc = -0.001;
	std::string file("/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1.h5");
	
	for(int i = 0; i < 1; i++){
	    double diOld = di + i * diInc;
	    int nstp;
	    double T, th, phi, err;
	    MatrixXd x;
	    CqcglReadRPO(file, diOld, 1, x, T, nstp, th, phi, err);

	    /* nstp += 20; */
	    /* T += 0.003; */
	    double diNew = di + (i+1)*diInc;
	    double h = T / nstp;
	    printf("\n diNew %g, T %g, nstp %d, h %g th %g, phi %g, err %g\n", 
		   diNew, T, nstp, h, th, phi, err);
	    CqcglRPO cglrpo(nstp, 1, N, d, h, 4.0, 0.8, 0.01, diNew, 4);
	    /* cglrpo.alpha1 = 0.1; */
	    /* cglrpo.alpha2 = 0.1; */
	    /* cglrpo.alpha3 = 0.1; */
	    auto result = cglrpo.findRPO_hook(x.col(0), T, th, phi, 5e-11, 1e-3, 30, 30, 1e-6, 500, 10);
	    CqcglWriteRPO(file, diNew, 1, 
			  std::get<0>(result), /* x */
			  std::get<1>(result), /* T */
			  nstp,		   /* nstp */
			  std::get<2>(result), /* th */
			  std::get<3>(result), /* phi */
			  std::get<4>(result)  /* err */
			  );
	}
	
	break;
	
    }

#endif
	
    default: {
	cout << "please choose a case" << endl;
    }
	
    }
    
    return 0;
    

}
