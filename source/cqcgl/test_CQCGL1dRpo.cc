/* to comiple:
 *
 * h5c++ test_CQCGL1dRpo.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$EIGEN -I$RESH/include  -L$RESH/lib -lCQCGL1dRpo -lCQCGL1dReq -lCQCGL1d -lmyfft -lfftw3 -lm -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 && ./a.out
 *
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>

#include "CQCGL1dReq.hpp"
#include "CQCGL1dRpo.hpp"

using namespace std; 
using namespace Eigen;
using namespace iterMethod;

#define CASE_20

int main(){
    
    cout.precision(15);
    
#ifdef CASE_10
    //======================================================================
    // to visulize the limit cycle first 
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8;
    double Gi = -3.6;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1.h5";
    ArrayXd a0;
    double T, th, phi, err;
    int nstp; 
    std::tie(a0, T, nstp, th, phi, err) = CQCGL1dRpo::readRpo(file, "0.360000/1");
    a0 *= 0.316;

    ArrayXXd aa = cgl.intg(a0, 1e-3, 20000, 10);
    savetxt("aa.dat", aa);
    savetxt("lte.dat", cgl.lte);
    
#endif
#ifdef CASE_20
    //======================================================================
    // find one limit cycle
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8;
    double Gi = -3.6;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/rpoT2X1.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::readRpo(file, "0.360000/1");
    a0 *= 0.316;
    T0 *= 10;
    ArrayXXd aa = cgl.intg(a0, 1e-3, 20000, 10);
    VectorXd x0(cgl.Ndim+3);
    x0 << aa.rightCols(1), T0, th0, phi0;
	
    double T, th, phi, err;
    MatrixXd x;
    int nstp = static_cast<int>( T0/1e-3/10)*10; // time is enlarged.
    int flag;

    cout << T0 << ' ' << nstp << endl;
    std::tie(x, err, flag) = cgl.findRPOM_hook2(x0, nstp, 8e-10, 1e-3, 30, 30, 1e-6, 500, 10);
    if (flag == 0) CQCGL1dRpo::writeRpo2("but.h5", cgl.toStr(Bi, Gi), x0, nstp, err);    
    
#endif


    






    // =============================================================
#if 0	
    switch (28){

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
	cglrpo.cgl1.rtol = 1e-9;
	cglrpo.cgl2.rtol = 1e-9;
	cglrpo.cgl3.rtol = 1e-9;
	auto result = cglrpo.findRPOM_hook2(x, 1e-12, 1e-3, 10, 20, 3e-1, 300, 1);

	CqcglWriteRPO2("rpo5.h5", "2", 
		       std::get<0>(result),
		       nstp,
		       std::get<1>(result)
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


	
    default: {
	cout << "please choose a case" << endl;
    }
	
    }

#endif

    return 0;
    

}
