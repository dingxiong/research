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

#define cee(x) (cout << (x) << endl << endl)

#define CASE_40

int main(){
    
    cout.precision(15);
    GMRES_IN_PRINT_FREQUENCE = 50;
    HOOK_PRINT_FREQUENCE = 1;
    GMRES_OUT_PRINT = false;
    

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
    // find one limit cycle using previous data
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
    std::tie(x, err, flag) = cgl.findRPOM_hook2(x0, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
    if (flag == 0) CQCGL1dRpo::writeRpo2("rpoBiGi.h5", cgl.toStr(Bi, Gi, 1), x, nstp, err);    
#endif
#ifdef CASE_25
    //======================================================================
    // use saved guess directively
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8;
    double Gi = -3.6;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/p.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::readRpo(file, cgl.toStr(Bi, Gi, 1));
    VectorXd x0(cgl.Ndim+3);
    x0 << a0, T0, th0, phi0;
	
    double T, th, phi, err;
    MatrixXd x;
    int nstp = nstp0;
    int flag;

    std::tie(x, err, flag) = cgl.findRPOM_hook2(x0, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
    if (flag == 0) CQCGL1dRpo::writeRpo2("rpoBiGi.h5", cgl.toStr(Bi, Gi, 1), x, nstp, err);   

#endif
#ifdef CASE_30
    //====================================================================== 
    // test the accuracy of a limit cycle
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8;
    double Gi = -3.6;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/rpoBiGi.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::readRpo(file, cgl.toStr(Bi, Gi, 1));
    printf("%g %g %g %g %d\n", T0, th0, phi0, err0, nstp0);

    double e = cgl.MFx2(a0, nstp0).norm();
    cee(e);

#endif
#ifdef CASE_40
    //====================================================================== 
    // find limit cycles by varying Bi and Gi
    const int N = 1024;
    const int L = 50;
    double Bi = 0.8;
    double Gi = -3.6;

    int id = 1;
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
	
    string file = "/usr/local/home/xiong/00git/research/data/cgl/rpoBiGi.h5";
    double stepB = -0.1;
    int NsB = 40;
    cgl.findRpoParaSeq(file, id, stepB, NsB, true);
    // for (int i = 1; i < NsB+1; i++){
    // 	cgl.Bi = Bi+i*stepB;
    // 	cgl.findReqParaSeq(file, id, -0.1, 50, false);
    // }


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

    }

#endif

    return 0;
    

}
