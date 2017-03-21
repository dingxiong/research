/* to comiple:
 *
 * h5c++ test_CQCGL1dRpo.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$EIGEN -I$RESH/include  -L$RESH/lib -lCQCGL1dRpo -lCQCGL1dReq -lCQCGL1d -lmyfft -lfftw3 -lm -lsparseRoutines -ldenseRoutines -literMethod -lmyH5
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
using namespace MyH5;

#define cee(x) (cout << (x) << endl << endl)

#define CASE_20

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
    std::tie(a0, T, nstp, th, phi, err) = CQCGL1dRpo::read(file, "0.360000/1");
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
    double Bi = 1.9;
    double Gi = -4.1;
    
    int id = 1;
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/rpoBiGi2.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::read(file, CQCGL1dRpo::toStr(Bi, Gi, id));
    
    cgl.Bi = 1.5;
    cgl.Gi -= 3.9;
    
    double T, th, phi, err;
    MatrixXd x;
    int nstp = nstp0;
    int flag;
    
    std::tie(x, err, flag) = cgl.findRPOM_hook2(a0, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
    if (flag == 0) 
	CQCGL1dRpo::write2("../../data/cgl/rpoBiGi2.h5", cgl.toStr(cgl.Bi, cgl.Gi, 1), x, nstp, err);   

#endif
#ifdef CASE_50
    //====================================================================== 
    // find limit cycles by varying Bi and Gi but using propagated initial
    // condition
    const int N = 1024;
    const int L = 50;
    double Bi = 4.8;
    double Gi = -4.8;

    int id = 1;
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
	
    string file = "../../data/cgl/rpoBiGi2.h5";
    ArrayXd x0;
    double T0, th0, phi0, err0;
    int nstp0;
    std::tie(x0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::read(file, CQCGL1dRpo::toStr(Bi, Gi, id));

    ArrayXd a0 = x0.head(cgl.Ndim);
    ArrayXXd aa = cgl.intg(a0, T0/nstp0, nstp0, 1);
    ArrayXd x1(cgl.Ndim+3);
    x1 << aa.col(3000), T0, th0, phi0;

    double T, th, phi, err;
    ArrayXd x;
    int flag, nstp;
    nstp = nstp0;
    
    cgl.Gi -= 0.1;
    if (!checkGroup(file, CQCGL1dRpo::toStr(cgl.Bi, cgl.Gi, id), false)){
	fprintf(stderr, "Bi = %g Gi = %g nstp = %d T0 = %g\n", cgl.Bi, cgl.Gi, nstp, T0);
	std::tie(x, err, flag) = cgl.findRPOM_hook2(x1, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
	if(flag == 0) CQCGL1dRpo::write2(file, cgl.toStr(cgl.Bi, cgl.Gi, id), x, nstp, err);
    }
    
#endif
#ifdef CASE_60
    //======================================================================
    // find rpo of next one in Bi-Gi plane but using multishooting
    const int N = 1024;
    const double L = 50;
    double Bi = 5.5;
    double Gi = -5.2;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/rpoBiGi2.h5";

    ArrayXd x0;
    double T0, th0, phi0, err0;
    int nstp0; 

    std::tie(x0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::read(file, CQCGL1dRpo::toStr(Bi, Gi, 1));
    ArrayXd a0 = x0.head(cgl.Ndim);
    ArrayXXd aa = cgl.intg(a0, T0/nstp0, nstp0, nstp0/4);
    MatrixXd x1(cgl.Ndim+3, 4);
    x1.col(0) << aa.col(0), T0/4, 0, 0;
    x1.col(1) << aa.col(1), T0/4, 0, 0;
    x1.col(2) << aa.col(2), T0/4, 0, 0;
    x1.col(3) << aa.col(3), T0/4, th0, phi0;
	
    double T, th, phi, err;
    MatrixXd x;
    int nstp = 1000;
    int flag;

    cgl.Gi += 0.1;
    cout << cgl.Bi << ' ' << cgl.Gi << endl;
    std::tie(x, err, flag) = cgl.findRPOM_hook2(x1, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
    if (flag == 0) CQCGL1dRpo::write2("rpoBiGi2.h5", cgl.toStr(Bi, Gi, 1), x, nstp, err);    

#endif
#ifdef CASE_70
    //======================================================================
    // use saved guess directively
    const int N = 1024;
    const double L = 50;
    double Bi = 0;
    double Gi = -3.5;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/p.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::read(file, cgl.toStr(Bi, Gi, 1));
    VectorXd x0(cgl.Ndim+3);
    x0 << a0, T0, th0, phi0;
    // x0 << a0, 3, th0, -7.3;
    
    double T, th, phi, err;
    MatrixXd x;
    int nstp = nstp0;
    int flag;

    std::tie(x, err, flag) = cgl.findRPOM_hook2(x0, nstp, 8e-10, 1e-3, 50, 30, 1e-6, 300, 1);
    if (flag == 0) CQCGL1dRpo::write2("../../data/cgl/rpoBiGi2.h5", cgl.toStr(Bi, Gi, 1), x, nstp, err);   

#endif
#ifdef CASE_80
    //====================================================================== 
    // test the accuracy of a limit cycle
    const int N = 1024;
    const double L = 50;
    double Bi = 4.1;
    double Gi = -4.4;
    
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/rpoBiGi2.h5";
    ArrayXd a0;
    double T0, th0, phi0, err0;
    int nstp0; 
    std::tie(a0, T0, nstp0, th0, phi0, err0) = CQCGL1dRpo::read(file, cgl.toStr(Bi, Gi, 1));
    printf("%g %g %g %g %d\n", T0, th0, phi0, err0, nstp0);

    double e = cgl.MFx2(a0, nstp0).norm();
    cee(e);

#endif
#ifdef CASE_100
    //====================================================================== 
    // find limit cycles by varying Bi and Gi
    const int N = 1024;
    const int L = 50;
    double Bi = 4.9;
    double Gi = -4.9;

    int id = 1;
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
	
    string file = "../../data/cgl/rpoBiGi2.h5";
    double step = -0.02;
    int Ns = 20;
    cgl.findRpoParaSeq(file, id, step, Ns, true, 1, 0);

    // for (int i = 1; i < NsB+1; i++){
    // 	cgl.Bi = Bi+i*stepB;
    // 	cgl.findRpoParaSeq(file, id, -0.1, 50, false);
    // }


#endif
#ifdef CASE_120
    //====================================================================== 
    // move rpo from one file to another file
    std::string fin = "../../data/cgl/rpoBiGiEV.h5";
    std::string fout = "../../data/cgl/rpoBiGiEV2.h5";
    
    for( int i = 0; i < 39; i++){
	double Bi = 1.9 + 0.1*i;
	for(int j = 0; j < 55; j++){
	    double Gi = -5.6 + 0.1*j;
	    string g = CQCGL1dRpo::toStr(Bi, Gi, 1);
	    if (checkGroup(fin, g, false) && !checkGroup(fout, g, false)){
		fprintf(stderr, "%d %g %g\n", 1, Bi, Gi);
		CQCGL1dRpo::move(fin, g, fout, g, 2);
	    }
	}
    }
    

#endif

    return 0;
    

}
