/* h5c++ test_KSEIDr.cc -std=c++11 -lksint -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3 && ./a.out 
 */
#include <iostream>
#include <ctime>
#include "KSEIDr.hpp"
#include "myH5.hpp"
#include "ksint.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N30

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "IFRK43", "IFRK54"};
    std::string file = "/usr/local/home/xiong/00git/research/data/ks22h001t120x64.h5";
    std::string poType = "ppo";

    MatrixXd a0;
    double T, r, s;
    int nstp;
    std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);

#ifdef N5
    //====================================================================================================
    // test the implemenation of Cox-Matthews is consistent with previous implementation by
    // verifying the accuarycy of ppo1
    KSEIDr ks(64, 22);
    ArrayXXd aa = ks.intgC(a0, T, T/nstp, 1, true);
    KS ks2(64, 22);
    VectorXd a1 = ks2.Reflection(aa.rightCols(1));
    ArrayXXd aa2 = ks2.intg(a0, T/nstp, nstp, 1);
    VectorXd a2 = ks2.Reflection(aa2.rightCols(1));
    double err1 = (a1-a0).norm();
    double err2 = (a2-a0).norm();
    cout << err1 << '\t' << err2 << endl;

    // Result:
    // 4.05043e-14	3.92468e-14

#endif
#ifdef N10
    //====================================================================================================
    // test whether the implementations are correct by using one ppo
    KSEIDr ks(64, 22);
    ArrayXXd aa;
    MatrixXd lte(nstp*2, scheme.size());
    for(int i = 0; i < scheme.size(); i++) {
	ks.setScheme(scheme[i]);
	aa = ks.intgC(a0, 2*T, T/nstp, 1, true);
	savetxt("N10_aa" + to_string(i) + ".dat", aa.topRows(3));
	lte.col(i) = ks.lte;
    }
    savetxt("N10_lte.dat", lte);
#endif
    
#if 0
    KS ks2(64, 22);
    ArrayXXd aa2;
    t = clock();
    for(int i = 0; i < 100; i++) aa2 = ks2.intg(a0, T/nstp, 2*nstp, 1);
    t = clock() - t;
    cee((double)t / CLOCKS_PER_SEC);
#endif
#ifdef N20
    //====================================================================================================
    // test the order of different schemes by varying time step
    // By constant stepping, the estimated LTEs are saved
    KSEIDr ks(64, 22);
    for(int k = 1; k < 1000; k*=10){
	MatrixXd lte(nstp*2/k, scheme.size());
	for(int i = 0; i < scheme.size(); i++) {
	    ks.setScheme(scheme[i]);
	    ArrayXXd aa = ks.intgC(a0, 2*T, T/nstp*k, 1, true);
	    lte.col(i) = ks.lte;
	}
	savetxt("N20_lte" + to_string(k) + ".dat", lte);
    }

#endif
#ifdef N30
    //====================================================================================================
    // by choosing a 
    KSEIDr ks(64, 22);
    for(int i = 0; i < scheme.size(); i++) {
	ks.setScheme(scheme[i]);
	ArrayXd x0;
	for(int k = 1; k < 10000; k*=10){
	    ArrayXXd aa = ks.intgC(a0, T, k*T/nstp/100, 1, true);
	    // cout << aa.cols() << '\t';
	    if (k == 1) x0 = aa.rightCols(1);
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    cout << err << '\t';
	}
	cout << endl;
    }
#endif

    return 0;
}
