/* h5c++ test_KSEIDr.cc -std=c++11 -lksint -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3 && ./a.out 
 */
#include <iostream>
#include <ctime>
#include "KSEIDr.hpp"
#include "myH5.hpp"
#include "ksint.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N50

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"Cox_Matthews", 
				       "Krogstad",
				       "Hochbruck_Ostermann", 
				       "Luan_Ostermann", 
				       "IFRK43",
				       "IFRK54", 
				       "SSPP43"};
    std::string file = "/usr/local/home/xiong/00git/research/data/ks22h001t120x64.h5";
    std::string poType = "ppo";

    MatrixXd a0;
    double T, r, s;
    int nstp;
    std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);

#ifdef N3
    //====================================================================================================
    // save the state of 2T of ppo1 by one scheme to check whether the orbit is closed or not
    KSEIDr ks(64, 22);
    ks.setScheme(scheme[6]);
    ArrayXXd aa = ks.intgC(a0, T/nstp, 2*T, 1);
    
    savetxt("aa.dat", aa.topRows(3));


#endif
#ifdef N5
    //====================================================================================================
    // test the implemenation of Cox-Matthews is consistent with previous implementation by
    // verifying the accuarycy of ppo1
    KSEIDr ks(64, 22);
    ArrayXXd aa = ks.intgC(a0, T/nstp, T, 1);
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
    // test whether the implementations are correct by using ppo1.
    // Criteria : after one period and reflection, the orbit should be closed
    KSEIDr ks(64, 22);
    KS ks2(64, 22);
    ArrayXXd aa;
    for(int i = 0; i < scheme.size(); i++) {
	ks.setScheme(scheme[i]);
	aa = ks.intgC(a0, T/nstp, T, 1);
	VectorXd a1 = ks2.Reflection(aa.rightCols(1));
	cout << (a1 - a0).norm() << endl;
    }
    // Output:
    // 4.08788e-14
    // 2.10358e-13
    // 3.26765e-13
    // 3.67868e-13
    // 7.0766e-11
    // 3.80098e-13
    // 3.1632e-11


#endif
#ifdef N20
    //====================================================================================================
    // test the order of different schemes by varying time step
    // By constant stepping, the estimated LTEs are saved
    KSEIDr ks(64, 22);
    for(int k = 1; k < 1000; k*=10){
	MatrixXd lte(nstp/k, scheme.size());
	for(int i = 0; i < scheme.size(); i++) {
	    ks.setScheme(scheme[i]);
	    ArrayXXd aa = ks.intgC(a0, T/nstp*k,  T, 1);
	    lte.col(i) = ks.lte;
	}
	savetxt("KS_N20_lte" + to_string(k) + ".dat", lte);
    }

#endif
#ifdef N30
    //====================================================================================================
    // Test the accuracy of constant stepping shemes.
    // Choose Luan-Ostermann method with a small time step as the base, then integrate the
    // system for one period of ppo1.
    KSEIDr ks(64, 22);
    ks.setScheme("Luan_Ostermann");

    double h0 = T/131072;	// 2^17
    ArrayXd x0 = ks.intgC(a0, h0, T, 10000).rightCols(1);

    int n = 10;

    MatrixXd erros(n, scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	ks.setScheme(scheme[i]);	
	for(int j = 0, k=8; j < n; j++, k*=2){
	    double h = k*h0;
	    if(i == 0) erros(j, 0) = h;
	    ArrayXXd aa = ks.intgC(a0, h, T, 1);
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, i+1) = err; 
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("KS_N30_err.dat", erros);

#endif
#ifdef N40
    //====================================================================================================
    // time adaptive stepping
    // test the implemenation of Cox-Matthews is consistent with previous implementation by
    // verifying the accuarycy of ppo1
    KSEIDr ks(64, 22);
    ArrayXXd aa = ks.intg(a0, T/nstp, T, 1);
    KS ks2(64, 22);
    VectorXd a1 = ks2.Reflection(aa.rightCols(1));
    ArrayXXd aa2 = ks2.intg(a0, T/nstp, nstp, 1);
    VectorXd a2 = ks2.Reflection(aa2.rightCols(1));
    double err1 = (a1-a0).norm();
    double err2 = (a2-a0).norm();
    cout << err1 << '\t' << err2 << endl;
    cout << aa.cols() << '\t' << ks.lte.maxCoeff() << '\t' << ks.hs.maxCoeff() << endl;
    cout << ks.eidr.NCalCoe << ' ' << ks.eidr.NReject << ' ' << ks.eidr.NCallF << ' ' << ks.eidr.NSteps << endl;

    // Output:
    // 1.19741e-08	3.92468e-14
    // 699	6.53879e-09	0.0152952
    // 8 2 3505 699

#endif
#ifdef N50
    //====================================================================================================
    // same as N40, but test all shemes
    KSEIDr ks(64, 22);
    KS ks2(64, 22);
    for(int i = 0; i < scheme.size(); i++) {
	ks.setScheme(scheme[i]);
	ArrayXXd aa = ks.intg(a0, T/nstp, T, 1);
	VectorXd a1 = ks2.Reflection(aa.rightCols(1));
	cout << (a1-a0).norm() << ' ' << ks.lte.maxCoeff() << ' ' << ks.hs.maxCoeff() << ' ';
	cout << ks.eidr.NCalCoe << ' ' << ks.eidr.NReject << ' ' << ks.eidr.NCallF << ' ' << ks.eidr.NSteps << endl;
    }
    
    // Output:
    // 
    // 1.19741e-08 6.53879e-09 0.0152952 8 2 3505 699
    // 1.45044e-08 6.51859e-09 0.0251392 9 2 2600 518
    // 7.36104e-09 6.53008e-09 0.0251002 10 2 2600 518
    // 2.12496e-09 2.44513e-09 0.082695 8 0 1215 135
    // 2.50557e-08 6.54958e-09 0.00559735 8 2 11695 2337
    // 1.90432e-08 5.89591e-09 0.0132776 8 1 6202 885
    // 1.94839e-10 6.55866e-09 0.00202678 7 2 154752 6446
    


#endif
    
    return 0;
}
