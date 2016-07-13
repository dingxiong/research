/* h5c++ test_CQCGL1dEIDc.cc -std=c++11 -lCQCGL -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3 && ./a.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL1dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N20

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "IFRK43", "IFRK54"};

#ifdef N10
    //====================================================================================================
    // test the performance of the EID compared with previous implementation.
    // The initial condition is the travelling wave, or a suppoposition of
    // two Gaussian waves. This asymmetric initial condition will result
    // asymmetric explosions.
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);

    std::string file = "/usr/local/home/xiong/00git/research/data/cgl/reqDi.h5";
    std::string groupName = to_string(di) + "/1";
    VectorXd a0;
    double wth, wphi, err;
    std::tie(a0, wth, wphi, err) = CqcglReadReq(file, groupName);

    if (true) {
	VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
	a0 = cgl2.Config2Fourier(A0);
    }
    
    double T = 4;

    time_t t = clock();
    ArrayXXd aa = cgl.intgC(a0, 0.001, T, 10);
    t = clock()-t;
    cout << static_cast<double>(t) / CLOCKS_PER_SEC << endl;

    t = clock();
    ArrayXXd aa2 = cgl2.intg(a0, 0.001, static_cast<int>(T/0.001), 10);
    t = clock()-t;
    cout << static_cast<double>(t) / CLOCKS_PER_SEC << endl;

    cout << aa.cols() << ' ' << aa2.cols() << ' '	 
	 << (aa-aa2.rightCols(aa.cols())).abs().maxCoeff() << ' '
	 << aa.abs().maxCoeff() << ' ' << cgl.lte.maxCoeff() << endl;

    savetxt("aa.dat", aa.transpose()); savetxt("aa2.dat", aa2.transpose()); 
    
    // Output:
    // 0.871511
    // 0.738813
    // 4000 4001 3197.97 3297.8 0.000316152
    
    // the performance are almost the same.

#endif
#ifdef N20
    //====================================================================================================
    // Test the accuracy of constant stepping shemes.
    // Choose Luan-Ostermann method with a small time step as the base, then integrate the
    // system for one period of ppo1.
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
 
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    double h0 = T / (1<<18);	// T / 2^20
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 10000).rightCols(1);

    int n = 10;

    MatrixXd erros(n, scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	for(int j = 0, k=1; j < n; j++, k*=2){
	    double h = k*h0;
	    if(i == 0) {
		erros(j, 0) = h; 
		cout << h << ' ';
	    }
	    ArrayXXd aa = cgl.intgC(a0, h, T, 10000);
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, i+1) = err; 
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("N20_err.dat", erros);

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
    


#endif
    
    return 0;
}
