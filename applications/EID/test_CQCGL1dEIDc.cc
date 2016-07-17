/* h5c++ test_CQCGL1dEIDc.cc -std=c++11 -lCQCGL -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3 -o cgl1d.out && time ./cgl1d.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL1dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N50

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "IFRK43", "IFRK54", "SSPP43"};

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
#ifdef N15
    //====================================================================================================
    // test a single method to see whether there is a bug
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
    
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    
    double T = 4;

    cgl.setScheme(scheme[6]);	
    ArrayXXd aa = cgl.intgC(a0, 0.001, T, 10);


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
    double h0 = T / (1<<20);	// T / 2^20
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 1000000).rightCols(1);

    int n = 10;

    MatrixXd erros(n, scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	for(int j = 0, k=2; j < n; j++, k*=2){
	    double h = k*h0;
	    if(i == 0) {
		erros(j, 0) = h; 
	    }
	    ArrayXXd aa = cgl.intgC(a0, h, T, 1000000);
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, i+1) = err; 
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("cqcgl1d_N20_err.dat", erros);

#endif
#ifdef N30
    //====================================================================================================
    // fix the time step. try to look at the estimated local error of all shemes
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
 
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    int n0 = 17, n1 = 6;
    double h0 = T / (1<<n0);	// T / 2^17

    MatrixXd ltes(1<<(n0-n1), scheme.size());
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	ArrayXXd aa = cgl.intgC(a0, h0, T, 1<<n1);
	ltes.col(i) = cgl.lte;
    }
    savetxt("cqcgl1d_N30_lte.dat", ltes);


#endif
#ifdef N50
    //====================================================================================================
    // static frame integration
    // set the rtol = 1e-10 and output some statics
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);

    cgl.eidc.rtol = 1e-10;

    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    double h0 = T / (1<<12);	// T / 2^12
    
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);
	ArrayXXd aa = cgl.intg(a0, h0, T, 1<<5);
	savetxt("cqcgl1d_N50_hs_"+to_string(i)+".dat", cgl.hs);
	// cout << cgl.hs << endl;
    }
    
#endif
#ifdef N70
    //====================================================================================================
    // comoving frame integration.
    // choose different rtol to obtain
    //      rtol vs global relative error
    //      rtol vs integration steps, N(x) evaluation times
    //      
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    CQCGL cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
 
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    double h0 = T / (1<<20);	// T / 2^20
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 1000000).rightCols(1);

    int n = 10;

    
    MatrixXd erros(n, scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	for(int j = 0, k=2; j < n; j++, k*=2){
	    double h = k*h0;
	    if(i == 0) {
		erros(j, 0) = h; 
	    }
	    ArrayXXd aa = cgl.intgC(a0, h, T, 1000000);
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, i+1) = err; 
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("cqcgl1d_N60_err.dat", erros);


#endif
    
    return 0;
}
