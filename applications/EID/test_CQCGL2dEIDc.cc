/* time h5c++ test_CQCGL2dEIDc.cc -std=c++11 -lCQCGL2d -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3  -o cgl2d.out && time ./cgl2d.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL2dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL2d.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N15

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "IFRK43", "IFRK54", "SSPP43"};

#ifdef N15
    //====================================================================================================
    // test a single method to see whether there is a bug
    // Also, save the state for orbits 
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL2dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL2d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4);
    cgl.IntPrint = 1;
    
    MatrixXcd A0 = Gaussian2d(N, N, N/2, N/2, N/30, N/30, 2.5) 
	+ Gaussian2d(N, N, 2*N/5, 2*N/5, N/30, N/30, 0.2);
    ArrayXXcd a0 = cgl2.Config2Fourier(A0);
    double T = 20;

    cgl.setScheme("Cox_Matthews");	
    // ArrayXXcd aa = cgl.intgC(a0, 0.002, T, 50, true, "aa.h5");

    if(true){
	cgl.eidc.rtol = 1e-8;

	cgl.setScheme("Cox_Matthews");
	ArrayXXcd aaAdapt = cgl.intg(a0, T/(1<<10), T, 50, true, "aaCox.h5");

	cgl.setScheme("SSPP43");
	aaAdapt = cgl.intg(a0, T/(1<<10), T, 50, true, "aaSS.h5");
    }
    

#endif
#ifdef N20
    //====================================================================================================
    // Test the accuracy of constant stepping shemes.
    // Choose Luan-Ostermann method with a small time step as the base.
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL2dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL2d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4);
    cgl.IntPrint = 10;
 
    MatrixXcd A0 = Gaussian2d(N, N, N/2, N/2, N/30, N/30, 2.5) 
	+ Gaussian2d(N, N, 2*N/5, 2*N/5, N/30, N/30, 0.2);
    ArrayXXcd a0 = cgl2.Config2Fourier(A0);

    double T = 20;
    double h0 = T / (1<<16);	// T / 2^18
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXXcd x0 = cgl.intgC(a0, h0, T, 1000000, false, "aa.h5").rightCols(cgl.Ne);

    int n = 10;

    MatrixXd erros(n, scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	for(int j = 0, k=1; j < n; j++, k*=2){
	    double h = k*h0;
	    if(i == 0) {
		erros(j, 0) = h; 
	    }
	    ArrayXXcd aa = cgl.intgC(a0, h, T, 1000000, false, "aa.h5").rightCols(cgl.Ne);
	    double err = (aa - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, i+1) = err; 
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("cqcgl2d_N20_err.dat", erros);

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
