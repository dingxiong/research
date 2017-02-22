/* time h5c++ test_CQCGL2dEIDc.cc -std=c++11 -lCQCGL2d -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3  -o cgl2d.out && time ./cgl2d.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL2dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL2d.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N30

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
    ArrayXXcd aa = cgl.intgC(a0, 0.002, T, 50, 0, "aa.h5");

    if(true){
	cgl.eidc.rtol = 1e-8;

	cgl.setScheme("Cox_Matthews");
	ArrayXXcd aaAdapt = cgl.intg(a0, T/(1<<10), T, 50, 0, "aaCox.h5");

	cgl.setScheme("SSPP43");
	aaAdapt = cgl.intg(a0, T/(1<<10), T, 50, 0, "aaSS.h5");
    }
    
#endif
#ifdef N30
    //====================================================================================================
    // fix the time step. try to look at the estimated local error of all shemes
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

    int n0 = 14, n1 = 5;
    double h0 = T / (1<<n0);	// T / 2^17
   
    double w[] = {0, -7.3981920609491505};
    MatrixXd ltes(1<<(n0 - n1), 2*scheme.size());

    for(int k = 0; k < 2; k++){
	cgl.changeOmega(w[k]);
	for(int i = 0; i < scheme.size(); i++) {
	    fprintf(stderr, "k = %d, scheme is %s \n", k, scheme[i].c_str());
	    cgl.setScheme(scheme[i]);	
	    ArrayXXcd aa = cgl.intgC(a0, h0, T, 1<<n1, 2, "aa.h5");
	    ltes.col(i) = cgl.lte;
	}
    }
    savetxt("cqcgl2d_N30_lte.dat", ltes);
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
