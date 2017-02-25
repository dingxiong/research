/* time h5c++ test_CQCGL2dEIDc.cc -std=c++11 -lCQCGL2d -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3  -o cgl2d.out && time ./cgl2d.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL2dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL2d.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N70

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"IFRK43", "IFRK54",
				       "Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "SSPP43"};

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
    cgl.IntPrint = 10;

    MatrixXcd A0 = Gaussian2d(N, N, N/2, N/2, N/30, N/30, 2.5) 
	+ Gaussian2d(N, N, 2*N/5, 2*N/5, N/30, N/30, 0.2);
    ArrayXXcd a0 = cgl2.Config2Fourier(A0);
    double T = 20;

    int n0 = 14, n1 = 5;
    double h0 = T / (1<<n0);	// T / 2^13   
    double w[] = {0, -7.3981920609491505};
    MatrixXd ltes(1<<(n0 - n1), 2*scheme.size());

    for(int k = 0; k < 2; k++){
	cgl.changeOmega(w[k]);
	for(int i = 0; i < scheme.size(); i++) {
	    fprintf(stderr, "k = %d, scheme is %s \n", k, scheme[i].c_str());
	    cgl.setScheme(scheme[i]);	
	    ArrayXXcd aa = cgl.intgC(a0, h0, T, 1<<n1, 2, "aa.h5");
	    ltes.col(k*scheme.size() + i) = cgl.lte;
	}
    }
    savetxt("cqcgl2d_N30_lte.dat", ltes);
#endif
#ifdef N50
    //====================================================================================================
    // static frame integration
    // set the rtol = 1e-9 and output time steps
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL2dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL2d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4);
    cgl.IntPrint = 10;
    cgl.eidc.rtol = 1e-9;

    MatrixXcd A0 = Gaussian2d(N, N, N/2, N/2, N/30, N/30, 2.5) 
	+ Gaussian2d(N, N, 2*N/5, 2*N/5, N/30, N/30, 0.2);
    ArrayXXcd a0 = cgl2.Config2Fourier(A0);
    double T = 20;
    double h0 = T / (1<<12);	// T / 2^12
    
    for(int i = 0; i < scheme.size(); i++) {
	fprintf(stderr, "scheme is %s \n", scheme[i].c_str());
	cgl.setScheme(scheme[i]);
	ArrayXXcd aa = cgl.intg(a0, h0, T, 1<<5, 2, "aa.h5");
	savetxt("cqcgl2d_N50_hs_"+to_string(i)+".dat", cgl.hs);
    }
    
#endif
#ifdef N60
    //====================================================================================================
    // Comoving frame integration
    // set the rtol = 1e-9 and output time steps
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL2dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL2d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4);
    cgl.IntPrint = 10;
    cgl.eidc.rtol = 1e-9;
    cgl.changeOmega(-7.3981920609491505);

    MatrixXcd A0 = Gaussian2d(N, N, N/2, N/2, N/30, N/30, 2.5) 
	+ Gaussian2d(N, N, 2*N/5, 2*N/5, N/30, N/30, 0.2);
    ArrayXXcd a0 = cgl2.Config2Fourier(A0);
    double T = 20;
    double h0 = T / (1<<12);	// T / 2^12
    
    for(int i = 0; i < scheme.size(); i++) {
	fprintf(stderr, "scheme is %s \n", scheme[i].c_str());
	cgl.setScheme(scheme[i]);
	ArrayXXcd aa = cgl.intg(a0, h0, T, 1<<5, 2, "aa.h5");
	savetxt("cqcgl2d_N60_comoving_hs_"+to_string(i)+".dat", cgl.hs);
    }
#endif
#ifdef N70
    //====================================================================================================
    // static & comoving frame integration with time step adaption turned on.
    // choose different rtol to obtain
    //      rtol vs global relative error
    //      rtol vs integration steps, N(x) evaluation times
    //      
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
    double w[] = {0, -7.3981920609491505};

    int n = 10;
    time_t t; 
	
    for(int p = 0; p < 2; p++){
	cgl.changeOmega(w[p]);

	MatrixXd erros(n, 5*scheme.size()+1);
	for(int i = 0; i < scheme.size(); i++) {
	    cgl.setScheme(scheme[i]);	
	    for(int j = 0, k=1; j < n; j++, k*=5){
		fprintf(stderr, "w = %d, scheme is %s, j = %d \n\n", p, scheme[i].c_str(), j);
		double rtol = k*1e-12;
		cgl.eidc.rtol = rtol;
		if(i == 0) {
		    erros(j, 0) = rtol; 
		}
		ArrayXXcd xf = cgl.intg(a0, T/(1<<12), T, 1000000, 2, "aa.h5");
		erros.row(j).segment(5*i+1, 5) << 
		    cgl.eidc.NCalCoe, 
		    cgl.eidc.NCallF,
		    cgl.eidc.NReject,
		    cgl.eidc.TotalTime,
		    cgl.eidc.CoefficientTime;
	    }
	}
	savetxt("cqcgl2d_N70_stat" + to_string(p) + ".dat", erros);
    }
#endif    
    return 0;
}
