/* h5c++ test_CQCGL1dEIDc.cc -std=c++11 -lCQCGL1d -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I $RESH/include/ -L $RESH/lib/ -I $XDAPPS/eigen/include/eigen3 -DEIGEN_FFTW_DEFAULT -O3 -o cgl1d.out && time ./cgl1d.out 
 */
#include <iostream>
#include <ctime>
#include "CQCGL1dEIDc.hpp"
#include "myH5.hpp"
#include "CQCGL1d.hpp"
#include "denseRoutines.hpp"

#define cee(x) cout << (x) << endl << endl;
#define N50

using namespace MyH5;
using namespace std;
using namespace Eigen;
using namespace denseRoutines;

int main(){

    std::vector<std::string> scheme = {"IFRK43", "IFRK54",
				       "Cox_Matthews", "Krogstad", "Hochbruck_Ostermann", 
				       "Luan_Ostermann", "SSPP43"};

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
    CQCGL1d cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);

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
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    
    VectorXcd A0 = Gaussian(N, N/2, N/30, 2.5) + Gaussian(N, 2*N/5, N/30, 0.2);
    VectorXd a0 = cgl2.Config2Fourier(A0);    
    double T = 20;

    cgl.setScheme(scheme[1]);	
    ArrayXXd aa = cgl.intgC(a0, 0.002, T, 50);
    savetxt("aa.dat", aa.transpose());

#endif
#ifdef N20
    //====================================================================================================
    // Test the accuracy of constant stepping shemes.
    // Choose Luan-Ostermann method with a small time step as the base
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    
    VectorXcd A0 = Gaussian(N, N/2, N/30, 2.5) + Gaussian(N, 2*N/5, N/30, 0.2);
    VectorXd a0 = cgl2.Config2Fourier(A0);    
    double T = 20;

    double h0 = T / (1<<18);	// T / 2^18
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 1000000).rightCols(1);

    int n = 8;

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
#ifdef N25
    //====================================================================================================
    // fix the time step. try to look at the estimated local error of a single sheme
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
 
    VectorXcd A0 = Gaussian(N, N/2, N/20, 1) + Gaussian(N, N/3, N/20, 0.4); 
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 40;
    int n0 = 17, n1 = 6;
    double h0 = T / (1<<n0);	// T / 2^17

    cgl.setScheme("Cox_Matthews");	
    // cgl.changeOmega(-17.667504892760448);
    ArrayXXd aa = cgl.intgC(a0, h0, T, 1<<n1);
    savetxt("ex0.dat", cgl.lte);
#endif
#ifdef N30
    //====================================================================================================
    // fix the time step. try to look at the estimated local error of all shemes
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;
    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    
    VectorXcd A0 = Gaussian(N, N/2, N/30, 2.5) + Gaussian(N, 2*N/5, N/30, 0.2);
    VectorXd a0 = cgl2.Config2Fourier(A0);    
    double T = 20;
    int n0 = 15, n1 = 6;
    double h0 = T / (1<<n0);	
    double w[] = {0, -17.667504892760448};

    MatrixXd ltes(1<<(n0-n1), 2*scheme.size());
    for(int k = 0; k < 2; k++){
	cgl.changeOmega(w[k]);
	for(int i = 0; i < scheme.size(); i++) {
	    cgl.setScheme(scheme[i]);	
	    ArrayXXd aa = cgl.intgC(a0, h0, T, 1<<n1);
	    ltes.col(k*scheme.size() + i) = cgl.lte;
	}
    }
    savetxt("cqcgl1d_N30_lte.dat", ltes);
#endif
#ifdef N40
    //====================================================================================================
    // compare static frame integration and comoving frame integration for a single method
    // by using different rtol so make the global error close
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    cgl.setScheme("Cox_Matthews");
    MatrixXd hs;
    
    VectorXcd A0 = Gaussian(N, N/2, N/30, 2.5) + Gaussian(N, 2*N/5, N/30, 0.2);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 20;
    double w[] = {0, -17.667504892760448};

    for(int i = 0; i < 2; i++){
	cgl.changeOmega(w[i]);
	cgl.eidc.rtol = i == 0 ? 1e-10 : 1e-11;
	ArrayXXd aa = cgl.intg(a0, T/(1<<12), T, 100);
	MatrixXd tmp(cgl.hs.size(), 2);
	tmp << cgl.Ts, cgl.hs;
	savetxt("cqcgl1d_N40_" + to_string(i) + ".dat", tmp);
	savetxt("aa_" + to_string(i) + ".dat", aa.transpose());
    }

#endif
#ifdef N50
    //====================================================================================================
    // static frame integration
    // set the rtol = 1e-10 and output some statics
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);
    cgl.eidc.rtol = 1e-10;

    VectorXcd A0 = Gaussian(N, N/2, N/30, 2.5) + Gaussian(N, 2*N/5, N/30, 0.2);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 20;
    double h0 = T / (1<<12);	// T / 2^12
    
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);
	ArrayXXd aa = cgl.intg(a0, h0, T, 1<<5);
	savetxt("cqcgl1d_N50_hs_"+to_string(i)+".dat", cgl.hs);
    }
    
#endif
#ifdef N60
    //====================================================================================================
    // Comoving frame integration
    // set the rtol = 1e-10 and output time steps
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    cgl.changeOmega(-176.67504941219335);
    CQCGL1d cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
    
    cgl.eidc.rtol = 1e-10;
    
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    double h0 = T / (1<<12);	// T / 2^12
    
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);
	ArrayXXd aa = cgl.intg(a0, h0, T, 1<<5);
	savetxt("cqcgl1d_N60_comoving_hs_"+to_string(i)+".dat", cgl.hs);
	// cout << cgl.hs << endl;
    }
#endif
#ifdef N65
    //====================================================================================================
    // comoving frame integration
    //       rtol vs global relative error for a single method
    //       
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    cgl.changeOmega(-17.667504892760448);
    CQCGL1d cgl2(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, -1);

    VectorXcd A0 = Gaussian(N, N/2, N/20, 2.5) + Gaussian(N, N/3, N/20, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);    
    double T = 40;
    double h0 = T / (1 << 20);
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 1000000).rightCols(1);

    int n = 10;
    time_t t;
    
    cgl.setScheme("Luan_Ostermann");
    VectorXd errors(n);
    for(int i = 0, k = 1; i < n; i++, k*=5){
	printf("i == %d\n", i);
	double rtol = k * 1e-15;
	cgl.eidc.rtol = rtol;
	ArrayXd lastState = cgl.intg(a0, T/(1<<12), T, 1000000).rightCols(1);
	double err = (lastState - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	errors[i] = err;
    }
    cout << errors << endl;

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
    // cgl.changeOmega(-17.667504892760448);
    CQCGL1d cgl2(N, d, 4, 0.8, 0.01, di, -1, 4);
 
    VectorXcd A0 = Gaussian(N, N/2, N/10, 3) + Gaussian(N, N/4, N/10, 0.5);
    VectorXd a0 = cgl2.Config2Fourier(A0);
    double T = 4;
    double h0 = T / (1<<20);	// T / 2^20
    
    cgl.setScheme("Luan_Ostermann");
    ArrayXd x0 = cgl.intgC(a0, h0, T, 1000000).rightCols(1);

    int n = 10;
    time_t t; 
    
    MatrixXd erros(n, 4*scheme.size()+1);
    for(int i = 0; i < scheme.size(); i++) {
	cgl.setScheme(scheme[i]);	
	for(int j = 0, k=1; j < n; j++, k*=5){
	    double rtol = k*1e-14;
	    cgl.eidc.rtol = rtol;
	    if(i == 0) {
		erros(j, 0) = rtol; 
	    }
	    t = clock();
	    ArrayXXd aa = cgl.intg(a0, T/(1<<12), T, 1000000);
	    t = clock() - t;
	    double err = (aa.rightCols(1) - x0).abs().maxCoeff() / x0.abs().maxCoeff();
	    erros(j, 4*i+1) = err; 
	    erros(j, 4*i+2) = cgl.eidc.NCalCoe;
	    erros(j, 4*i+3) = cgl.eidc.NCallF;
	    erros(j, 4*i+4) = static_cast<double>(t) / CLOCKS_PER_SEC;
	    cout << err << ' ';
	}
	cout << endl;
    }
    savetxt("cqcgl1d_N70_stat.dat", erros);
#endif
#ifdef N80
    //====================================================================================================
    // test the accuracy of calculating coefficient
    // there is no problem with coefficients
    const int N = 1024; 
    const double d = 30;
    const double di = 0.06;
    CQCGL1dEIDc cgl(N, d, 4, 0.8, 0.01, di);
    cgl.changeOmega(-176.67504941219335);
    
    int n = 3;
    ArrayXXcd c(N, n);
    for (int i = 0; i < n; i++){
	cgl.eidc.M = 64 * (1<<i);
	cout << cgl.eidc.M << endl;
	cgl.eidc.calCoe(1e-3);
	c.col(i) = cgl.eidc.b[3];
	
    }
    
    MatrixXd err(n, n);
    for( int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++){
	    err(i, j) = (c.col(i) - c.col(j)).abs().maxCoeff();
	}
    }
    
    cout << err << endl;

#endif
#ifdef N90
    //====================================================================================================
    // test the norm of b for estimating the local error
    const int N = 1024; 
    const double d = 50;
    double Bi = 0.8, Gi = -0.6;

    CQCGL1dEIDc cgl(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi);
    
    int bindex[] = {3, 3, 4, 7};

    ArrayXXcd bs1(N, 4);
    for (int i = 0; i < 4; i++){
	cgl.setScheme(scheme[i]);
	cgl.eidc.calCoe(1e-3);
	bs1.col(i) = cgl.eidc.b[bindex[i]];
    }
    savetxt("l1r.dat", cgl.L.real()); 
    savetxt("l1c.dat", cgl.L.imag());
    
    cgl.changeOmega(-17.667504892760448);
    
    ArrayXXcd bs2(N, 4);
    for (int i = 0; i < 4; i++){
	cgl.setScheme(scheme[i]);
	cgl.eidc.calCoe(1e-3);
	bs2.col(i) = cgl.eidc.b[bindex[i]];
    }
    savetxt("l2r.dat", cgl.L.real());
    savetxt("l2c.dat", cgl.L.imag());

    for(int i = 0; i < 4; i++) 
	cout << bs1.col(i).matrix().norm() << '\t';
    cout << endl;
    for(int i = 0; i < 4; i++)
	cout << bs2.col(i).matrix().norm() << '\t';
    cout << endl;

    savetxt("bs1r.dat", bs1.real()); 
    savetxt("bs1c.dat", bs1.imag());
    savetxt("bs2r.dat", bs2.real()); 
    savetxt("bs2c.dat", bs2.imag());

#endif
    
    return 0;
}
