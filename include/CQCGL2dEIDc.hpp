#ifndef CQCGL1DEIDC_H
#define CQCGL1DEIDC_H

#include "EIDc.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include "myH5.hpp"

using namespace MyH5;

class CQCGL2dEIDc {
    
public :

    typedef std::complex<double> dcp;
    
    const int N, M;		/* dimension of FFT */
    const double dx, dy;	/* system domain size */
    int Ne, Me;
    int Nplus, Nminus, Nalias;
    int Mplus, Mminus, Malias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;
    ArrayXd Kx, QKx, Ky, QKy;
    ArrayXXcd L;
    ArrayXcd Lv;
	
    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 50;
    
    struct NL {
	CQCGL2dEIDc *cgl;
	dcp B, G;
	MatrixXcd A;
	
	NL(CQCGL2dEIDc *cgl) : cgl(cgl), B(cgl->Br, cgl->Bi), G(cgl->Gr, cgl->Gi) {
	    A.resize(cgl->M, cgl->N);
	}
	~NL(){}	
	
	void operator()(ArrayXcd &x, ArrayXcd &dxdt, double t){
	    Map<MatrixXcd> xv(x.data(), cgl->M, cgl->N);
	    Map<MatrixXcd> dxdtv(dxdt.data(), cgl->M, cgl->N);
	    cgl->fft.inv2(A.data(), x.data(), cgl->N, cgl->M);
	    ArrayXXcd A2 = A.array() * A.array().conjugate();
	    A = B * A.array() * A2 + G * A.array() * A2.square();
	    cgl->fft.fwd2(dxdt.data(), A.data(), cgl->N, cgl->M);
	    dxdtv.middleRows(cgl->Mplus, cgl->Malias).setZero(); /* dealias */
	    dxdtv.middleCols(cgl->Nplus, cgl->Nalias).setZero(); /* dealias */
	}
    };
    
    NL nl;

    ArrayXcd Yv[10], Nv[10];
    EIDc eidc;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    CQCGL2dEIDc(int N, int M, double dx, double dy,
		double Mu, double Dr, double Di,
		double Br, double Bi, double Gr,
		double Gi) :
	N(N), M(M), dx(dx), dy(dy), Mu(Mu), Dr(Dr), Di(Di), Br(Br), Bi(Bi), Gr(Gr), Gi(Gi),
	nl(this)
    {
	Ne = (N/3) * 2 - 1;
	Nplus = (Ne + 1) / 2;
	Nminus = (Ne - 1) / 2;
	Nalias = N - Ne;

	Me = (M/3) * 2 - 1;
	Mplus = (Me + 1) / 2;
	Mminus = (Me - 1) / 2;
	Malias = M - Me;

	// calculate the Linear part
	Kx.resize(N,1);
	Kx << ArrayXd::LinSpaced(N/2, 0, N/2-1), N/2, ArrayXd::LinSpaced(N/2-1, -N/2+1, -1);
	Ky.resize(M,1);
	Ky << ArrayXd::LinSpaced(M/2, 0, M/2-1), 0, ArrayXd::LinSpaced(M/2-1, -M/2+1, -1);        

	QKx = 2*M_PI/dx * Kx;  
	QKy = 2*M_PI/dy * Ky;
    
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					     QKy.square().replicate(1, N)); 
	L.middleRows(Mplus, Malias).setZero();
	L.middleCols(Nplus, Nalias).setZero();
	Lv = Map<VectorXcd>(L.data(), M*N);
	
	int nnl0 = eidc.nnls[eidc.scheme];
	for(int i = 0; i < nnl0; i++){
	    Yv[i].resize(M*N);
	    Nv[i].resize(M*N);
	}
	eidc.init(&Lv, Yv, Nv);
	eidc.CN = 8*M;
    }
    
    CQCGL2dEIDc(int N, int M, double dx, double dy,
		double b, double c, double dr, double di)
	: CQCGL2dEIDc(N, M, dx, dy, -1, 1, b, 1, c, -dr, -di)
    { }

    CQCGL2dEIDc(int N, double d,
		double b, double c, double dr, double di)
	: CQCGL2dEIDc(N, N, d, d, b, c, dr, di)
    { }

    ~CQCGL2dEIDc(){}


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void 
    setScheme(std::string x){
	int nnl0 = eidc.nnls[eidc.scheme];	
	eidc.scheme = eidc.names[x];
	int nnl1 = eidc.nnls[eidc.scheme];
	for (int i = nnl0; i < nnl1; i++) {
	    Yv[i].resize(M*N);
	    Nv[i].resize(M*N);
	}
    }

    inline void 
    changeOmega(double w){
	Omega = w;
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					     QKy.square().replicate(1, N)); 
	L.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
	L.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
	Lv = Map<VectorXcd>(L.data(), M*N);
    }

    inline 
    ArrayXXcd
    intgC(const ArrayXXcd &a0, const double h, const double tend, const int skip_rate,
	  const bool doSaveDisk, const string fileName){
	
	const int Nt = (int)round(tend/h);
	const int m = (Nt+skip_rate-1)/skip_rate;
	lte.resize(m);

	ArrayXXcd aa;
	H5File file;
	if (doSaveDisk) file = H5File(fileName, H5F_ACC_TRUNC); /* openFile fails */
	else aa.resize(Me, Ne*m);
	
	ArrayXXcd tu0 = pad(a0); 
	ArrayXcd u0 = Map<ArrayXcd>(tu0.data(), M*N);

	int ks = 0;
	auto ss = [this, &ks, &aa, &file, &doSaveDisk](ArrayXcd &x, double t, double h, double err){
	    fprintf(stderr, "%d ", ks);
	    Map<ArrayXXcd> xv(x.data(), M, N);
	    ArrayXXcd uxv = unpad(xv);

	    lte(ks) = err;

	    if(doSaveDisk){
		char groupName[10];
		sprintf (groupName, "%.6d", ks);
		std::string s = "/"+std::string(groupName);
		Group group(file.createGroup(s));
		std:: string DS = s + "/";
		
		writeMatrixXd(file, DS + "ar", uxv.real());
		writeMatrixXd(file, DS + "ai", uxv.imag());
	    }
	    else{
		aa.middleCols((ks)*Ne, Ne) = uxv;
	    }
	    	    
	    ks++;
	};
	
        eidc.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return aa;
    }

    inline 
    ArrayXXcd
    intg(const ArrayXXcd &a0, const double h, const double tend, const int skip_rate,
	 const bool doSaveDisk, const string fileName){
	
	const int Nt = (int)round(tend/h);
	const int m = (Nt+skip_rate-1)/skip_rate;
	Ts.resize(m);
	hs.resize(m);
	lte.resize(m);

	ArrayXXcd aa;
	H5File file;
	if (doSaveDisk) file = H5File(fileName, H5F_ACC_TRUNC); /* openFile fails */
	else aa.resize(Me, Ne*m);
	
	ArrayXXcd tu0 = pad(a0); 
	ArrayXcd u0 = Map<ArrayXcd>(tu0.data(), M*N);
	
	int ks = 0;
	auto ss = [this, &ks, &aa, &file, &doSaveDisk](ArrayXcd &x, double t, double h, double err){
	    fprintf(stderr, "%d ", ks);
	    Map<ArrayXXcd> xv(x.data(), M, N);
	    ArrayXXcd uxv = unpad(xv);

	    int m = Ts.size();
	    if(ks >= m){
		Ts.conservativeResize(m+cellSize);
		hs.conservativeResize(m+cellSize);
		lte.conservativeResize(m+cellSize);
	    }
	    hs(ks) = h;
	    lte(ks) = err;
	    Ts(ks) = t;
	    
	    if(doSaveDisk){
		char groupName[10];
		sprintf (groupName, "%.6d", ks);
		std::string s = "/"+std::string(groupName);
		Group group(file.createGroup(s));
		std:: string DS = s + "/";
		
		writeMatrixXd(file, DS + "ar", uxv.real());
		writeMatrixXd(file, DS + "ai", uxv.imag());
	    }
	    else{
		if(ks >= m) aa.conservativeResize(Eigen::NoChange, m+cellSize*Ne);
		aa.middleCols(ks*Ne, Ne) = uxv;
	    }

	    ks++;
	};
	
	eidc.intg(nl, ss, 0, u0, tend, h, skip_rate);
	
	hs.conservativeResize(ks);
	lte.conservativeResize(ks);
	Ts.conservativeResize(ks);
	aa.conservativeResize(Eigen::NoChange, ks);

	return aa;
    }

    ArrayXXcd unpad(const ArrayXXcd &v){
	int m = v.rows();
	int n = v.cols();
	assert(m == M && n % N == 0);
	int s = n / N;
    
	ArrayXXcd vt(Me, Ne*s);
	for (int i = 0; i < s; i++){
	
	    vt.middleCols(i*Ne, Ne) <<
	    
		v.block(0, i*N, Mplus, Nplus), 
		v.block(0, i*N+Nplus+Nalias, Mplus, Nminus),

		v.block(Mplus+Malias, i*N, Mminus, Nplus),
		v.block(Mplus+Malias, i*N+Nplus+Nalias, Mminus, Nminus)
		;
	}
    
	return vt;
    }

    ArrayXXcd pad(const ArrayXXcd &v){
	int m = v.rows();
	int n = v.cols();
	assert( n % Ne == 0 && m == Me);
	int s = n / Ne;
    
	ArrayXXcd vp(M, N*s);
	for (int i = 0; i < s; i++){
	
	    vp.middleCols(i*N, N) << 
	    
		v.block(0, i*Ne, Mplus, Nplus), 
		ArrayXXcd::Zero(Mplus, Nalias), 
		v.block(0, i*Ne+Nplus, Mplus, Nminus),
	    
		ArrayXXcd::Zero(Malias, N),
	    
		v.block(Mplus, i*Ne, Mminus, Nplus), 
		ArrayXXcd::Zero(Mminus, Nalias), 
		v.block(Mplus, i*Ne+Nplus, Mminus, Nminus)
		;
	}
    
	return vp;
    }

};

#endif	/* CQCGL1DEIDC_H */
