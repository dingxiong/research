#ifndef EID_H
#define EID_H

#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
// #include "denseRoutines.hpp"

using namespace std;
using namespace Eigen;

//////////////////////////////////////////////////////////////////////////////////
//              Exponetial Integrator with Diagonal Linear Part
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Exponential Integrator
 *
 * DT          : data type (double/complex)
 */
template<typename DT>
class EID {
    
public:

    typedef std::complex<double> dcp;
    typedef Array<DT, Dynamic, 1> Ary;

    //////////////////////////////////////////////////////////////////////

    Ary *L;
    ArrayXXcd *N, *Y;		// single or multiple states

    struct Scheme {
	// string name;		// scheme name
	int nstage; /* number of stages, which is used to indicate which Y[?]  store the update */
	int nYN;    // number of Y[], N[] needed in the internal stage
	int nnl;    // number of nonlinear evaluations per step
	int nab;    // number of calculation of a, b whenever change step
	int order;  // orders of scheme
	
	Scheme(int nstage, int nYN, int nnl, int nab, int order) :
	    nstage(nstage), nYN(nYN), nnl(nnl), nab(nab), order(order)
	{}
    };

    const std::unordered_map<std::string, Scheme> names = {
	//   name                    nstage   nYN   nnl     nab     order
        {"Cox_Matthews",        Scheme(4,      5,    5,      4,      4)},
        {"Krogstad",            Scheme(4,      5,    5,      8,      4)},
        {"Hochbruck_Ostermann", Scheme(5,      5,    5,      11,     4)},
	{"Luan_Ostermann",      Scheme(8,      9,    9,      23,     5)},
	{"IFRK43",              Scheme(4,      5,    5,      0,      4)},
	{"IFRK54",              Scheme(6,      7,    7,      0,      5)},
	{"SSPP43",              Scheme(3,      4,    24,     0,      4)}               
    };

    std::string scheme = "Cox_Matthews"; /* default scheme  */
    
    Ary a[9][9], b[9], c[9];
    
    int M = 64;			/* number of sample points */
    int R = 1;			/* radius for evaluating phi(z) */
    int CN = 0;			/* counter number. When L has large dimension, we should split
				   L into pieces and then do counter average. The purpose is
				   to save memory. Name, L -> L[0:CN], L[CN+1, 2*CN], ...
				   If CN <= 0, then not split.
				*/

    double a1, a2, a3;		// for split step method 

    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-8;
    double nu = 0.9;	       /* safe factor */
    double mumax = 4;	       /* maximal time step increase factor */
    double mumin = 0.4;	       /* minimal time step decrease factor */
    double mue = 1.25;	       /* upper lazy threshold */
    double muc = 0.85;	       /* lower lazy threshold */
    //double mue = 1.35;
    //double  muc = 0.8;

    int NCalCoe = 0;	      /* times to evaluate coefficient */
    int NReject = 0;	      /* times that new state is rejected */
    int NCallF = 0;	      /* times to call velocity function f */
    int NSteps = 0;	      /* total number of integrations steps */
    double TotalTime = 0;
    double CoefficientTime = 0;
        
    double err = 0;	 /* LTE : local truncation error estimation */

    ////////////////////////////////////////////////////////////
    // constructor and desctructor
    
    EID(){}
    EID(Ary *L, ArrayXXcd *Y, ArrayXXcd *N) : L(L), Y(Y), N(N){}
    ~EID(){}
    
    void init(Ary *L, ArrayXXcd *Y, ArrayXXcd *N){
	this->L = L;
	this->Y = Y;
	this->N = N;
    }
    ////////////////////////////////////////////////////////////

    // NL : the nonlinear function
    // SS : save state function
    template<class NL, class SS>
    void 
    intg(NL nl, SS saveState, const double t0, const ArrayXXcd &u0, const double tend, const double h0, 
	 const int skip_rate){
	auto t_start = std::chrono::high_resolution_clock::now();
	
	int ns = names.at(scheme).nstage;
	int od = names.at(scheme).order;
	int nnl = names.at(scheme).nnl;
	int nab = names.at(scheme).nab;

	NCalCoe = 0;
	NReject = 0;
	NCallF = 0;    
	NSteps = 0;
	TotalTime = 0;
	CoefficientTime = 0;
	
	double h = h0;
	CoefficientTime += calCoe(h);
	NCalCoe += nab;
	
	double t = t0;
	Y[0] = u0;
	bool doChange, doAccept;

	bool TimeEnds = false;
	while(!TimeEnds){ 

	    if ( t + h > tend){
		h = tend - t;
		CoefficientTime += calCoe(h);
		NCalCoe += nab;
		TimeEnds = true;
	    }

	    oneStep(t, h, nl);
	    NCallF += nnl;		
	    double s = nu * std::pow(rtol/err, 1.0/od);
	    double mu = adaptTs(doChange, doAccept, s);
	
	    if (doAccept){
		t += h;
		NSteps++;
		Y[0] = Y[ns];
		if ( NSteps % skip_rate == 0 || TimeEnds) saveState(Y[0], t, h, err);
	    }
	    else {
		NReject++;
		TimeEnds = false;
	    }
	
	    if (doChange) {
		h *= mu;
		CoefficientTime += calCoe(h);
		NCalCoe += nab;
	    }
	}
	
	auto t_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> t_elapse = t_end - t_start;
	TotalTime = t_elapse.count();
    }
    
    /// @brief constant time-stepping integrator.
    /// @note the initial condition will not be recorded.
    template<class NL, class SS>
    void 
    intgC(NL nl, SS saveState, const double t0, const ArrayXXcd &u0, const double tend, const double h, 
	  const int skip_rate){
	int ns = names.at(scheme).nstage;
	int nnl = names.at(scheme).nnl;
	int nab = names.at(scheme).nab;

	NCallF = 0;
	NSteps = 0; 
	NCalCoe = 0;

	calCoe(h);
	NCalCoe += nab;

	const int Nt = static_cast<int>( round((tend-t0)/h) );
	
	double t = t0;
	Y[0] = u0; 
	for(int i = 0; i < Nt; i++){
	    oneStep(t, h, nl); 
	    NCallF += nnl;
	    NSteps++;
	    t += h;
	    Y[0] = Y[ns]; 
	    if((i+1)%skip_rate == 0 || i == Nt-1) saveState(Y[0], t, h, err);
	}	
    }

    /**
     * @brief calculat the damping factor of time step
     *
     * @param[out] doChange    true if time step needs change
     * @param[out] doAccept    true if accept current time step
     * @param[in]  s           estimate damping factor
     * @return     mu          final dampling factor 
     */
    inline 
    double
    adaptTs(bool &doChange, bool &doAccept, const double s){
	double mu = 1;
	doChange = true;
	doAccept = true;

	if ( s > mumax)
	    mu = mumax;
	else if (s > mue)
	    mu = s;
	else if (s >= 1) {
	    mu = 1;
	    doChange = false;
	}
	else {
	    doAccept = false; 
	    if (s > muc) 
		mu = muc;
	    else if (s > mumin) 
		mu = s;
	    else 
		mu = mumin; // cout << mu << '\t' << rtol << '\t' << err << '\t' << NSteps << endl;
	}
	return mu;
    }
    
    /**
     * nonlinear class NL provides container for intermediate steps
     *
     * @note the multiplication operator blow is overloaded.
     */
    template<class NL>
    void 
    oneStep(double t, double h, NL nl){

	if(scheme == "Cox_Matthews") { 
	    nl(Y[0], N[0], t);
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);     
	    
	    nl(Y[1], N[1], t+h/2);	
	    Y[2] = MUL(c[1], Y[0]) + MUL(a[1][0], N[1]);
	    
	    nl(Y[2], N[2], t+h/2);	
	    Y[3] = MUL(c[1], Y[1]) + MUL(a[1][0], 2*N[2]-N[0]);
	
	    nl(Y[3], N[3], t+h);	
	    Y[4] = MUL(c[3], Y[0]) + MUL(b[0], N[0]) + MUL(b[1], N[1]+N[2]) + MUL(b[3], N[3]);
	    
	    nl(Y[4], N[4], t+h);
	
	    err = MUL(b[3], N[4]-N[3]).abs().maxCoeff() / Y[4].abs().maxCoeff(); 
	}
	else if (scheme == "Krogstad") {
	    nl(Y[0], N[0], t);	
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = MUL(c[1], Y[0]) + MUL(a[2][0], N[0]) + MUL(a[2][1], N[1]);

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = MUL(c[3], Y[0]) + MUL(a[3][0], N[0]) + MUL(a[3][2], N[2]);
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = MUL(c[3], Y[0]) + MUL(b[0], N[0]) + MUL(b[1], N[1]+N[2]) + MUL(b[3], N[3]);
	
	    nl(Y[4], N[4], t+h);
	
	    err = MUL(b[3], N[4]-N[3]).abs().maxCoeff() / Y[4].abs().maxCoeff();
	}
	else if (scheme == "Hochbruck_Ostermann") {
	    nl(Y[0], N[0], t);	
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = MUL(c[1], Y[0]) + MUL(a[2][0], N[0]) + MUL(a[2][1], N[1]);

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = MUL(c[3], Y[0]) + MUL(a[3][0], N[0]) + MUL(a[3][1], N[1]+N[2]);
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = MUL(c[1], Y[0]) + MUL(a[4][0], N[0]) + MUL(a[4][1], N[1]+N[2]) + MUL(a[4][3], N[3]);
	
	    nl(Y[4], N[4], t+h/2);
	    Y[5] = MUL(c[3], Y[0]) + MUL(b[0], N[0]) + MUL(b[3], N[3]) + MUL(b[4], N[4]);

	    err = MUL(b[4], N[4]-0.5*N[2]-0.5*N[1]).abs().maxCoeff() / Y[5].abs().maxCoeff();
	}
	else if (scheme == "Luan_Ostermann"){
	    nl(Y[0], N[0], t);	
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = MUL(c[1], Y[0]) + MUL(a[2][0], N[0]) + MUL(a[2][1], N[1]);

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = MUL(c[3], Y[0]) + MUL(a[3][0], N[0]) + MUL(a[3][2], N[2]);
	
	    nl(Y[3], N[3], t+h/4);
	    Y[4] = MUL(c[1], Y[0]) + MUL(a[4][0], N[0]) + MUL(a[4][2], N[2]) + MUL(a[4][3], N[3]);
	
	    nl(Y[4], N[4], t+h/2);
	    Y[5] = MUL(c[5], Y[0]) + MUL(a[5][0], N[0]) + MUL(a[5][3], N[3]) + MUL(a[5][4], N[4]);

	    nl(Y[5], N[5], t+h/5);
	    Y[6] = MUL(c[6], Y[0]) + MUL(a[6][0], N[0]) + MUL(a[6][3], N[3]) + MUL(a[6][4], N[4]) + MUL(a[6][5], N[5]);

	    nl(Y[6], N[6], t+2*h/3);
	    Y[7] = MUL(c[7], Y[0]) + MUL(a[7][0], N[0]) + MUL(a[7][4], N[4]) + MUL(a[7][5], N[5]) + MUL(a[7][6], N[6]);
	    
	    nl(Y[7], N[7], t+h);
	    Y[8] = MUL(c[7], Y[0]) + MUL(b[0], N[0]) + MUL(b[5], N[5]) + MUL(b[6], N[6]) + MUL(b[7], N[7]);

	    nl(Y[8], N[8], t+h);
	    
	    err = MUL(b[7], N[8]-N[7]).abs().maxCoeff() / Y[8].abs().maxCoeff();
	}
	else if (scheme == "IFRK43") {
	    nl(Y[0], N[0], t);	
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = MUL(c[1], Y[0]) + (h*0.5)*N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = MUL(c[3], Y[0]) + MUL(a[3][2], N[2]);
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = MUL(c[3], Y[0]) + MUL(b[0], N[0]) + MUL(b[1], N[1]+N[2]) + (h/6)*N[3];

	    nl(Y[4], N[4], t+h);

	    err = h*(1.0/10)*(N[4] - N[3]).abs().maxCoeff() / Y[4].abs().maxCoeff();
	}
	else if (scheme == "IFRK54"){
	    nl(Y[0], N[0], t);	
	    Y[1] = MUL(c[1], Y[0]) + MUL(a[1][0], N[0]);

	    nl(Y[1], N[1], t+h/5);
	    Y[2] = MUL(c[2], Y[0]) + MUL(a[2][0], N[0]) + MUL(a[2][1], N[1]);

	    nl(Y[2], N[2], t+3*h/10);
	    Y[3] = MUL(c[3], Y[0]) + MUL(a[3][0], N[0]) + MUL(a[3][1], N[1]) + MUL(a[3][2], N[2]);
	
	    nl(Y[3], N[3], t+4*h/5);
	    Y[4] = MUL(c[4], Y[0]) + MUL(a[4][0], N[0]) + MUL(a[4][1], N[1]) + MUL(a[4][2], N[2]) + MUL(a[4][3], N[3]);

	    nl(Y[4], N[4], t+8*h/9);
	    Y[5] = MUL(c[5], Y[0]) + MUL(a[5][0], N[0]) + MUL(a[5][1], N[1]) + MUL(a[5][2], N[2]) + MUL(a[5][3], N[3]) + MUL(a[5][4], N[4]);

	    nl(Y[5], N[5], t+h);
	    Y[6] = MUL(c[5], Y[0]) + MUL(b[0], N[0]) + MUL(b[2], N[2]) + MUL(b[3], N[3]) + MUL(b[4], N[4]) + (h*11./84)*N[5];

	    nl(Y[6], N[6], t+h);	    
	    err = h*(-71./57600*N[0] + 71./16695*N[2] - 71./1920*N[3] + 17253./339200*N[4]
		     -22./525*N[5] + 1./40*N[6]).abs().maxCoeff() / Y[4].abs().maxCoeff();
	}
	else if (scheme == "SSPP43") {
	    ArrayXcd y0 = Y[0];
	    
	    Y[0] = MUL(c[0], Y[0]);
	    Y[0] = rk4(t+a1*h, a3*h, nl);

	    Y[0] = MUL(c[1], Y[0]);
	    Y[0] = rk4(t+(a1+a2+a3)*h, a2*h, nl);

	    Y[0] = MUL(c[2], Y[0]);
	    ArrayXcd t1 = rk4(t+(a1+2*a3+2*a2)*h, a1*h, nl);

	    Y[0] = y0;	    	    

	    Y[0] = rk4(t, a1*h, nl);
	    Y[0] = MUL(c[2], Y[0]);

	    Y[0] = rk4(t+(a1+a3)*h, a2*h, nl);
	    Y[0] = MUL(c[1], Y[0]);

	    Y[0] = rk4(t+(a1+a3+2*a2)*h, a3*h, nl);
	    ArrayXcd t2 = MUL(c[0], Y[0]);
	    
	    Y[3] = 0.5*(t1 + t2);
	    err = 0.5*(t1 - t2).abs().maxCoeff() / Y[3].abs().maxCoeff();
	    
	    Y[0] = y0;		// reset previous state
	}
	else {
	    fprintf(stderr, "Please indicate the scheme !\n");
	    exit(1);
	}
    }

    template<class NL>
    ArrayXcd rk4(double t, double h, NL nl){
	nl(Y[0], N[0], t);
	Y[1] = Y[0] + h/2 * N[0];
	
	nl(Y[1], N[1], t+h/2);
	Y[2] = Y[0] + h/2 * N[1];
	
	nl(Y[2], N[2], t+h/2);
	Y[3] = Y[0] + h * N[2];
	
	nl(Y[3], N[3], t+h);
	
	return Y[0] + h/6* (N[0] + 2*(N[1]+N[2]) + N[3]);
    }

    /// @brief calculate the coefficients of Butcher table.
    /// The size of coefficient is determined by L.
    /// 
    /// @return time used in seconds
    double
    calCoe(double h){
	auto t_start = std::chrono::high_resolution_clock::now();
	
	int sL = (*L).size();	// size of L
	int p = 1;
	int n = sL;
	
	if (CN > 0){
	    p = sL / CN;
	    n = CN;
	} 
	
	Ary hL = h * (*L);

	if(scheme == "Cox_Matthews") {
	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0].resize(sL);
	    b[0].resize(sL);
	    b[1].resize(sL);
	    b[3].resize(sL);

	    for (int i = 0; i < p; i++){
		int s = i != p-1 ? n : sL-(p-1)*n;
		Ary hLs = h * (*L).segment(i*n, s);
		ArrayXXcd z = ZR(hLs);
		
		ArrayXXcd z2 = z.square();
		ArrayXXcd z3 = z.cube();
		ArrayXXcd ze = z.exp();
		ArrayXXcd zeh = (z/2).exp();
		
		a[1][0].segment(i*n, s) = h * mean( (zeh - 1)/z );
		
		b[0].segment(i*n, s) = h * mean( (-4.0 - z + ze*(4.0 - 3.0 * z + z2)) / z3 );
		b[1].segment(i*n, s) = h*2*mean( (2.0 + z + ze*(-2.0 + z)) / z3 );
		b[3].segment(i*n, s) = h * mean( (-4.0 - 3.0*z -z2 + ze*(4.0 - z) ) / z3 );

	    }
	}
	else if (scheme == "Krogstad") {
	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();
	    
	    a[1][0].resize(sL);
	    a[2][0].resize(sL);
	    a[2][1].resize(sL);
	    a[3][0].resize(sL);
	    a[3][2].resize(sL);
	    b[0].resize(sL);
	    b[1].resize(sL);
	    b[3].resize(sL);
	    
	    for(int i = 0; i < p; i++){
		int s = i != p-1 ? n : sL-(p-1)*n;
		Ary hLs = h * (*L).segment(i*n, s);
		ArrayXXcd z = ZR(hLs);
		
		ArrayXXcd z2 = z.square();
		ArrayXXcd z3 = z.cube();
		ArrayXXcd ze = z.exp();
		ArrayXXcd zeh = (z/2).exp();

		ArrayXXcd t1 = z + 2;
		ArrayXXcd t2 = z - 2;
		ArrayXXcd t3 = z - 4;
		ArrayXXcd t4 = z + 4;
		
	    

		a[1][0].segment(i*n, s) = h * mean( (zeh - 1)/z );
		a[2][0].segment(i*n, s) = h * mean( (zeh*t3 + t4) / z2 );
		a[2][1].segment(i*n, s) = h*2*mean( (2*zeh - t1) / z2 );
		a[3][0].segment(i*n, s) = h * mean( (ze*t2 + t1) / z2 );
		a[3][2].segment(i*n, s) = h*2*mean( (ze - z - 1)  / z2 );
	    
		b[0].segment(i*n, s) = h * mean( (-t4 + ze*(4.0 - 3.0 * z + z2)) / z3 );
		b[1].segment(i*n, s) = h*2*mean( (t1 + ze*t2) / z3 );
		b[3].segment(i*n, s) = h * mean( (-4.0 - 3.0*z -z2 - ze*t3 ) / z3 );
	    }   
	}
	else if (scheme == "Hochbruck_Ostermann") {	    
	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0].resize(sL);
	    a[2][0].resize(sL);
	    a[2][1].resize(sL);
	    a[3][0].resize(sL);
	    a[3][1].resize(sL);
	    a[4][0].resize(sL);
	    a[4][1].resize(sL);
	    a[4][3].resize(sL);
	    b[0].resize(sL);
	    b[3].resize(sL);
	    b[4].resize(sL);
	    
	    for (int i = 0; i < p; i++){
		int s = i != p-1 ? n : sL-(p-1)*n;
		Ary hLs = h * (*L).segment(i*n, s);
		ArrayXXcd z = ZR(hLs);

		ArrayXXcd z2 = z.square();
		ArrayXXcd z3 = z.cube();
		ArrayXXcd ze = z.exp();
		ArrayXXcd zeh = (z/2).exp();



		ArrayXXcd t1 = -4 + z;	   
		ArrayXXcd t2 = 4*z3;
		ArrayXXcd t3 = 20 + ze*t1;
		ArrayXXcd t4 = 4 + z;
		ArrayXXcd t5 = -2 + z;
		ArrayXXcd t6 = 2 + z;

		a[1][0].segment(i*n, s) = h * mean( (zeh - 1)/z );
		a[2][0].segment(i*n, s) = h * mean( (zeh*t1 + t4) / z2 );
		a[2][1].segment(i*n, s) = h*2*mean( (2*zeh - t6) / z2 );
		a[3][0].segment(i*n, s) = h * mean( (ze*t5 + t6) / z2 );
		a[3][1].segment(i*n, s) = h * mean( (ze - z - 1)  / z2 );
		a[4][0].segment(i*n, s) =-h * mean( (t3 - z + z2 - 4*zeh*(4-3*z+z2))  / t2 );
		a[4][1].segment(i*n, s) = h * mean( (t3 + 3*z - z2 + 8*zeh*t5)  / t2 );
		a[4][3].segment(i*n, s) =-h * mean( (t3 + 7*z + z2 + 4*zeh*t1)  / t2 );

		b[0].segment(i*n, s) = h * mean( (-t4 + ze*(4 - 3*z + z2)) / z3 );
		b[3].segment(i*n, s) =-h * mean( (4 + 3*z + z2 + ze*t1) / z3 );
		b[4].segment(i*n, s) = h*4*mean( (t6 + ze*t5 ) / z3 );

	    }
	}
	    
	else if (scheme == "Luan_Ostermann"){
	    c[1] = (hL/2).exp();
	    c[3] = (hL/4).exp();
	    c[5] = (hL/5).exp();
	    c[6] = (2*hL/3).exp();
	    c[7] = hL.exp();

	    a[1][0].resize(sL);
	    a[2][0].resize(sL);
	    a[2][1].resize(sL);
	    a[3][0].resize(sL);
	    a[3][2].resize(sL);
	    a[4][0].resize(sL);
	    a[4][2].resize(sL);
	    a[4][3].resize(sL);
	    a[5][0].resize(sL);
	    a[5][3].resize(sL);
	    a[5][4].resize(sL);
	    a[6][0].resize(sL);
	    a[6][3].resize(sL);
	    a[6][4].resize(sL);
	    a[6][5].resize(sL);
	    a[7][0].resize(sL);
	    a[7][4].resize(sL);
	    a[7][5].resize(sL);
	    a[7][6].resize(sL);
	    b[0].resize(sL);
	    b[5].resize(sL);
	    b[6].resize(sL);
	    b[7].resize(sL);
	    
	    for (int i = 0; i < p; i++){
		int s = i != p-1 ? n : sL-(p-1)*n;
		Ary hLs = h * (*L).segment(i*n, s);
		ArrayXXcd z = ZR(hLs);

		ArrayXXcd z2 = z.square();
		ArrayXXcd z3 = z.cube();
		ArrayXXcd z4 = z2.square();
		ArrayXXcd ze = z.exp();
		ArrayXXcd zeh = (z/2).exp();
		ArrayXXcd ze4 = (z/4).exp();
		ArrayXXcd ze5 = (z/5).exp();
		ArrayXXcd ze3 = (2*z/3).exp();



		ArrayXXcd t1 = -4 + z;	   
		ArrayXXcd t2 = 2*z2;
		ArrayXXcd t3 = 25*z3;
		ArrayXXcd t4 = 4 + z;
		ArrayXXcd t5 = 60-14*z+z2;
		ArrayXXcd t6 = -375*ze5*t5 - 486*ze3*t5;
		ArrayXXcd t7 = 16-6*z+z2;
		ArrayXXcd t8 = 100-5*z-3*z2;
	    
		a[1][0].segment(i*n, s) = h * mean( (zeh - 1)/z );
		a[2][0].segment(i*n, s) = h * mean( (zeh*(-2+z) + 2) / z2 );
		a[2][1].segment(i*n, s) = h * mean( (2*zeh - z - 2) / z2 );
		a[3][0].segment(i*n, s) = h * mean( (2*ze4*(-2+z)- t1) / t2 );
		a[3][2].segment(i*n, s) = h * mean( (4*ze4 - t4)  / t2 );	    
		a[4][0].segment(i*n, s) = h * mean( (zeh*t7 - 2*(8+z))  / z3 );
		a[4][2].segment(i*n, s) =-h * mean( (2*zeh*(-8+z) + 16 + 6*z + z2)  / z3 );
		a[4][3].segment(i*n, s) = h*8*mean( (zeh*t1 + t4)  / z3 );
		a[5][0].segment(i*n, s) = h * mean( (-400 + 70*z - 3*z2 + 25*ze5*t7) / t3 );
		a[5][3].segment(i*n, s) = h*8*mean( (t8 + 25*ze5*t1) / t3 );
		a[5][4].segment(i*n, s) = h*2*mean( (-200 - 25*ze5*(-8+z) - 15*z + z2) / t3 );
		a[6][0].segment(i*n, s) =-h * mean( (3740 + 125*ze5*t1 + 1001*z + 111*z2 - 162*ze3*(20-7*z+z2))
						    / (162*z3) );
		a[6][3].segment(i*n, s) =h*20*mean( (-t8 - 25*ze5*t1) / (81*z3) );
		a[6][4].segment(i*n, s) =-h * mean( (2740 + 324*ze3*(-10+z) - 125*ze5*t1 + 1861*z + 519*z2) / (243*z3) );
		a[6][5].segment(i*n, s) =h*25*mean( (125*ze5*t1 + 162*ze3*t1 + 7*(164+35*z+3*z2)) / (486*z3) );
		a[7][0].segment(i*n, s) = h * mean( (t6 + 35*ze*(-180+82*z-17*z2+2*z3) + 28*(2070+547*z+110*z2+14*z3))
						    / (70*z4) );
		a[7][4].segment(i*n, s) = h*4*mean( (t6 - 140*ze*(45-13*z+z2) + 7*(8280+2338*z+525*z2+76*z3))
						    / (105*z4) );
		a[7][5].segment(i*n, s) = h*5*mean( (-t6 + 350*ze*(18-7*z+z2) - 7*(8280+2248*z+465*z2+61*z3))
						    / (147*z4) );
		a[7][6].segment(i*n, s) =h*27*mean( (125*ze5*t5 + 162*ze3*t5 + 35*ze*t5 - 14*(1380+398*z+95*z2+16*z3))
						    / (490*z4) );
	    

		b[0].segment(i*n, s) = h * mean( (90 + 34*z + 4*z2 + ze*(-90 + 56*z -15*z2 + 2*z3)) / (2*z4) );
		b[5].segment(i*n, s)=h*125*mean( (-18 - 8*z - z2 + 2*ze*(9 - 5*z + z2)) / (28*z4) );
		b[6].segment(i*n, s)=-h*27*mean( (ze*(30 - 12*z + z2) - 2*(15 + 9*z + 2*z2)) / (14*z4) );
		b[7].segment(i*n, s) =-h * mean( (90 + 64*z + 21*z2 + 4*z3 - 2*ze*(45 - 13*z + z2)) / (4*z4) );

	    }
	}
	else if (scheme == "IFRK43") {
	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();
	    
	    a[1][0] = h/2 * c[1];
	    a[3][2] = h * c[1];

	    b[0] = h * (1.0/6) * c[3];
	    b[1] = h * (1.0/3) * c[1];
	}
	else if (scheme == "IFRK54"){
	    c[1] = (hL/5).exp();
	    c[2] = (3*hL/10).exp();
	    c[3] = (4*hL/5).exp();
	    c[4] = (8*hL/9).exp();
	    c[5] = hL.exp();
	    
	    a[1][0] = (h/5) * c[1];
	    a[2][0] = (h*3/40) * c[2];
	    a[2][1] = (h*9/40) * (hL/10).exp();
	    a[3][0] = (h*44/45) * c[3];
	    a[3][1] = (-h*56.0/15) * (3*hL/5).exp();
	    a[3][2] = (h*32.0/9) * (hL/2).exp();
	    a[4][0] = (h*19372.0/6561) * c[4];
	    a[4][1] = (-h*25360.0/2187) * (31*hL/45).exp();
	    a[4][2] = (h*64448.0/6561) * (53*hL/90).exp();
	    a[4][3] = (-h*212.0/729) * (4*hL/45).exp();
	    a[5][0] = (h*9017.0/3168) * c[5];
	    a[5][1] = (-h*355.0/33) * (4*hL/5).exp();
	    a[5][2] = (h*46732.0/5247) * (7*hL/10).exp();
	    a[5][3] = (h*49.0/176) * c[1];
	    a[5][4] = (-h*5103.0/18656) * (hL/9).exp();

	    b[0] = (h*35.0/384) * c[5];
	    b[2] = (h*500./1113) * (7*hL/10).exp();
	    b[3] = (h*125./192) * c[1];
	    b[4] = (-h*2187./6784) * (hL/9).exp();
	}
	else if (scheme == "SSPP43") {
	    a1 = 0.268330095781759925;
	    a2 = -0.187991618799159782;
	    a3 = 0.919661523017399857;
	
	    c[0] = (a1*hL).exp();
	    c[1] = (a2*hL).exp();
	    c[2] = (a3*hL).exp();
	}
	else {
	    fprintf(stderr, "Please indicate the scheme !\n");
	    exit(1);
	}
	
	auto t_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> t_elapse = t_end - t_start;
	
	return t_elapse.count();
    }

    /// @brief multiplication
    /// @note in order to pass expressions as arguments, the arguments
    ///       must be declared as const
    inline 	
    virtual
    ArrayXXcd MUL(const Ary &C, const ArrayXXcd &Y){
	// for ArrayXcd * (ArrayXcd/ArrayXXcd): Y.colwise() * C 
	// for ArrayXd  * (ArrayXcd/ArrayXXcd): C.matrix().asDiagonal() * Y.matrix()
    }

    inline 
    virtual 
    ArrayXXcd ZR(Ary &z){}
    
    inline 
    virtual
    Ary mean(const Ref<const ArrayXXcd> &x){}

};


#endif	/* EID_H */



