#ifndef EID_H
#define EID_H

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

using namespace std;
using namespace Eigen;

// using namespace denseRoutines;

//////////////////////////////////////////////////////////////////////////////////
//              Exponetial Integrator with diagonal linear part
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief ETDRK4 integration class
 *
 * DT   : data type (double/complex)
 * NL   : the nonlinear function type
 */
template<typename DT>
class EID {
    
public:

    typedef std::complex<double> dcp;
    typedef Array<DT, Dynamic, 1> Ary;

    //////////////////////////////////////////////////////////////////////

    Ary &L;
    ArrayXcd *N, *Y;

    enum Scheme {
	Cox_Matthews,
	Krogstad,
	Hochbruck_Ostermann,
	Luan_Ostermann
    };
    Scheme scheme = Cox_Matthews; /* scheme  */
    
    std::unordered_map<int, int> nstages = { /* number of stages */
        {Cox_Matthews,        4},
        {Krogstad,            4},
        {Hochbruck_Ostermann, 5},
	{Luan_Ostermann,      8},
	{IFRK43,              4},
	{IFRK54,              5}
    };

    std::unordered_map<int, int> orders = { /* orders of schemes */
        {Cox_Matthews,        4},
        {Krogstad,            4},
        {Hochbruck_Ostermann, 4},
	{Luan_Ostermann,      8},
	{IFRK43,              4},
	{IFRK54,              5}
    };


    Ary a[9][9], b[9], c[9];
    
    int M = 64;			/* number of sample points */
    int R = 1;			/* radius for evaluating phi(z) */
    
    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-8;
    double nu = 0.9;	       /* safe factor */
    double mumax = 2.5;	       /* maximal time step increase factor */
    double mumin = 0.4;	       /* minimal time step decrease factor */
    double mue = 1.25;	       /* upper lazy threshold */
    double muc = 0.85;	       /* lower lazy threshold */

    int NCalCoe = 0;	      /* times to evaluate coefficient */
    int NReject = 0;	      /* times that new state is rejected */
    int NCallF = 0;	      /* times to call velocity function f */
    int NSteps = 0;	      /* total number of integrations steps */
    
    VectorXd hs;	       /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    
    double err = 0;	 /* LTE : local truncation error estimation */

    ////////////////////////////////////////////////////////////
    // constructor and desctructor
    EID(Ary &L, ArrayXcd *Y, ArrayXcd *N) : L(L), Y(Y), N(N){}
    ~EID(){}
    
    ////////////////////////////////////////////////////////////
 
    template<class NL, class SS>
    void 
    intg(NL nl, SS saveState, const double t0, const ArrayXcd &u0, const double tend, const double h0, 
	 const int skip_rate){
	int ns = nstages[scheme];
	int od = orders[scheme];
	double h = h0;
	calCoe(h);
	NCalCoe++;
	
	NCalCoe = 0;
	NReject = 0;
	NCallF = 0;    
	NSteps = 0;
	
	const int N = u0.size();    
	const int Nt = (int)round((tend-t0)/h);
	const int M = Nt /skip_rate + 1;

	double t = t0;
	Y[0] = u0;
	bool doChange, doAccept;

	bool TimeEnds = false;
	while(!TimeEnds){ 

	    if ( t + h > tend){
		h = tend - t;
		calCoe(h);
		NCalCoe++;
		TimeEnds = true;
	    }

	    oneStep(t, h, nl);
	    NCallF += ns+1;		
	    double s = nu * std::pow(rtol/du, 1.0/od);
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
		calCoe(h);
		NCalCoe++;
	    }
	}
    }
    
    template<class NL, class SS>
    void 
    intgC(NL nl, SS saveState, const double t0, const ArrayXcd &u0, const double tend, const double h, 
	  const int skip_rate){
	int ns = nstages[scheme];
	calCoe(h);
	NCalCoe++;
	NCallF = 0;
	NSteps = 0;

	const int Nt = (int)round((tend-t0)/h);

	double t = t0;
	Y[0] = u0;
	saveState(Y[0], 0);
	for(int i = 0; i < Nt; i++){
	    oneStep(t, h, nl);
	    NCallF += ns+1;
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

	if ( s > mumax) mu = mumax;
	else if (s > mue) mu = s;
	else if (s >= 1) {
	    mu = 1;
	    doChange = false;
	}
	else {
	    doAccept = false;
	    if (s > muc) mu = muc;
	    else if (s > mumin) mu = s;
	    else mu = mumin;
	}

	return mu;
    }
    
    /**
     * nonlinear class NL provides container for intermediate steps
     */
    template<class NL>
    void 
    oneStep(double t, double h, NL nl){

	switch (scheme) {
      
	case  Cox_Matthews : 
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + a[1][0]*N[0];
	    
	    nl(Y[1], N[1], t+h/2);	
	    Y[2] = c[1]*Y[0] + a[1][0]*N[1];
	    
	    nl(Y[2], N[2], t+h/2);	
	    Y[3] = c[1]*Y[0] + a[1][0]*(2*N[2]-N[0]);
	
	    nl(Y[3], N[3], t+h);	
	    Y[4] = c[3]*Y[0] + b[0]*N[0] + b[1]*(N[1] + N[2]) + b[3]*N[3];
	    
	    nl(Y[4], N[4], t+h);
	
	    err = (b[3]*(N[4] - N[3])).abs().maxCoeff() / Y[4].abs().maxCoeff();

	    break;
	
	case  Krogstad : 
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + a[1][0]*N[0];

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = c[1]*Y[0] + a[2][0]*N[0] + a[2][1]*N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = c[3]*Y[0] + a[3][0]*N[0] + a[3][2]*N[2];
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = c[3]*Y[0] + b[0]*N[0] + b[1]*(N[1] + N[2]) + b[3]*N[3];
	
	    nl(Y[4], N[4], t+h);
	
	    err = (b[3]*(N[4] - N[3])).abs().maxCoeff() / Y[4].abs().maxCoeff();

	    break;

	case Hochbruck_Ostermann :
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + a[1][0]*N[0];

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = c[1]*Y[0] + a[2][0]*N[0] + a[2][1]*N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = c[3]*Y[0] + a[3][0]*N[0] + a[3][1]*(N[1] + N[2]);
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = c[1]*Y[0] + a[4][0]*N[0] + a[4][1]*(N[1] + N[2]) + a[4][3]*N[3];
	
	    nl(Y[4], N[4], t+h/2);
	    Y[5] = c[3]*Y[0] + b[0]*N[0] + b[3]*N[3] + b[4]*N[4];

	    err = (b[4]*(N[4] - N[2])).abs().maxCoeff() / Y[5].abs().maxCoeff();

	    break;

	case Luan_Ostermann :
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + a[1][0]*N[0];

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = c[1]*Y[0] + a[2][0]*N[0] + a[2][1]*N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = c[3]*Y[0] + a[3][0]*N[0] + a[3][2]* N[2];
	
	    nl(Y[3], N[3], t+h/4);
	    Y[4] = c[1]*Y[0] + a[4][0]*N[0] + a[4][2]*N[2] + a[4][3]*N[3];
	
	    nl(Y[4], N[4], t+h/2);
	    Y[5] = c[5]*Y[0] + a[5][0]*N[0] + a[5][3]*N[3] + a[5][4]*N[4];

	    nl(Y[5], N[5], t+h/5);
	    Y[6] = c[6]*Y[0] + a[6][0]*N[0] + a[6][3]*N[3] + a[6][4]*N[4] + a[6][5]*N[5];

	    nl(Y[6], N[6], t+2*h/3);
	    Y[7] = c[7]*Y[0] + a[7][0]*N[0] + a[7][4]*N[4] + a[7][5]*N[5] + a[7][6]*N[6];
	    
	    nl(Y[7], N[7], t+h);
	    Y[8] = c[8]*Y[0] + b[0]*N[0] + b[5]*N[5] + b[6]*N[6] + b[7]*N[7];

	    nl(Y[8], N[8], t+h);
	    
	    err = (b[7]*(N[8] - N[7])).abs().maxCoeff() / Y[8].abs().maxCoeff();
	    
	    break;

	case IFRK43 :
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + 0.5 * c[1]*N[0];

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = c[1]*Y[0] + 0.5 * N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = c[3]*Y[0] + c[1]* N[2];
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = c[1]*Y[0] + (1.0/6)*c[3]*N[0] + (1.0/3)*c[1]*(N[1]+N[2]) + (1.0/6)*N[3];

	    nl(Y[4], N[4], t+h);

	    err = (1.0/10)*(N[4] - N[3]).abs().maxCoeff() / Y[4].abs().maxCoeff();
	    
	    break;
	    
	case IFRK54 :
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1]*Y[0] + 1.0/5*c[1]*N[0];

	    nl(Y[1], N[1], t+h/5);
	    Y[2] = c[2]*Y[0] + 3.0/40*c[2]*N[0] + a[2][1]* N[1];

	    nl(Y[2], N[2], t+3*h/10);
	    Y[3] = c[3]*Y[0] + 44.0/45*c[3]*N[0] + a[3][1]*N[1] + a[3][2]*N[2];
	
	    nl(Y[3], N[3], t+4*h/5);
	    Y[4] = c[4]*Y[0] + (19372.0/6561)*c[4]*N[0] + a[4][1]*N[1] + a[4][2]*N[2] + a[4][3]*N[3];

	    nl(Y[4], N[4], t+8*h/9);
	    Y[5] = c[5]*Y[0] + (9017.0/3168)*c[5]*N[0] + a[5][1]*N[1] + a[5][2]*N[2] + a[5][3]*N[3] + a[5][4]*N[4];

	    nl(Y[5], N[5], t+h);
	    Y[6] = c[5]*Y[0] + (35.0/384)*c[5]*N[0] + b[2]*N[2] + b[3]*N[3] + b[4]*N[4] + 11./84*N[5];

	    err = (1.0/10)*(N[4] - N[3]).abs().maxCoeff() / Y[4].abs().maxCoeff();
	    
	    break;

	default :
	    fprintf(stderr, "Please indicate the scheme !\n");
	    exit(1);
	}

    }

    void 
    calCoe(double h){

	switch (scheme) {
    
	case Cox_Matthews : {	    
	    Ary hL = h * L;
	    ArrayXXcd z = ZR(hL);

	    ArrayXXcd z2 = z.square();
	    ArrayXXcd z3 = z.cube();
	    ArrayXXcd ze = z.exp();
	    ArrayXXcd zeh = (z/2).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0] = h * mean( (zeh - 1)/z );
	    
	    b[0] = h * mean( (-4.0 - z + ze*(4.0 - 3.0 * z + z2)) / z3 );
	    b[1] = h*2*mean( (2.0 + z + ze*(-2.0 + z)) / z3 );
	    b[3] = h * mean( (-4.0 - 3.0*z -z2 + ze*(4.0 - z) ) / z3 );

	    break;
	}

	case Krogstad : {
	    Ary hL = h * L;
	    ArrayXXcd z = ZR(hL);

	    ArrayXXcd z2 = z.square();
	    ArrayXXcd z3 = z.cube();
	    ArrayXXcd ze = z.exp();
	    ArrayXXcd zeh = (z/2).exp();

	    ArrayXXcd t1 = z + 2;
	    ArrayXXcd t2 = z - 2;
	    ArrayXXcd t3 = z - 4;
	    ArrayXXcd t4 = z + 4;

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();
	    

	    a[1][0] = h * mean( (zeh - 1)/z );
	    a[2][0] = h * mean( (zeh*t3 + t4) / z2 );
	    a[2][1] = h*2*mean( (2*zeh - t1) / z2 );
	    a[3][0] = h * mean( (ze*t2 + t1) / z2 );
	    a[3][2] = h*2*mean( (ze - z - 1)  / z2 );
	    
	    b[0] = h * mean( (-t4 + ze*(4.0 - 3.0 * z + z2)) / z3 );
	    b[1] = h*2*mean( (t1 + ze*t2) / z3 );
	    b[3] = h * mean( (-4.0 - 3.0*z -z2 - ze*t3 ) / z3 );
	    
	    break;
	}
	    
	case Hochbruck_Ostermann : {
	    Ary hL = h * L;
	    ArrayXXcd z = ZR(hL);

	    ArrayXXcd z2 = z.square();
	    ArrayXXcd z3 = z.cube();
	    ArrayXXcd ze = z.exp();
	    ArrayXXcd zeh = (z/2).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    ArrayXXcd t1 = -4 + z;	   
	    ArrayXXcd t2 = 4*z3;
	    ArrayXXcd t3 = 20 + ze*t1;
	    ArrayXXcd t4 = 4 + z;

	    a[1][0] = h * mean( (zeh - 1)/z );
	    a[2][0] = h * mean( (zeh*t1 + t4) / z2 );
	    a[2][1] = h*2*mean( (2*zeh - z - 2) / z2 );
	    a[3][0] = h * mean( (ze*(z-2) + z + 2) / z2 );
	    a[3][1] = h * mean( (ze - z - 1)  / z2 );
	    a[4][0] =-h * mean( (t3 - z + z2 - 4*zeh*(4-3*z+z2))  / t2 );
	    a[4][1] = h * mean( (t3 + 3*z - z2 + 8*zeh*(-2-z))  / t2 );
	    a[4][2] =-h * mean( (t3 + 7*z + z2 + 4*zeh*t1)  / t2 );

	    b[0] = h * mean( (-t4 + ze*(4 - 3*z + z2)) / z3 );
	    b[3] =-h * mean( (4 + 3*z + z2 + ze*t1) / z3 );
	    b[4] = h*4*mean( (2 + z + ze*(-2 + z) ) / z3 );

	    break;
	}
	    
	case Luan_Ostermann : {
	    Ary hL = h * L;
	    ArrayXXcd z = ZR(hL);

	    ArrayXXcd z2 = z.square();
	    ArrayXXcd z3 = z.cube();
	    ArrayXXcd z4 = z2.square();
	    ArrayXXcd ze = z.exp();
	    ArrayXXcd zeh = (z/2).exp();
	    ArrayXXcd ze4 = (z/4).exp();
	    ArrayXXcd ze5 = (z/5).exp();
	    ArrayXXcd ze3 = (2*z/3).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    ArrayXXcd t1 = -4 + z;	   
	    ArrayXXcd t2 = 2*z2;
	    ArrayXXcd t3 = 25*z3;
	    ArrayXXcd t4 = 4 + z;
	    ArrayXXcd t5 = 60-14*z+z2;
	    ArrayXXcd t6 = -375*ze5*t5 - 486*ze3*t5;
	    
	    a[1][0] = h * mean( (zeh - 1)/z );
	    a[2][0] = h * mean( (zeh*(-2+z) + 2) / z2 );
	    a[2][1] = h * mean( (2*zeh - z - 2) / z2 );
	    a[3][0] = h * mean( (4 + 2*ze4*(-2+z)-z) / t2 );
	    a[3][2] = h * mean( (4*ze4 - t4)  / t2 );	    
	    a[4][0] = h * mean( (zeh*(16-6*z+z2) - 2*(8+z))  / z3 );
	    a[4][2] =-h * mean( (2*zeh*(-8+z) + 16 + 6*z + z2)  / z3 );
	    a[4][3] = h*8*mean( (zeh*t1 + t4)  / t2 );
	    a[5][0] = h * mean( (-400 + 70*z - 3*z2 + 25*ze5*(16-6*z+z2)) / t3 );
	    a[5][3] = h*8*mean( (100 + 25*ze5*t1 - 5*z - 3*z2) / t3 );
	    a[5][4] = h*2*mean( (-200 - 25*ze5*(-8+z) - 15*z + z2) / t3 );
	    a[6][0] =-h * mean( (3740 + 125*ze5*t1 + 1001*z + 111*z2 - 162*ze3*(20-7+z2)) / (162*z3) );
	    a[6][3] =h*20*mean( (-100 - 25*ze5*t1 + 5*z + 3*z2) / (81*t3) );
	    a[6][4] =-h * mean( (2740 + 324*ze3*(-10+z) - 125*ze5*t1 + 1861*z + 519*z2) / (243*z3) );
	    a[6][5] =h*25*mean( (125*ze5*t1 + 162*ze3*t1 + 7*(164+35*z+3*z2)) / (486*z3) );
	    a[7][0] = h * mean( (t6 + 35*ze*(-180+82*z-17*z2+2*z3) + 28*(2070+547*z+110*z2+14*z3)) / (70*z4) );
	    a[7][4] = h*4*mean( (t6 - 140*ze*(45-13*z+z2) + 7*(8280+2338*z+525*z2+76*z3)) / (105*z4) );
	    a[7][5] = h*5*mean( (-t6 + 350*ze*(18-7*z+z2) - 7*(8280+2248*z+465*z2+61*z3)) / (147*z4) );
	    a[7][6] =h*27*mean( (125*ze5*t5 + 162*ze3*t5 + 35*ze*t5 - 14*(1380+398*z+95*z2+16*z3)) / (490*z4) );
	    

	    b[0] = h * mean( (90 + 34*z + 4*z2 + ze*(-90 + 56*z -15*z2 + 2*z3)) / (2*z4) );
	    b[5]=h*125*mean( (-18 - 8*z - z2 + 2*ze*(9 - 5*z + z2)) / (28*z4) );
	    b[6]=-h*27*mean( (ze*(30 - 12*z + z2) - 2*(15 + 9*z + 2*z2)) / (14*z4) );
	    b[7] =-h * mean( (90 + 64*z + 21*z2 + 4*z3 - 2*ze*(45 - 13*z + z2)) / (4*z4) );

	    break;
	}

	case IFRK43 : {
	    Ary hL = h * L;

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    break;	
	}

	case IFRK54 : {
	    Ary hL = h * L;

	    c[1] = (hL/5).exp();
	    c[2] = (3*hL/10).exp();
	    c[3] = (4*hL/5).exp();
	    c[4] = (8*hL/9).exp();
	    c[5] = hL.exp();
	    
	    a[2][1] = 9.0/40 * (hL/10).exp();
	    a[3][1] = 56.0/15 * (3*hL/5).exp();
	    a[3][2] = 32.0/9 * (hL/2).exp();
	    a[4][1] = 25360.0/2187 * (31*hL/45).exp();
	    a[4][2] = 64448.0/6561 * (53*hL/90).exp();
	    a[4][3] = 212.0/729 * (4*hL/45).exp();
	    a[5][1] = 355.0/33 * (4*hL/5).exp();
	    a[5][2] = 46732.0/5247 * (7*hL/10).exp();
	    a[5][3] = 49.0/176 * c[1];
	    a[5][4] = 5103.0/18656 * (hL/9).exp();

	    b[2] = 500./1113 * (7*hL/10).exp();
	    b[3] = 125./192 * c[1];
	    b[4] = 2187./6784 * (hL/9).exp();

	    break;	
	}
	    
	default : fprintf(stderr, "Please indicate the scheme !\n");
	    
	}
	
    }
    
    inline 
    virtual 
    ArrayXXcd ZR(Ary &z){}
    
    inline 
    virtual
    Ary mean(ArrayXcd &x){}

};


#endif	/* EID_H */



