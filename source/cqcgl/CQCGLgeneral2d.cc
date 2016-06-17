#include "CQCGLgeneral2d.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>    // std::max

#define cee(x) (cout << (x) << endl << endl)

using namespace denseRoutines;
using namespace MyH5;
using namespace Eigen;
using namespace std;
using namespace MyFFT;

//////////////////////////////////////////////////////////////////////
//                        Class CQCGLgeneral                             //
//////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------ */
/* ----                constructor/destructor     ------- */
/* ------------------------------------------------------ */

/**
 * Constructor of cubic quintic complex Ginzburg-Landau equation
 * A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
 *
 * @param[in] N            the number of Fourier modes
 * @param[in] d            the spacial period, size
 * @param[in] doEnableTan   false : forbid tangent space integration 
 * @param[in] threadNum    number of threads for integration
 */
CQCGLgeneral2d::CQCGLgeneral2d(int N, int M, double dx, double dy,
			       double Mu, double Dr, double Di,
			       double Br, double Bi, double Gr,
			       double Gi,  
			       int threadNum)
    : N(N), M(M), dx(dx), dy(dy),
      Mu(Mu), Br(Br), Bi(Bi),
      Dr(Dr), Di(Di), Gr(Gr), Gi(Gi),
      
      F{ FFT2d(M, N, threadNum), 
	FFT2d(M, N, threadNum), 
	FFT2d(M, N, threadNum), 
	FFT2d(M, N, threadNum), 
	FFT2d(M, N, threadNum) },    
				      
				      JF{ FFT2d(M, N, threadNum), 
					      FFT2d(M, N, threadNum), 
					      FFT2d(M, N, threadNum), 
					      FFT2d(M, N, threadNum), 
					      FFT2d(M, N, threadNum) }      
{
    CGLInit(); // calculate coefficients.
}

CQCGLgeneral2d::~CQCGLgeneral2d(){}

CQCGLgeneral2d & CQCGLgeneral2d::operator=(const CQCGLgeneral2d &x){
    return *this;
}

/* ------------------------------------------------------ */
/* ----          Initialization functions         ------- */
/* ------------------------------------------------------ */

/**
 * @brief calculate the number of effective modes: Ne
 * @note N must be initialized first
 */
inline int CQCGLgeneral2d::calNe(const double N){
    return (N/3) * 2 - 1;	/* make it an odd number */
}


/**
 * @brief set the corret dimension of the system
 *     
 * Dealiasing is the method to calculate correct convolution term. For centralized
 * FFT, it works as set Fourier modes a_k = 0 for wave number |k| > 2/3 * N.
 * More specifically, in this code, the middle part of modes
 * is set to zero.
 *
 *    |<---------------------------------------------------------->|
 *                             FFT length: N
 *               
 *    |<--------------->|<--------------------->|<---------------->|
 *        (Ne + 1) / 2         N - Ne                (Ne - 1) /2
 *         = Nplus             = Nalias               = Nminus
 *
 *    @Note each term has real and imaginary part, so the indices are
 *          0, 1, 2,... Nplus*2, Nplus*2+1,... 2*Ne-1
 */
void CQCGLgeneral2d::CGLInit(){
    Ne = calNe(N);			
    Nplus = (Ne + 1) / 2;
    Nminus = (Ne - 1) / 2;
    Nalias = N - Ne;

    Me = calNe(M);
    Mplus = (Me + 1) / 2;
    Mminus = (Me - 1) / 2;
    Malias = M - Me;
    
    // calculate the Linear part
    Kx.resize(N,1);
    Kx << ArrayXd::LinSpaced(N/2, 0, N/2-1), 0, ArrayXd::LinSpaced(N/2-1, -N/2+1, -1); 
    Kx2.resize(Ne, 1);
    Kx2 << ArrayXd::LinSpaced(Nplus, 0, Nplus-1), ArrayXd::LinSpaced(Nminus, -Nminus, -1); 

    Ky.resize(M,1);
    Ky << ArrayXd::LinSpaced(M/2, 0, M/2-1), 0, ArrayXd::LinSpaced(M/2-1, -M/2+1, -1); 
    Ky2.resize(Me, 1);
    Ky2 << ArrayXd::LinSpaced(Mplus, 0, Mplus-1), ArrayXd::LinSpaced(Mminus, -Mminus, -1); 
    
    QKx = 2*M_PI/dx * Kx;  
    QKy = 2*M_PI/dy * Ky;
    
    L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					 QKy.square().replicate(1, N)); 
    L.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
    L.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
}

void CQCGLgeneral2d::changeOmega(double w){
    Omega = w;
    L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					 QKy.square().replicate(1, N)); 
    L.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
    L.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
}

/** 
 * @brief calculate the coefficients of ETDRK4 or Krogstad
 *
 * Note, we give up using
 *     Map<ArrayXcd> hLv(hL.data(), hL.size());
 *     ArrayXXcd LR = ZR(hLv);
 * to save memory
 */
void CQCGLgeneral2d::calCoe(const double h){
    
    ArrayXXcd hL = h*L;
    
    E = hL.exp();
    E2 = (hL/2).exp();
    
    a21.resize(M, N);
    b1.resize(M, N);
    b2.resize(M, N);
    b4.resize(M, N);
    
    if (Method == 2){
	a31.resize(M, N);
	a32.resize(M, N);
	a41.resize(M, N);
	a43.resize(M, N);
    }
    
    for (int i = 0; i < N; i++){

	ArrayXXcd LR = ZR(hL.col(i));

	ArrayXXcd LR2 = LR.square();
	ArrayXXcd LR3 = LR.cube();
	ArrayXXcd LRe = LR.exp();
	ArrayXXcd LReh = (LR/2).exp();
    
	a21.col(i) = h * ( (LReh - 1)/LR ).rowwise().mean(); 
	b1.col(i) = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
	b2.col(i) = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean();
	b4.col(i) = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();
    
	if (Method == 2) {
	    a31.col(i) = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean();
	    a32.col(i) = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean();
	    a41.col(i) = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean();
	    a43.col(i) = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean();
	}

    }
}


void 
CQCGLgeneral2d::oneStep(double &du, const bool onlyOrbit){

    if (1 == Method) {
	NL(0, onlyOrbit);
	
	F[1].v1 = E2 * F[0].v1 + a21 * F[0].v3;
	if(!onlyOrbit) JF[1].v1 = E2 * JF[0].v1 + a21 * JF[0].v3;
	NL(1, onlyOrbit);
	
	F[2].v1 = E2 * F[0].v1 + a21 * F[1].v3;
	if(!onlyOrbit) JF[2].v1 = E2 * JF[0].v1 + a21 * JF[1].v3;
	NL(2, onlyOrbit);
	
	F[3].v1 = E2 * F[1].v1 + a21 * (2*F[2].v3 - F[0].v3);
	if(!onlyOrbit) JF[3].v1 = E2 * JF[1].v1 + a21 * (2*JF[2].v3 - JF[0].v3);
	NL(3, onlyOrbit);

	F[4].v1 = E * F[0].v1 + b1 * F[0].v3 + b2 * (F[1].v3+F[2].v3) + b4 * F[3].v3;
	if(!onlyOrbit) JF[4].v1 = E * JF[0].v1 + b1 * JF[0].v3 + b2 * (JF[1].v3+JF[2].v3) + b4 * JF[3].v3;
	NL(4, onlyOrbit);

    }
    else {
	NL(0, onlyOrbit);

	F[1].v1 = E2 * F[0].v1 + a21 * F[0].v3; 
	if(!onlyOrbit) JF[1].v1 = E2 * JF[0].v1 + a21 * JF[0].v3; 
	NL(1, onlyOrbit);

	F[2].v1 = E2 * F[0].v1 + a31 * F[0].v3 + a32 * F[1].v3;
	if(!onlyOrbit) JF[2].v1 = E2 * JF[0].v1 + a31 * JF[0].v3 + a32 * JF[1].v3;
	NL(2, onlyOrbit);

	F[3].v1 = E * F[0].v1 + a41 * F[0].v3 + a43 * F[2].v3;
	if(!onlyOrbit) JF[3].v1 = E * JF[0].v1 + a41 * JF[0].v3 + a43 * JF[2].v3;
	NL(3, onlyOrbit);
	
	F[4].v1 = E * F[0].v1 + b1 * F[0].v3 + b2 * (F[1].v3+F[2].v3) + b4 * F[3].v3;
	if(!onlyOrbit) JF[4].v1 = E * JF[0].v1 + b1 * JF[0].v3 + b2 * (JF[1].v3+JF[2].v3) + b4 * JF[3].v3;
	NL(4, onlyOrbit);

    }

    // infinity norm
    du = (b4 * (F[4].v3-F[3].v3)).abs().maxCoeff() / F[4].v1.abs().maxCoeff();
    if(!onlyOrbit){
	double x = (b4 * (JF[4].v3-JF[3].v3)).abs().maxCoeff() / JF[4].v1.abs().maxCoeff();
	du = std::max(du, x);
    }
}



/**
 * @brief calcuate the matrix to do averge of phi(z). 
 */

ArrayXXcd CQCGLgeneral2d::ZR(const Ref<const ArrayXcd> &z){
    
    int M1 = z.size();
    ArrayXd K = ArrayXd::LinSpaced(MC, 1, MC); // 1,2,3,...,M 
    
    ArrayXXcd r = R * (K/MC*dcp(0,2*M_PI)).exp().transpose(); // row array.
    
    return z.replicate(1, MC) + r.replicate(M1, 1);
    
}

/**
 * @brief calculat the damping factor of time step
 *
 * @param[out] doChange    true if time step needs change
 * @param[out] doAccept    true if accept current time step
 * @param[in]  s           estimate damping factor
 * @return     mu          final dampling factor 
 */
double
CQCGLgeneral2d::adaptTs(bool &doChange, bool &doAccept, const double s){
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

void CQCGLgeneral2d::saveState(H5File &file, int id, const ArrayXXcd &a, 
			       const ArrayXXcd &v, const int flag){
    char groupName[10];
    sprintf (groupName, "%.6d", id);
    std::string s = "/"+std::string(groupName);
    Group group(file.createGroup(s));
    std:: string DS = s + "/";
    
    if (1 == flag || 0 == flag){
	writeMatrixXd(file, DS + "ar", a.real());
	writeMatrixXd(file, DS + "ai", a.imag());
    }
    if (2 == flag || 0 == flag) {
	writeMatrixXd(file, DS + "vr", v.real());
	writeMatrixXd(file, DS + "vi", v.imag());
    }
}

/**
 * @brief Constant time step integrator
 */
ArrayXXcd 
CQCGLgeneral2d::constETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
			 const double h, const int Nt, 
			 const int skip_rate, const bool onlyOrbit,
			 const bool doSaveDisk, const string fileName){
    int s = 1;
    if(!onlyOrbit) s = 2;
    
    ArrayXXcd aa;
    H5File file;
    if (doSaveDisk){
	file = H5File(fileName, H5F_ACC_TRUNC); /* openFile fails */
	saveState(file, 0, a0, v0, onlyOrbit ? 1 : 0);
    }
    else{
	const int M = (Nt+skip_rate-1)/skip_rate + 1;
	aa.resize(Me, Ne*M*s);
	aa.leftCols(Ne) = a0;
	if(!onlyOrbit) aa.middleCols(Ne, Ne) = v0;
    }
    
    F[0].v1 = pad(a0);
    if(!onlyOrbit) JF[0].v1 = pad(v0);
    lte.resize(M-1);
    NCallF = 0;

    calCoe(h);

    double du;
    int num = 0;
    for(int i = 0; i < Nt; i++){
	if( constETDPrint > 0 && i % constETDPrint == 0) fprintf(stderr, "%d/%d\n", i, Nt);
	
	oneStep(du, onlyOrbit);
	F[0].v1 = F[4].v1;	// update state
	if(!onlyOrbit) JF[0].v1 = JF[4].v1;
	NCallF += 5;
	if ( (i+1)%skip_rate == 0 || i == Nt-1) {
	    if(doSaveDisk){
		saveState(file, num+1, unpad(F[4].v1), unpad(JF[4].v1), onlyOrbit ? 1 : 0);
	    }
	    else{
		aa.middleCols((num+1)*Ne, Ne) = unpad(F[4].v1);
		if(!onlyOrbit) aa.middleCols((M+num+1)*Ne, Ne) = unpad(JF[4].v1);
	    }
	    lte(num++) = du;
	}
    }	
    
    return aa;
}



/**
 * @brief time step adaptive integrator
 */
ArrayXXcd
CQCGLgeneral2d::adaptETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
			 const double h0, const double tend, 
			 const int skip_rate, const bool onlyOrbit,
			 const bool doSaveDisk, const string fileName){
    
    int s = 1;
    if(!onlyOrbit) s = 2;
    
    double h = h0; 
    calCoe(h);

    const int Nt = (int)round(tend/h);
    const int M = (Nt+skip_rate-1) /skip_rate + 1;
    F[0].v1 = pad(a0);
    if(!onlyOrbit) JF[0].v1 = pad(v0);
						
    ArrayXXcd aa;
    H5File file;
    if (doSaveDisk) {
	file = H5File(fileName, H5F_ACC_TRUNC);
	saveState(file, 0, a0, v0, onlyOrbit ? 1:0);
    }
    else {
	aa.resize(Me, Ne*M*s);
	aa.leftCols(Ne) = a0;
	if(!onlyOrbit) aa.middleCols(Ne, Ne) = v0;
    }

    Ts.resize(M);
    Ts(0) = 0;

    NCalCoe = 0;
    NReject = 0;
    NCallF = 0;    
    NSteps = 0;
    hs.resize(M-1);
    lte.resize(M-1);

    double t = 0;
    double du = 0;
    int num = 1;
    bool doChange, doAccept;

    bool TimeEnds = false;
    while(!TimeEnds){ 

	if ( t + h > tend){
	    h = tend - t;
	    calCoe(h);
	    NCalCoe++;
	    TimeEnds = true;
	}

	oneStep(du, onlyOrbit);
	NCallF += 5;		
	double s = nu * std::pow(rtol/du, 1.0/4);
	double mu = adaptTs(doChange, doAccept, s);
	
	if (doAccept){
	    t += h;
	    NSteps++;
	    F[0].v1 = F[4].v1;
	    if(!onlyOrbit) JF[0].v1 = JF[4].v1;
	    if ( NSteps % skip_rate == 0 || TimeEnds) {
		if (num >= Ts.size()) {
		    int m = Ts.size();
		    Ts.conservativeResize(m+cellSize);
		    if(!doSaveDisk) aa.conservativeResize(Eigen::NoChange, (m+cellSize)*Ne*s); 
		    hs.conservativeResize(m-1+cellSize);
		    lte.conservativeResize(m-1+cellSize);
		}
		hs(num-1) = h;
		lte(num-1) = du;
		if(doSaveDisk) saveState(file, num, unpad(F[4].v1), unpad(JF[4].v1), onlyOrbit ? 1:0);
		else {
		    aa.middleCols(num*Ne*s, Ne) = unpad(F[4].v1);
		    if (!onlyOrbit) aa.middleCols(num*Ne*s+1, Ne) = unpad(JF[4].v1);
		}
		Ts(num) = t;
		num++;
	    }
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
    
    // lte = lte.head(num) has aliasing problem 
    hs.conservativeResize(num-1);
    lte.conservativeResize(num-1);
    Ts.conservativeResize(num);
    
    if (doSaveDisk) return aa;
    else return aa.leftCols(num*Ne*s);
}


/** @brief Integrator of 2d cqCGL equation.
 *
 */
ArrayXXcd 
CQCGLgeneral2d::intg(const ArrayXXcd &a0, const double h, const int Nt, const int skip_rate,
		     const bool doSaveDisk, const string fileName){
    
    assert(a0.rows() == Me && a0.cols() == Ne);
    ArrayXXcd v0(1, 1);
    return constETD(a0, v0, h, Nt, skip_rate, true, doSaveDisk, fileName);
}


ArrayXXcd
CQCGLgeneral2d::aintg(const ArrayXXcd &a0, const double h, const double tend, 
		      const int skip_rate, const bool doSaveDisk, const string fileName){
    assert(a0.rows() == Me && a0.cols() == Ne);
    ArrayXXcd v0(1, 1);
    return adaptETD(a0, v0, h, tend, skip_rate, true, doSaveDisk, fileName);
}

/**
 * @brief integrate the state and a subspace in tangent space
 */
ArrayXXcd
CQCGLgeneral2d::intgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
		      const int Nt, const int skip_rate,
		      const bool doSaveDisk, const string fileName){
    assert(a0.rows() == Me && a0.cols() == Ne && v0.rows() == Me && v0.cols() == Ne);
    ArrayXXcd aa = constETD(a0, v0, h, Nt, skip_rate, false, doSaveDisk, fileName);
    return aa;
}

ArrayXXcd 
CQCGLgeneral2d::aintgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
		       const double tend, const int skip_rate,
		       const bool doSaveDisk, const string fileName){
    assert(a0.rows() == Me && a0.cols() == Ne && v0.rows() == Me && v0.cols() == Ne);
    ArrayXXcd aa = adaptETD(a0, v0, h, tend, skip_rate, false, doSaveDisk, fileName);
    return aa;
}

/**
 * @brief get rid of the high modes
 */
void CQCGLgeneral2d::dealias(const int k, const bool onlyOrbit){
    F[k].v3.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
    F[k].v3.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
    if (!onlyOrbit) {
	JF[k].v3.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
	JF[k].v3.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
    }
}

/* 3 different stage os ETDRK4:
 *  v --> ifft(v) --> fft(g(ifft(v)))
 * */
void CQCGLgeneral2d::NL(const int k, const bool onlyOrbit){
    dcp B(Br, Bi);
    dcp G(Gr, Gi);

    if(onlyOrbit){
	F[k].ifft();
	ArrayXXcd A2 = (F[k].v2.real().square() + F[k].v2.imag().square()).cast<dcp>();
	F[k].v2 = B * F[k].v2 * A2 + G * F[k].v2 * A2.square();
	F[k].fft();

	dealias(k, onlyOrbit);
    }
    else {
	F[k].ifft();
	JF[k].ifft(); 
	
	ArrayXXcd aA2 = (F[k].v2.real().square() + F[k].v2.imag().square()).cast<dcp>();
	ArrayXXcd A2 = F[k].v2.square();
	
	F[k].v2 = B * F[k].v2 * aA2 + G * F[k].v2 * aA2.square();
	JF[k].v2 = JF[k].v2.conjugate() * ((B+G*2.0*aA2) * A2) + JF[k].v2 * ((2.0*B+3.0*G*aA2)*aA2);
	
	F[k].fft();
	JF[k].fft();
	
	dealias(k, onlyOrbit);
    }
}

ArrayXXcd CQCGLgeneral2d::unpad(const ArrayXXcd &v){
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

ArrayXXcd CQCGLgeneral2d::pad(const ArrayXXcd &v){
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

ArrayXXd CQCGLgeneral2d::c2r(const ArrayXXcd &v){
    return Map<ArrayXXd>((double*)&v(0,0), 2*v.rows(), v.cols());
}

ArrayXXcd CQCGLgeneral2d::r2c(const ArrayXXd &v){
    assert( 0 == v.rows() % 2);
    return Map<ArrayXXcd>((dcp*)&v(0,0), v.rows()/2, v.cols());
}



/* -------------------------------------------------- */
/* -------  Fourier/Configure transformation -------- */
/* -------------------------------------------------- */

/**
 * @brief back Fourier transform of the states. 
 */
ArrayXXcd CQCGLgeneral2d::Fourier2Config(const Ref<const ArrayXXcd> &aa){
    ArrayXXcd ap = pad(aa);
    int s = ap.cols() / N;
    ArrayXXcd AA(M, N*s);
    
    for (size_t i = 0; i < s; i++){
	F[0].v1 = ap.middleCols(i*N, N);
	F[0].ifft();
	AA.middleCols(i*N, N) = F[0].v2;
    }
    
    return AA;
}


/**
 * @brief Fourier transform of the states. Input and output are both real.
 */
ArrayXXcd CQCGLgeneral2d::Config2Fourier(const Ref<const ArrayXXcd> &AA){
    int s = AA.cols() / N;
    ArrayXXcd aa(M, N*s);
    
    for(size_t i = 0; i < s; i++){
	F[0].v2 = AA.middleCols(i*N, N);
	F[0].fft();
	aa.middleCols(i*N, N) = F[0].v3;
    }
    
    return unpad(aa);
}



/* -------------------------------------------------- */
/* --------            velocity field     ----------- */
/* -------------------------------------------------- */

/**
 * @brief velocity field
 */
ArrayXXcd CQCGLgeneral2d::velocity(const ArrayXXcd &a0){
    assert(a0.rows() == Me && a0.cols() == Ne);
    F[0].v1 = pad(a0);
    NL(0, true);
    ArrayXXcd vel = L*F[0].v1 + F[0].v3;
    return unpad(vel);
}

/**
 * @brief the generalized velociyt for relative equilibrium
 *
 *   v(x) + \omega_\tau * t_\tau(x) + \omega_\rho * t_\rho(x)
 */
ArrayXXcd CQCGLgeneral2d::velocityReq(const ArrayXXcd &a0, const double wthx,
				    const double wthy, const double wphi){
    return velocity(a0) + wthx*tangent(a0, 1) + wthy*tangent(a0, 2) + 
	wphi*tangent(a0, 3);
}


/* -------------------------------------------------- */
/* --------          stability matrix     ----------- */
/* -------------------------------------------------- */
/**
 * @brief calculate the product of stability matrix with a vector
 */
ArrayXXcd CQCGLgeneral2d::stab(const ArrayXXcd &a0, const ArrayXXcd &v0){
    assert(a0.rows() == Me && a0.cols() == Ne && v0.rows() == Me && v0.cols() == Ne);
    F[0].v1 = pad(a0);
    JF[0].v1 = pad(v0); 

    NL(0, false);
    ArrayXXcd Ax = L*JF[0].v1 + JF[0].v3;
    return unpad(Ax);
}

/**
 * @brief stability for relative equilbrium
 */
ArrayXXcd CQCGLgeneral2d::stabReq(const ArrayXXcd &a0, const ArrayXXcd &v0,
				  const double wthx, const double wthy,
				  const double wphi){
    ArrayXXcd z = stab(a0, v0);
    return z + wthx*tangent(v0, 1) + wthy*tangent(v0, 2) + wphi*tangent(v0, 3);
}

/* -------------------------------------------------- */
/* ------           symmetry related           ------ */
/* -------------------------------------------------- */

ArrayXXcd CQCGLgeneral2d::rotate(const Ref<const ArrayXXcd> &a0, const int mode, const double th1,
				 const double th2, const double th3){
    switch(mode) {

    case 1 : {		     /* translation rotation in x direction */
	double thx = th1;
	ArrayXXd th = (thx*Kx2).replicate(1, Me).transpose();

	return a0 * (dcp(0, 1) * th).exp();
    }
	
    case 2: {		     /* translation rotation in y direction */
	double thy = th1;
	ArrayXXd th = (thy*Ky2).replicate(1, Ne);

	return a0 * (dcp(0, 1) * th).exp();
    }

    case 3 : {			/* phase rotation */
	double phi = th1;
	return a0 * exp(dcp(0, 1)*phi); // a0*e^{i\phi}
    }
	
    case 4 : {	  /* translation rotation in both x and y direction */
	double thx = th1;
	double thy = th2;
	ArrayXXd th = (thx*Kx2).replicate(1, Me).transpose() + (thy*Ky2).replicate(1, Ne);

	return a0 * (dcp(0, 1) * th).exp();
    }

    case 5 : {			/* both rotate */
	double thx = th1;
	double thy = th2;
	double phi = th3;
	ArrayXXd th = (thx*Kx2).replicate(1, Me).transpose() + (thy*Ky2).replicate(1, Ne) + phi;
	
	return a0 * (dcp(0, 1) * th).exp();
    }

    default :{
	fprintf(stderr, "indicate a valid rotate mode !\n");
    }
	
    }

}

ArrayXXcd CQCGLgeneral2d::tangent(const Ref<const ArrayXXcd> &a0, const int mode){
    switch (mode) {
    
    case 1 : {		      /* translation tangent in x direction */
	ArrayXXcd R = dcp(0, 1) * Kx2;
	return a0 * R.replicate(1, Me).transpose();
    }

    case 2 : {		      /* translation tangent in y direction */
	ArrayXXcd R = dcp(0, 1) * Ky2;
	return a0 * R.replicate(1, Ne);
    }

    case 3 : {			/* phase rotation tangent  */
	return a0 * dcp(0, 1);
    }
	
    default :{
	fprintf(stderr, "indicate a valid tangent mode !\n");
    }
	
    }
}


#if 0 

/**
 * @brief rotate the state points in the full state space to the slice
 *
 * @param[in] aa       states in the full state space
 * @return    aaHat, theta, phi
 *
 * @note  g(theta, phi)(x+y) is different from gx+gy. There is no physical
 *        meaning to transform the sum/subtraction of two state points.
 */
std::tuple<ArrayXXd, ArrayXd, ArrayXd>
CQCGLgeneral::orbit2sliceWrap(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert(Ndim == n);
    ArrayXXd raa(n, m);
    ArrayXd th(m);
    ArrayXd phi(m);

    for(size_t i = 0; i < m; i++){
	double am1 = atan2(aa(n-1, i), aa(n-2, i));
	double a1 = atan2(aa(3, i), aa(2, i));
	phi(i) = 0.5 * (a1 + am1);
	th(i) = 0.5 * (a1 - am1);
	raa.col(i) = Rotate(aa.col(i), -th(i), -phi(i));
    }
    return std::make_tuple(raa, th, phi);
}

/**
 * @brief reduce the continous symmetries without wrapping the phase
 *        so there is no continuity
 * @see orbit2sliceWrap()
 */
std::tuple<ArrayXXd, ArrayXd, ArrayXd>
CQCGLgeneral::orbit2slice(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert(Ndim == n);
    ArrayXXd raa(n, m);
    ArrayXd th(m);
    ArrayXd phi(m);
    
    for(size_t i = 0; i < m; i++){
	double am1 = atan2(aa(n-1, i), aa(n-2, i));
	double a1 = atan2(aa(3, i), aa(2, i));
	phi(i) = 0.5 * (a1 + am1);
	th(i) = 0.5 * (a1 - am1);
    }

    const double M_2PI = 2 * M_PI;
    for(size_t i = 1; i < m; i++){
	double t0 = th(i) - th(i-1);
	double t1 = t0 - M_PI;
	double t2 = t0 + M_PI;
	double t0WrapAbs = fabs(remainder(t0, M_2PI));
	if(fabs(t1) < t0WrapAbs) { // theta jump pi up
	    th(i) = remainder(th(i) - M_PI, M_2PI);
	    phi(i) = remainder(phi(i) - M_PI, M_2PI);
	    continue;
	}
	if(fabs(t2) < t0WrapAbs) { // theta jump pi down
	    th(i) = remainder(th(i) + M_PI, M_2PI);
	    phi(i) = remainder(phi(i) + M_PI, M_2PI);
	}
    }
    
    for(size_t i = 0; i < m; i++){
	raa.col(i) = Rotate(aa.col(i), -th(i), -phi(i));
    }

    return std::make_tuple(raa, th, phi);
}

/**
 * @ simple version of orbit2slice(). Discard translation and phase information.
 */
ArrayXXd CQCGLgeneral::orbit2sliceSimple(const Ref<const ArrayXXd> &aa){
    auto tmp = orbit2slice(aa);
    return std::get<0>(tmp);
}


/** @brief project covariant vectors to 1st mode slice
 *
 * projection matrix is h = (I - |tx><tp|/<tx|tp>) * g(-th), so eigenvector |ve> is
 * projected to |ve> - |tx>*(<tp|ve>|/<tx|tp>), before which, g(-th) is 
 * performed.
 *
 * In 1th and -1st mode slice, template point is 
 * |xp_\rho>=(0, ...,1, 0) ==> |tp_\rho>=(0,0,0,...,0,1)
 * |xp_\tau>=(0,0,1,...,0) ==> |tp_\tau>=(0,0,0,1,...,0)
 * <tp_\rho|ve> = ve.bottomRows(1) // last row
 * <tp_\tau|ve> = ve.row(3) // 4th row
 *
 * @note vectors are not normalized
 */
MatrixXd CQCGLgeneral::ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x){
    int n = x.size();
    std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(x);
    ArrayXXd &xhat =  std::get<0>(tmp); // dimension [2*N, 1]
    double th = std::get<1>(tmp)[0];
    double phi = std::get<2>(tmp)[0];
    VectorXd tx_rho = phaseTangent(xhat);
    VectorXd tx_tau = transTangent(xhat);
	
    MatrixXd vep = Rotate(ve, -th, -phi);
    vep = vep - 0.5 * ((tx_rho - tx_tau) * vep.row(n-1) / xhat(n-2) +
		       (tx_rho + tx_tau) * vep.row(3) / xhat(2));
  
    return vep;

}

#endif
