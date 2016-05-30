#include "CQCGLgeneral2d.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>    // std::max

#define CE(x) (cout << (x) << endl << endl)

using namespace sparseRoutines;
using namespace denseRoutines;
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
 * @param[in] h            integration time step
 * @param[in] doEnableTan   false : forbid tangent space integration 
 * @param[in] Njacv        number of tangent vectors. so the number of fft columns
 *                         is Njacv + 1.
 *                         If Njacv <= 0 integrate Jacobian
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
 */
void CQCGLgeneral2d::calCoe(const double h){
    
    ArrayXXcd hL = h*L;
    Map<ArrayXcd> hLv(hL.data(), hL.size());
    ArrayXXcd LR = ZR(hLv);
    
    E = hL.exp();
    E2 = (hL/2).exp();
    
    ArrayXXcd LR2 = LR.square();
    ArrayXXcd LR3 = LR.cube();
    ArrayXXcd LRe = LR.exp();
    ArrayXXcd LReh = (LR/2).exp();
    
    a21 = h * ( (LReh - 1)/LR ).rowwise().mean(); 
    b1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
    b2 = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean();
    b4 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();
    
    a21.resize(M, N);
    b1.resize(M, N);
    b2.resize(M, N);
    b4.resize(M, N);

    if (Method == 2) {
	a31 = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean();
	a32 = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean();
	a41 = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean();
	a43 = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean();

	a31.resize(M, N);
	a32.resize(M, N);
	a41.resize(M, N);
	a43.resize(M, N);
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

/**
 * @brief Constant time step integrator
 */
ArrayXXcd 
CQCGLgeneral2d::constETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
			 const double h, const int Nt, 
			 const int skip_rate, const bool onlyOrbit){
    int s = 1;
    if(!onlyOrbit) s = 2;
    
    const int M = (Nt+skip_rate-1)/skip_rate + 1;
    F[0].v1 = pad(a0);
    if(!onlyOrbit) JF[0].v1 = pad(v0);
    ArrayXXcd aa(Me, Ne*M*s);
    aa.leftCols(Ne) = a0;
    if(!onlyOrbit) aa.middleCols(Ne*M, Ne) = v0;
    lte.resize(M-1);
    NCallF = 0;

    calCoe(h);

    double du;
    int num = 0;
    for(int i = 0; i < Nt; i++){
	oneStep(du, onlyOrbit);
	F[0].v1 = F[4].v1;	// update state
	if(!onlyOrbit) JF[0].v1 = JF[4].v1;
	NCallF += 5;
	if ( (i+1)%skip_rate == 0 || i == Nt-1) {
	    aa.middleCols((num+1)*Ne, Ne) = unpad(F[4].v1);
	    if(!onlyOrbit) aa.middleCols((M+num+1)*Ne, Ne) = unpad(JF[4].v1);
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
			 const int skip_rate, const bool onlyOrbit){
    
    int s = 1;
    if(!onlyOrbit) s = 2;
    
    double h = h0; 
    calCoe(h);

    const int Nt = (int)round(tend/h);
    const int M = (Nt+skip_rate-1) /skip_rate + 1;
    F[0].v1 = pad(a0);
    if(!onlyOrbit) JF[0].v1 = pad(v0);

    ArrayXXcd aa(Me, Ne*M*s);
    aa.leftCols(Ne) = a0;
    if(!onlyOrbit) aa.middleCols(Ne, Ne) = v0;
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
		    aa.conservativeResize(Eigen::NoChange, (m+cellSize)*Ne*s); 
		    hs.conservativeResize(m-1+cellSize);
		    lte.conservativeResize(m-1+cellSize);
		}
		hs(num-1) = h;
		lte(num-1) = du;
		aa.middleCols(num*Ne*s, Ne) = unpad(F[4].v1);
		if (!onlyOrbit) aa.middleCols(num*Ne*s+1, Ne) = unpad(JF[4].v1);
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
    
    return aa.leftCols(num*Ne*s);
}


/** @brief Integrator of 2d cqCGL equation.
 *
 */
ArrayXXcd 
CQCGLgeneral2d::intg(const ArrayXXcd &a0, const double h, const int Nt, const int skip_rate){
    
    assert(a0.rows() == Me && a0.cols() == Ne);
    ArrayXXcd v0(1, 1);
    return constETD(a0, v0, h, Nt, skip_rate, true);
}


ArrayXXcd
CQCGLgeneral2d::aintg(const ArrayXXcd &a0, const double h, const double tend, 
		      const int skip_rate){
    assert(a0.rows() == Me && a0.cols() == Ne);
    ArrayXXcd v0(1, 1);
    return adaptETD(a0, v0, h, tend, skip_rate, true);
}

/**
 * @brief integrate the state and a subspace in tangent space
 */
ArrayXXcd
CQCGLgeneral2d::intgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
		      const int Nt, const int skip_rate){
    assert(a0.rows() == Me && a0.cols() == Ne && v0.rows() == Me && v0.cols() == Ne);
    ArrayXXcd aa = constETD(a0, v0, h, Nt, skip_rate, false);
    return aa;
}

ArrayXXcd 
CQCGLgeneral2d::aintgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
		       const double tend, const int skip_rate){
    assert(a0.rows() == Me && a0.cols() == Ne && v0.rows() == Me && v0.cols() == Ne);
    ArrayXXcd aa = adaptETD(a0, v0, h, tend, skip_rate, false);
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
	
	ArrayXcd aA2 = (F[k].v2.real().square() + F[k].v2.imag().square()).cast<dcp>();
	ArrayXcd A2 = F[k].v2.square();
	
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

#if 0 

/* -------------------------------------------------- */
/* --------            velocity field     ----------- */
/* -------------------------------------------------- */

/**
 * @brief velocity field
 */
ArrayXd CQCGLgeneral::velocity(const ArrayXd &a0){
    assert( Ndim == a0.rows() );
    F[0].v1 = R2C(a0);
    NL(0, true);
    ArrayXcd vel = L*F[0].v1 + F[0].v3;
    return C2R(vel);
}

/**
 * @brief the generalized velociyt for relative equilibrium
 *
 *   v(x) + \omega_\tau * t_\tau(x) + \omega_\rho * t_\rho(x)
 */
ArrayXd CQCGLgeneral::velocityReq(const ArrayXd &a0, const double wth,
				  const double wphi){
    return velocity(a0) + wth*transTangent(a0) + wphi*phaseTangent(a0);    
}

/**
 * velocity in the slice
 *
 * @param[in]  aH  state in the slice
 */
VectorXd CQCGLgeneral::velSlice(const Ref<const VectorXd> &aH){
    VectorXd v = velocity(aH);
    
    Vector2d c;
    c << v(Ndim-1), v(3);
    
    Matrix2d Linv;
    Linv << 0.5/aH(Ndim-2), 0.5/aH(2),
	-0.5/aH(Ndim-2), 0.5/aH(2); 

    VectorXd tp = phaseTangent(aH);
    VectorXd tt = transTangent(aH);
    MatrixXd vs(Ndim, 2);
    vs << tp, tt;

    return v - vs * (Linv * c);
}

VectorXd CQCGLgeneral::velPhase(const Ref<const VectorXd> &aH){
    VectorXd v = velocity(aH);
 
    VectorXd tp = phaseTangent(aH);   
    double c = v(Ndim-1) / aH(Ndim-2);

    return v - c * tp;
}

/* -------------------------------------------------- */
/* --------          stability matrix     ----------- */
/* -------------------------------------------------- */
MatrixXd CQCGLgeneral::stab(const ArrayXd &a0){
    ArrayXXd v0(Ndim, Ndim+1); 
    v0 << a0, MatrixXd::Identity(Ndim, Ndim);
    JF[0].v1 = R2C(v0);
    NL(0, false);
    ArrayXXcd j0 = R2C(MatrixXd::Identity(Ndim, Ndim));
    MatrixXcd Z = j0.colwise() * L + JF[0].v3.rightCols(Ndim);
  
    return C2R(Z);
}

/**
 * @brief stability for relative equilbrium
 */
MatrixXd CQCGLgeneral::stabReq(const ArrayXd &a0, double wth, double wphi){
    MatrixXd z = stab(a0);
    return z + wth*transGenerator() + wphi*phaseGenerator();
}

/**
 * @brief stability exponents of req
 */
VectorXcd CQCGLgeneral::eReq(const ArrayXd &a0, double wth, double wphi){
    return eEig(stabReq(a0, wth, wphi));
}

/**
 * @brief stability vectors of req
 */
MatrixXcd CQCGLgeneral::vReq(const ArrayXd &a0, double wth, double wphi){
    return vEig(stabReq(a0, wth, wphi));
}

/**
 * @brief stability exponents and vectors of req
 */
std::pair<VectorXcd, MatrixXcd>
CQCGLgeneral::evReq(const ArrayXd &a0, double wth, double wphi){
    return evEig(stabReq(a0, wth, wphi));
}


/* -------------------------------------------------- */
/* ------           symmetry related           ------ */
/* -------------------------------------------------- */

/**
 * @brief reflect the states
 *
 * Reflection : a_k -> a_{-k}. so a_0 keeps unchanged
 */
ArrayXXd CQCGLgeneral::reflect(const Ref<const ArrayXXd> &aa){
    ArrayXXcd raa = R2C(aa);
    const int n = raa.rows(); // n is an odd number
    for(size_t i = 1; i < (n+1)/2; i++){
	ArrayXcd tmp = raa.row(i);
	raa.row(i) = raa.row(n-i);
	raa.row(n-i) = tmp;
    }
    return C2R(raa);
}

/**
 * @ brief calculate (x^2 - y^2) / \sqrt{x^2 + y^2}
 */
inline ArrayXd CQCGLgeneral::rcos2th(const ArrayXd &x, const ArrayXd &y){
    ArrayXd x2 = x.square();
    ArrayXd y2 = y.square();
    return (x2 - y2) / (x2 + y2).sqrt();
}

/**
 * @ brief calculate x * y / \sqrt{x^2 + y^2}
 */
inline ArrayXd CQCGLgeneral::rsin2th(const ArrayXd &x, const ArrayXd &y){
    return x * y / (x.square() + y.square()).sqrt();
}

/**
 * @brief calculate the gradient of rcos2th()
 *
 *        partial derivative over x :   (x^3 + 3*x*y^2) / (x^2 + y^2)^{3/2}
 *        partial derivative over y : - (y^3 + 3*y*x^2) / (x^2 + y^2)^{3/2}
 */
inline double CQCGLgeneral::rcos2thGrad(const double x, const double y){
    // only return derivative over x. Derivative over y can be obtained
    // by exchange x and y and flip sign
    double denorm = sqrt(x*x + y*y);
    double denorm3 = denorm * denorm * denorm;
    return x * (x*x + 3*y*y) / denorm3;
}

/**
 * @brief calculate the gradient of rsin2th()
 *
 *        partial derivative over x :   y^3 / (x^2 + y^2)^{3/2}
 *        partial derivative over y :   x^3 / (x^2 + y^2)^{3/2}
 */
inline double CQCGLgeneral::rsin2thGrad(const double x, const double y){
    // only return derivative over x. Derivative over y can be obtained
    // by exchange x and y
    double denorm = sqrt(x*x + y*y);
    double denorm3 = denorm * denorm * denorm;
    return y*y*y / denorm3;
}

/**
 * @brief the first step to reduce the discrete symmetry
 *
 * @param[in] aaHat   states after reducing continous symmetries
 */
ArrayXXd CQCGLgeneral::reduceRef1(const Ref<const ArrayXXd> &aaHat){
    const int m = aaHat.cols(); 
    const int n = aaHat.rows(); 
    assert(n == Ndim);
    
    ArrayXXd step1(n, m);
    step1.topRows<2>() = aaHat.topRows<2>();
    for(size_t i = 1; i < Nplus; i++){
	step1.row(2*i) = 0.5*(aaHat.row(2*i) - aaHat.row(n-2*i));
	step1.row(n-2*i) = 0.5*(aaHat.row(2*i) + aaHat.row(n-2*i));
	step1.row(2*i+1) = 0.5*(aaHat.row(2*i+1) - aaHat.row(n+1-2*i));
	step1.row(n+1-2*i) = 0.5*(aaHat.row(2*i+1) + aaHat.row(n+1-2*i));
    }

    return step1;
}

ArrayXXd CQCGLgeneral::reduceRef2(const Ref<const ArrayXXd> &step1){
    ArrayXXd step2(step1);
    ArrayXd p1s = step1.row(2).square(); 
    ArrayXd q1s = step1.row(3).square();
    ArrayXd denorm = (p1s + q1s).sqrt();
    step2.row(2) = (p1s - q1s) / denorm;
    step2.row(3) = step1.row(2) * step1.row(3) / denorm.transpose(); 
    
    for(size_t i = 4; i < 2*Nplus; i++){
	ArrayXd denorm = (step1.row(i-1).square() +  step1.row(i).square()).sqrt();
	step2.row(i) = step1.row(i-1) * step1.row(i) / denorm.transpose() ;
    }

    return step2;
}

/**
 * @brief get the indices which reflect sign in the 3rd step of reflection
 *        reduction
 *
 *        1, 4, 6, ...
 */
std::vector<int> CQCGLgeneral::refIndex3(){
    std::vector<int> index; // vector storing indices which flip sign
    index.push_back(1);
    for(size_t i = 2; i < Nplus; i++) index.push_back(2*i);
    for(size_t i = Nplus; i < Ne; i++) {
	if(i%2 != 0){		// the last mode a_{-1} has index Ne-1 even
	    index.push_back(2*i);
	    index.push_back(2*i+1);
	}
    }
    return index;
}


/**
 * @brief the 3rd step to reduce the discrete symmetry
 *
 */
ArrayXXd CQCGLgeneral::reduceRef3(const Ref<const ArrayXXd> &aa){

    ArrayXXd aaTilde(aa);
    aaTilde.row(0) = rcos2th(aa.row(0), aa.row(1));
    aaTilde.row(1) = rsin2th(aa.row(0), aa.row(1));
    
    std::vector<int> index = refIndex3();
    for(size_t i = 1; i < index.size(); i++){
	aaTilde.row(index[i]) = rsin2th(aa.row(index[i-1]), aa.row(index[i]));
    }

    return aaTilde;
}

ArrayXXd CQCGLgeneral::reduceReflection(const Ref<const ArrayXXd> &aaHat){
    return reduceRef3(reduceRef2(reduceRef1(aaHat)));
}

/**
 * @brief The gradient of the reflection reduction transformation for the
 *        firt step.
 *
 * step 1: ---------------------------------------
 *         | 1					 |
 *         |   1				 |
 *         |     1/2                   -1/2	 |
 *         |         1/2                   -1/2	 |
 *         |                   ...		 |
 *         |             1/2   -1/2		 |
 *         |                1/2    -1/2		 |
 *         |             1/2    1/2		 |
 *         |                1/2     1/2		 |
 *         |                   ...		 |
 *         |     1/2                   1/2 	 |
 *         |         1/2                   1/2   |
 *         ---------------------------------------
 */
MatrixXd CQCGLgeneral::refGrad1(){
    MatrixXd Gamma(MatrixXd::Zero(Ndim, Ndim));
    Gamma(0, 0) = 1;
    Gamma(1, 1) = 1;
    for (size_t i = 1; i < Nplus; i++){
	Gamma(2*i, 2*i) = 0.5;
	Gamma(2*i+1, 2*i+1) = 0.5;
	Gamma(2*i, Ndim - 2*i) = -0.5;
	Gamma(2*i+1, Ndim - 2*i + 1) = -0.5;
    }
    for(size_t i = Nplus; i < Ne; i++){
	Gamma(2*i, 2*i) = 0.5;
	Gamma(2*i, Ndim - 2*i) = 0.5;
	Gamma(2*i+1, 2*i+1) = 0.5;
	Gamma(2*i+1, Ndim - 2*i + 1) = 0.5;
    }
    return Gamma;
}

/**
 * @brief The gradient of the reflection reduction transformation for the
 *        2nd step.
 *        
 * step 2: ------------------------------------
 *         | 1				      |
 *         |   1			      |
 *         |     *  *                         |
 *         |     *  *                         |
 *         |        *  *         	      |
 *         |           *  *      	      |
 *         |               ...  	      |
 *         |                  *  *            |
 *         |                       1          |
 *         |                         1        |
 *         |                           ...    |
 *         |                               1  |
 *         ------------------------------------
 *               
 */
MatrixXd CQCGLgeneral::refGrad2(const ArrayXd &x){
    assert (x.size() == Ndim);
    MatrixXd Gamma(MatrixXd::Zero(Ndim, Ndim));
    Gamma(0, 0) = 1;
    Gamma(1, 1) = 1;
    Gamma(2, 2) = rcos2thGrad(x(2), x(3));
    Gamma(2, 3) = - rcos2thGrad(x(3), x(2));
    for (size_t i = 3; i < 2*Nplus; i++){
	Gamma(i, i) = rsin2thGrad(x(i), x(i-1));
	Gamma(i, i-1) = rsin2thGrad(x(i-1), x(i));
    }
    for (size_t i = 2*Nplus; i < Ndim; i++){
	Gamma(i, i) = 1;
    }
    return Gamma;
}

/**
 * @brief The gradient of the reflection reduction transformation for the
 *        3rd step.
 */
MatrixXd CQCGLgeneral::refGrad3(const ArrayXd &x){
    assert(x.size() == Ndim);
    MatrixXd Gamma(MatrixXd::Identity(Ndim, Ndim));
    std::vector<int> index = refIndex3();
    Gamma(0, 0) = rcos2thGrad(x(0), x(1));
    Gamma(0, 1) = - rcos2thGrad(x(1), x(0));
    Gamma(1, 1) = rsin2thGrad(x(1), x(0));
    Gamma(1, 0) = rsin2thGrad(x(0), x(1));
    
    for(size_t i = 1; i < index.size(); i++){
	Gamma(index[i], index[i]) = rsin2thGrad(x(index[i]), x(index[i-1]));
	Gamma(index[i], index[i-1]) = rsin2thGrad(x(index[i-1]), x(index[i]));
    }
    return Gamma;
}

/**
 * @brief calculate the tranformation matrix for reflection reduction
 */
MatrixXd CQCGLgeneral::refGradMat(const ArrayXd &x){
    ArrayXd step1 = reduceRef1(x); 
    ArrayXd step2 = reduceRef2(step1);
    return refGrad3(step2) * refGrad2(step1) * refGrad1();
}

/**
 * @brief transform covariant vectors after reducing reflection
 *
 * @param[in] veHat    covariant vectors after reducing the continuous symmetries.
 * @param[in] xHat     orbit point after reducing continuous symmetries.
 */
MatrixXd CQCGLgeneral::reflectVe(const MatrixXd &veHat, const Ref<const ArrayXd> &xHat){
    MatrixXd Gamma = refGradMat(xHat);
    return Gamma * veHat;
}

/** @beief reduce reflection symmetry of all the Floquet vectors along a po
 *
 *  Usaully, aaHat has one more column the the Floquet vectors, so you can
 *  call this function like:
 *  \code
 *      reflectVeAll(veHat, aaHat.leftCols(aa.cols()-1))
 *  \endcode
 *  
 *  @param[in] veHat   Floquet vectors along the orbit in the 1st mode slice.
 *                     Dimension: [N, M*Trunc]
 *  @param[in] aaHat   the orbit in the  slice
 *  @param[in] trunc   the number of vectors at each orbit point.
 *                     trunc = 0 means full set of vectors
 *  @return            transformed to the reflection invariant space.
 *                     Dimension [N, M*Trunc]
 *
 *  @note vectors are not normalized
 */
MatrixXd CQCGLgeneral::reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
				    const int trunc /* = 0*/){
    int Trunc = trunc;
    if(trunc == 0) Trunc = veHat.rows();

    assert(veHat.cols() % Trunc == 0);
    const int n = veHat.rows();  
    const int m = veHat.cols()/Trunc;
    const int n2 = aaHat.rows();
    const int m2 = aaHat.cols();

    assert(m == m2 && n == n2);
    MatrixXd veTilde(n, Trunc*m);
    for(size_t i = 0; i < m; i++){
	veTilde.middleCols(i*Trunc, Trunc) =
	    reflectVe(veHat.middleCols(i*Trunc, Trunc), aaHat.col(i));
    }

    return veTilde;
}


/** @brief group rotation for spatial translation of set of arrays.
 *  th : rotation angle
 *  */
ArrayXXd CQCGLgeneral::transRotate(const Ref<const ArrayXXd> &aa, const double th){
    ArrayXcd R = ( dcp(0,1) * th * K2 ).exp(); // e^{ik\theta}
    ArrayXXcd raa = r2c(aa); 
    raa.colwise() *= R;
  
    return c2r(raa);
}

/** @brief group tangent in angle unit.
 *
 *  x=(b0, c0, b1, c1, b2, c2 ...) ==> tx=(0, 0, -c1, b1, -2c2, 2b2, ...)
 */
ArrayXXd CQCGLgeneral::transTangent(const Ref<const ArrayXXd> &aa){
    ArrayXcd R = dcp(0,1) * K2;
    ArrayXXcd raa = r2c(aa);
    raa.colwise() *= R;
  
    return c2r(raa);
}

/** @brief group generator. */
MatrixXd CQCGLgeneral::transGenerator(){
    MatrixXd T = MatrixXd::Zero(Ndim, Ndim);
    for(size_t i = 0; i < Ne; i++){
	T(2*i, 2*i+1) = -K2(i);
	T(2*i+1, 2*i) = K2(i);
    }
    return T;
}


/** @brief group transform for complex rotation
 * phi: rotation angle
 * */
ArrayXXd CQCGLgeneral::phaseRotate(const Ref<const ArrayXXd> &aa, const double phi){
    return c2r( r2c(aa) * exp(dcp(0,1)*phi) ); // a0*e^{i\phi}
}

/** @brief group tangent.  */
ArrayXXd CQCGLgeneral::phaseTangent(const Ref<const ArrayXXd> &aa){
    return c2r( r2c(aa) * dcp(0,1) );
}

/** @brief group generator  */
MatrixXd CQCGLgeneral::phaseGenerator(){
    MatrixXd T = MatrixXd::Zero(Ndim, Ndim);
    for(size_t i = 0; i < Ne; i++){
	T(2*i, 2*i+1) = -1;
	T(2*i+1, 2*i) = 1;
    }
    return T;
}

/**
 * @brief apply both continous symmetries
 *
 * @note     for performance purpose, this function is not written as the
 *           superposition of 2 functions
 */
ArrayXXd CQCGLgeneral::Rotate(const Ref<const ArrayXXd> &aa, const double th,
			      const double phi){
    ArrayXcd R = ( dcp(0,1) * (th * K2 + phi) ).exp(); // e^{ik\theta + \phi}
    ArrayXXcd raa = r2c(aa); 
    raa.colwise() *= R;
  
    return c2r(raa);
}

/**
 * @brief rotate the whole orbit with different phase angles at different point
 */
ArrayXXd CQCGLgeneral::rotateOrbit(const Ref<const ArrayXXd> &aa, const ArrayXd &th,
				   const ArrayXd &phi){
    const int m = aa.cols();
    const int n = aa.rows();
    const int m2 = th.size();
    const int m3 = phi.size();
    assert( m == m2 && m2 == m3);

    ArrayXXd aaHat(n, m);
    for( size_t i = 0; i < m; i++){
	aaHat.col(i) = Rotate(aa.col(i), th(i), phi(i));
    }

    return aaHat;
}

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

/**
 * @brief a wrap function => reduce all symmetries of an orbit
 */
std::tuple<ArrayXXd, ArrayXd, ArrayXd>
CQCGLgeneral::reduceAllSymmetries(const Ref<const ArrayXXd> &aa){
    std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(aa);
    return std::make_tuple(reduceReflection(std::get<0>(tmp)),
			   std::get<1>(tmp), std::get<2>(tmp));
}

/**
 * @brief a wrap function => reduce all the symmetries of covariant vectors
 */
MatrixXd CQCGLgeneral::reduceVe(const ArrayXXd &ve, const Ref<const ArrayXd> &x){
    std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(x);
    return reflectVe(ve2slice(ve, x), std::get<0>(tmp).col(0));
}

#if 0
/* -------------------------------------------------- */
/* --------          shooting related     ----------- */
/* -------------------------------------------------- */

/**
 * @brief form the multishooting vector
 *
 *  df  = [ f(x_0, nstp) - x_1,
 *          f(x_1, nstp) - x_2,
 *          ...
 *          g(\theta, \phi) f(x_{M-1}, nstp) - x_0 ]
 * 
 * @param[in] x      a stack of state vectors on a orbit. Each column is one point
 * @param[in] nstp   integration time steps for each point in x
 * @param[in] th     translation rotation angle
 * @param[in] phi    phase change angle
 */
VectorXd CQCGLgeneral::multiF(const ArrayXXd &x, const int nstp, const double th, const double phi){
    int m = x.cols();
    int n = x.rows();
    assert( Ndim == n );
    
    VectorXd F(m*n);
    for(size_t i = 0; i < m; i++){
	ArrayXXd aa = intg(x.col(i), nstp, nstp);
	if(i < m-1) F.segment(i*n, n) = aa.col(1) - x.col(i+1);
	else F.segment(i*n, n) =  Rotate(aa.col(1), th, phi)  - x.col((i+1)%m);
    }

    return F;
}

/**
 * @brief form the multishooting matrix and the difference vector
 *
 *  df  = [ f(x_0, nstp) - x_1,
 *          f(x_1, nstp) - x_2,
 *          ...
 *          g(\theta, \phi) f(x_{M-1}, nstp) - x_0 ]
 *          
 *  A relative periodic orbit has form \f$ x(0) = g_\tau(\theta) g_\rho(\phi) x(T_p) \f$.
 *  Take derivative of the above vector, we obtain
 *             [ J(x_0)   -I                                      v(f(x_0))         
 *                       J(x_1)   -I                              v(f(x_1))
 *  Jacobian =                    ....                             ...
 *               -I                   g(\theta, \phi)J(x_{M-1})   v(f(x_{M-1}))  tx1  tx2 ]
 *  
 */
pair<CQCGLgeneral::SpMat, VectorXd>
CQCGLgeneral::multishoot(const ArrayXXd &x, const int nstp, const double th,
			 const double phi, bool doesPrint /* = false*/){
    int m = x.cols();		/* number of shooting points */
    int n = x.rows();
    assert( Ndim == n );
  
    SpMat DF(m*n, m*n+3);
    VectorXd F(m*n);
    std::vector<Tri> nz;
    nz.reserve(2*m*n*n);
    
    if(doesPrint) printf("Forming multishooting matrix:");


    for(size_t i = 0 ; i < m; i++){
	if(doesPrint) printf("%zd ", i);
	std::pair<ArrayXXd, ArrayXXd> aadaa = intgj(x.col(i), nstp, nstp, nstp); 
	ArrayXXd &aa = aadaa.first;
	ArrayXXd &J = aadaa.second;
	
	if(i < m-1){
	    // J
	    std::vector<Tri> triJ = triMat(J, i*n, i*n);
	    // velocity
	    std::vector<Tri> triv = triMat(velocity(aa.col(1)), i*n, m*n);

	    {
		nz.insert(nz.end(), triJ.begin(), triJ.end());
		nz.insert(nz.end(), triv.begin(), triv.end());
		// f(x_i) - x_{i+1}
		F.segment(i*n, n) = aa.col(1) - x.col(i+1);
	    }
	    
	} else {
	    ArrayXd gfx = Rotate(aa.col(1), th, phi); /* gf(x) */
	    // g*J
	    std::vector<Tri> triJ = triMat(Rotate(J, th, phi), i*n, i*n);
	    // R*velocity
	    std::vector<Tri> triv = triMat(Rotate(velocity(aa.col(1)), th, phi), i*n, m*n);
	    // T_\tau * g * f(x_{m-1})
	    VectorXd tx_trans = transTangent(gfx) ;
	    std::vector<Tri> tritx_trans = triMat(tx_trans, i*n, m*n+1);	    
	    // T_\phi * g * f(x_{m-1})
	    ArrayXd tx_phase = phaseTangent( gfx );
	    std::vector<Tri> tritx_phase = triMat(tx_phase, i*n, m*n+2);
	    

	    {
		nz.insert(nz.end(), triJ.begin(), triJ.end());
		nz.insert(nz.end(), triv.begin(), triv.end());
		nz.insert(nz.end(), tritx_trans.begin(), tritx_trans.end());
		nz.insert(nz.end(), tritx_phase.begin(), tritx_phase.end());
		// g*f(x_{m-1}) - x_0
		F.segment(i*n, n) = gfx  - x.col((i+1)%m);
	    }
	}
	
	std::vector<Tri> triI = triDiag(n, -1, i*n, ((i+1)%m)*n);

	nz.insert(nz.end(), triI.begin(), triI.end());
    }
    
    if(doesPrint) printf("\n");
    
    DF.setFromTriplets(nz.begin(), nz.end());

    return make_pair(DF, F);
}

/**
 * @brief calculate the Jacobian for finding relative equilibria
 * 
 */
std::pair<MatrixXd, VectorXd>
CQCGLgeneral::newtonReq(const ArrayXd &a0, const double wth, const double wphi){
    int n = a0.rows();
    assert(Ndim == n);
    
    MatrixXd DF(n, n+2); 
    ArrayXd tx_trans = transTangent(a0);
    ArrayXd tx_phase = phaseTangent(a0);
    DF.leftCols(n) = stabReq(a0, wth, wphi); 
    DF.col(n)= tx_trans;
    DF.col(n+1) = tx_phase;

    VectorXd F(n);
    F.head(n) = velocity(a0) + wth*tx_trans + wphi*tx_phase;

    return make_pair(DF, F);
}

/**
 * @brief use Levenberg-Marquardt algorithm to find relative equlibirum
 *
 * To find relative equilibrium, we do not need to integrate the system. We only
 * need to use velocity
 *
 * @param[in] a0          initial guess of relative equilibrium
 * @param[in] wth0        initial guess of the translation angular velocity
 * @param[in] wphi0       initial guess of phase roation velocity
 * @param[in] MaxN        maximal iteration number
 * @param[in] tol         convergence tolerence
 * @return    [a, wth, wphi, err]
 */
std::tuple<ArrayXd, double, double, double>
CQCGLgeneral::findReq(const ArrayXd &a0, const double wth0, const double wphi0,
		      const int MaxN /* = 100 */, const double tol /* = 1e-14 */,
		      const bool doesUseMyCG /* = true */,
		      const bool doesPrint /* = true */){ 
    const int n = a0.rows();
    assert(n == Ndim);
    
    ArrayXd a = a0;
    double wth = wth0;
    double wphi = wphi0;
    double lam = 1;

    ConjugateGradient<MatrixXd> CG;
    PartialPivLU<MatrixXd> solver; // used in the pre-CG method
    for(size_t i = 0; i < MaxN; i++){
	if (lam > 1e10) break;
	if(doesPrint) printf("\n ********  i = %zd/%d   ******** \n", i, MaxN);	
	ArrayXd F = velocityReq(a, wth, wphi); 
	double err = F.matrix().norm(); 
	if(err < tol){
	    if(doesPrint) printf("stop at norm(F)=%g\n", err);
	    break;
	}

	std::pair<MatrixXd, VectorXd> p = newtonReq(a, wth, wphi); 
	MatrixXd JJ = p.first.transpose() * p.first;
	VectorXd JF = p.first.transpose() * p.second;
	

	for(size_t j = 0; j < 20; j++){ 
	    // printf("inner iteration j = %zd\n", j);
	    //MatrixXd H = JJ + lam * JJ.diagonal().asDiagonal(); 
	    MatrixXd H = JJ;
	    H.diagonal() *= (1+lam);
	    VectorXd dF; 
	    if(doesUseMyCG){
		std::pair<VectorXd, std::vector<double> > cg = iterMethod::ConjGradSSOR<MatrixXd>
		    (H, -JF, solver, VectorXd::Zero(H.rows()), H.rows(), 1e-6);
		dF = cg.first;
		if(doesPrint) printf("CG error %g after %lu iterations.\n", cg.second.back(), cg.second.size());
	    }
	    else{
		CG.compute(H);     
		dF = CG.solve(-JF);
		if(doesPrint) printf("CG error %g, iteration number %d\n", CG.error(), CG.iterations());
	    }
	    ArrayXd aNew = a + dF.head(n).array();
	    double wthNew = wth + dF(n); 
	    double wphiNew = wphi + dF(n+1);
	    // printf("wthNew = %g, wphiNew = %g\n", wthNew, wphiNew);
      
	    ArrayXd FNew = velocityReq(aNew, wthNew, wphiNew);
	    double errNew = FNew.matrix().norm();
	    if(doesPrint) printf("errNew = %g\n", errNew);
	    if (errNew < err){
		a = aNew;
		wth = wthNew;
		wphi = wphiNew;
		lam = lam/10;
		// if(doesPrint) printf("lam = %g\n", lam);
		break;
	    }
	    else {
		lam *= 10;
		// if(doesPrint) printf("lam = %g\n", lam);
		if( lam > 1e10){
		    if(doesPrint) printf("lam = %f too large.\n", lam);
		    break;
		}
	    }
      
	}
    }

    ArrayXd velReq = velocityReq(a, wth, wphi);
    return std::make_tuple(a, wth, wphi, velReq.matrix().norm());
}

/** @brief find the optimal guess of wth and wphi for a candidate req
 * 
 *  When we find a the inital state of a candidate of req, we also need
 *  to know the appropriate th and phi to start the Newton search.
 *  According to the definition of velocityReq(), this is a residual
 *  minimization problem.
 *
 *  @return    [wth, wphi, err] such that velocityReq(a0, wth, wphi) minimal
 */
std::vector<double>
CQCGLgeneral::optThPhi(const ArrayXd &a0){ 
    VectorXd t1 = transTangent(a0);
    VectorXd t2 = phaseTangent(a0);
    double c = t2.dot(t1) / t1.dot(t1);
    VectorXd t3 = t2 - c * t1;

    VectorXd v = velocity(a0);
    double a1 = t1.dot(v) / t1.dot(t1);
    double a2 = t3.dot(v) / t3.dot(t3);
    
    double err = (v - a1 * t1 - a2 * t3).norm();
    
    std::vector<double> x = {-(a1-a2*c), -a2, err};
    return x;
}


#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                   Abandoned                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
/**
 * @brief pad the input with zeros
 *
 * For example:
 *     if N = 256
 *     0, 1, 2, ..., 127, -127, ..., -1 => insert between 127 and -127
 *     The left half has one mode more than the second half
 */
ArrayXXd CQCGLgeneral::pad(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();		
    int m = aa.cols();
    assert(Ndim == n);
    ArrayXXd paa(2*N, m);
    paa << aa.topRows(2*Nplus), ArrayXXd::Zero(2*Nalias, m), 
	aa.bottomRows(2*Nminus);
    return paa;
}

/**
 * @brief general padding/squeeze an array (arrays).
 *
 * This function is different from pad() that it only requrie n/2
 * is an odd number. It is used to prepare initial conditions for
 * Fourier modes doubled/halfed system. Alos the padding result does not
 * have dimension 2*N, but Ndim.
 *
 * @note It is user's duty to rescale the values since Fourier mode magnitude
 *       changes after doubling/halfing.
 *       For example,
 *       Initially, a_k = \sum_{i=0}^{N} f(x_n)e^{ikx}
 *       After doubling modes, ap_k is a sum of 2*N terms, and
 *       each adjacent pair is also as twice large as the previous
 *       corresponding one term.
 */
ArrayXXd CQCGLgeneral::generalPadding(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert( n % 4 == 2);
    ArrayXXd paa(Ndim, m);
    if (n < Ndim){
	paa << aa.topRows(n/2 + 1), ArrayXXd::Zero(Ndim - n, m),
	    aa.bottomRows(n/2 - 1);
    }
    else {
	paa << aa.topRows(Ne + 1), aa.bottomRows(Ne - 1);
    }
    return paa;
}

ArrayXXcd CQCGLgeneral::padcp(const Ref<const ArrayXXcd> &x){
    int n = x.rows();
    int m = x.cols();
    assert(Ne == n);
    ArrayXXcd px(N, m);
    px << x.topRows(Nplus), ArrayXXcd::Zero(Nalias, m),
	x.bottomRows(Nminus);

    return px;
}

ArrayXXd CQCGLgeneral::unpad(const Ref<const ArrayXXd> &paa){
    int n = paa.rows();
    int m = paa.cols();
    assert(2*N == n);
    ArrayXXd aa(Ndim, m);
    aa << paa.topRows(2*Nplus), paa.bottomRows(2*Nminus);
    return aa;
}

MatrixXd CQCGLgeneral::rk4(const VectorXd &a0, const double dt, const int nstp, const int nq){
    VectorXd x(a0);
    MatrixXd xx(Ndim, nstp/nq+1);
    xx.col(0) = x;

    for(int i = 0; i < nstp; i++){
	VectorXd k1 = velocity(x);
	VectorXd k2 = velocity(x + dt/2 * k1);
	VectorXd k3 = velocity(x + dt/2 * k2);
	VectorXd k4 = velocity(x + dt * k3);
	x += dt/6 * (k1 + 2*k2 + 2*k3 + k4);

	if((i+1)%nq == 0) xx.col((i+1)/nq) = x;
    }

    return xx;
}

MatrixXd CQCGLgeneral::velJ(const MatrixXd &xj){
    MatrixXd vj(Ndim, Ndim+1);
    vj.col(0) = velocity(xj.col(0));
    vj.middleCols(1, Ndim) = stab(xj.col(0)) * xj.middleCols(1, Ndim);
    
    return vj;
}

std::pair<MatrixXd, MatrixXd>
CQCGLgeneral::rk4j(const VectorXd &a0, const double dt, const int nstp, const int nq, const int nqr){
    MatrixXd x(Ndim, Ndim + 1);
    x << a0, MatrixXd::Identity(Ndim, Ndim);
    
    MatrixXd xx(Ndim, nstp/nq+1);
    xx.col(0) = a0;
    MatrixXd JJ(Ndim, Ndim*(nstp/nqr));
    
    for(int i = 0; i < nstp; i++){
	MatrixXd k1 = velJ(x);
	MatrixXd k2 = velJ(x + dt/2 *k1);
	MatrixXd k3 = velJ(x + dt/2 *k2);
	MatrixXd k4 = velJ(x + dt *k3);

	x += dt/6 * (k1 + 2*k2 + 2*k3 + k4);

	if((i+1)%nq == 0) xx.col((i+1)/nq) = x.col(0);
	if((i+1)%nqr == 0){
	    int k = (i+1)/nqr - 1;
	    JJ.middleCols(k*Ndim, Ndim) = x.middleCols(1, Ndim);
	    x.middleCols(1, Ndim) = MatrixXd::Identity(Ndim, Ndim);
	}
    }

    return std::make_pair(xx, JJ);
}

/* -------------------------------------------------- */
/* --------         Lyapunov functional   ----------- */
/* -------------------------------------------------- */

/**
 * @brief calculate the Lyapunov functional
 *
 *  L = \int dx [ -\mu |A|^2 + D |A_x|^2 - 1/2 \beta |A|^4 - 1/3 \gamma |A|^6 ]
 *  Here FFT[ A_{x} ]_k = iq_k a_k, so 
 *          \int dx |A_x|^2 = 1/N \sum (q_k^2 |a_k|^2)
 *  The rest term can be obtained by invere FFT
 *  =>
 *  L = \sum_{k=1}^N [ -\mu |A_k|^2 - 1/2 \beta |A_k|^4 - 1/3 \gamma |A_k|^6]
 *     +\sum_{k=1}^N 1/N D [ q_k^2 |a_k|^2 ]
 *
 *  Note, there are may be pitfalls, but there is indeed asymmetry here:
 *  \sum_n |A_n|^2 = 1/N \sum_k |a_k|^2
 */
ArrayXcd
CQCGLgeneral::Lyap(const Ref<const ArrayXXd> &aa){
    int M = aa.cols();
    VectorXcd lya(M);
    for (size_t i = 0; i < M; i++){
	ArrayXcd a = R2C(aa.col(i));
	ArrayXcd a2 = a * a.conjugate();
	F[0].v1 = a;
	F[0].ifft();
	ArrayXcd A = F[0].v2;
	ArrayXcd A2 = A * A.conjugate();
	lya(i) =  -Mu * A2.sum()
	    - 1.0/2 * dcp(Br, Bi) * (A2*A2).sum()
	    - 1.0/3 * dcp(Gr, Gi) * (A2*A2*A2).sum()
	    + 1.0/N * dcp(Dr, Di) * (QK.square() * a2).sum();
    }
    
    return lya;
}

/**
 * @brief calculate the time derivative of Lyapunov functional
 *
 * The time derivative of Lyapunov functional is L_t = -2 \int dx |A_t|^2
 * 
 * @see Lyap(), velocity()
 */
ArrayXd
CQCGLgeneral::LyapVel(const Ref<const ArrayXXd> &aa){
    int M = aa.cols();
    VectorXd lyavel(M);
    for (size_t i = 0; i < M; i++){
	ArrayXd vel = velocity(aa.col(i)); // Fourier mode of velocity
	ArrayXcd cvel = R2C(vel);
	Fv.v1 = cvel;
	Fv.ifft();		// Fv.v2 is A_t
	lyavel(i) = -2 * (Fv.v2 * Fv.v2.conjugate()).sum().real();
    }
    return lyavel;
}

#endif 

#endif
