#include "ksint.hpp"
#include "denseRoutines.hpp"
#include "iterMethod.hpp"
#include <cmath>
#include <iostream>

#define PROD(x, y) ((x).matrix().asDiagonal() * (y).matrix())
//#define PROD(x, y) (y.colwise() * x)

using namespace std;
using namespace Eigen;
using namespace MyFFT;
using namespace denseRoutines;
using namespace iterMethod;

/*============================================================
 *                       Class : KS integrator
 *============================================================*/

/*-------------------- constructor, destructor -------------------- */
KS::KS(int N, double d) : 
    N(N), d(d), 
    F{ RFFT(N, 1), RFFT(N, 1), RFFT(N, 1), RFFT(N, 1), RFFT(N, 1)},
    JF{ RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1)}
{
    ksInit();
}

KS::KS(const KS &x) : 
    N(x.N), d(x.d),
    F{ RFFT(N, 1), RFFT(N, 1), RFFT(N, 1), RFFT(N, 1), RFFT(N, 1)},
    JF{ RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1), RFFT(N, N-1)}
{
    ksInit();
}

KS & KS::operator=(const KS &x){
    return *this;
}

KS::~KS(){}

/*------------------- member methods ------------------ */

/**
 * @brief calculate the coefficients of ETDRK4 or Krogstad
 */
void KS::calCoe(const double h){

    ArrayXd hL = h*L;
    ArrayXXcd LR = ZR(hL);
    
    E = hL.exp();
    E2 = (hL/2).exp();

    ArrayXXcd LR2 = LR.square();
    ArrayXXcd LR3 = LR.cube();
    ArrayXXcd LRe = LR.exp();
    ArrayXXcd LReh = (LR/2).exp();
    
    a21 = h * ( (LReh - 1)/LR ).rowwise().mean().real(); 
    b1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean().real();
    b2 = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean().real();
    b4 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean().real();

    if (Method == 2) {
	a31 = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean().real();
	a32 = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean().real();
	a41 = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean().real();
	a43 = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean().real();
    }
 
}


/**
 * @brief calcuate the matrix to do averge of phi(z). 
 */

ArrayXXcd KS::ZR(ArrayXd &z){
    
    int M1 = z.size();
    ArrayXd K = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 

    ArrayXXcd r = R * ((K-0.5)/M * dcp(0, M_PI)).exp().transpose();

    return z.template cast<std::complex<double>>().replicate(1, M) + r.replicate(M1, 1);
}


/* calculate the linear part and coefficient of nonlinear part */
void KS::ksInit(){
    K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d; //2*PI/d*[0, 1, 2,...,N/2]
    K(N/2) = 0;
    L = K*K - K*K*K*K; 
    G = 0.5 * dcp(0,1) * K * N;   
    jG = ArrayXXcd::Zero(G.rows(), N-1); 
    jG << G, 2.0*G.replicate(1, N-2); 
}

/**
 * @brief one step integrating the orbit
 *
 * Local truncation error estimation is using Frobenius norm. If we are only integrating
 * an orbit, it is reduced to L2 norm.
 */
void 
KS::oneStep(double &du, const bool onlyOrbit){
    RFFT *f = F;
    if (!onlyOrbit)  f = JF;

    if (1 == Method) {
	NL(0, onlyOrbit);
	
	f[1].vc1 = PROD(E2, f[0].vc1) + PROD(a21, f[0].vc3); 
	NL(1, onlyOrbit);

	f[2].vc1 = PROD(E2, f[0].vc1) + PROD(a21, f[1].vc3);
	NL(2, onlyOrbit);
	
	f[3].vc1 = PROD(E2, f[1].vc1) + PROD(a21, 2*f[2].vc3 - f[0].vc3);
	NL(3, onlyOrbit);

	f[4].vc1 = PROD(E, f[0].vc1) + PROD(b1, f[0].vc3) + PROD(b2, f[1].vc3+f[2].vc3) + PROD(b4, f[3].vc3);
	NL(4, onlyOrbit);

	du = PROD(b4, f[4].vc3-f[3].vc3).norm() / f[4].vc1.matrix().norm(); 
    }
    else {
	NL(0, onlyOrbit);

	f[1].vc1 = PROD(E2, f[0].vc1) + PROD(a21, f[0].vc3); 
	NL(1, onlyOrbit);

	f[2].vc1 = PROD(E2, f[0].vc1) + PROD(a31, f[0].vc3) + PROD(a32, f[1].vc3);
	NL(2, onlyOrbit);

	f[3].vc1 = PROD(E, f[0].vc1) + PROD(a41, f[0].vc3) + PROD(a43, f[2].vc3);
	NL(3, onlyOrbit);
	
	f[4].vc1 = PROD(E, f[0].vc1) + PROD(b1, f[0].vc3) + PROD(b2, f[1].vc3+f[2].vc3) + PROD(b4, f[3].vc3);
	NL(4, onlyOrbit);

	du = PROD(b4, f[4].vc3-f[3].vc3).norm() / f[4].vc1.matrix().norm();
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
double
KS::adaptTs(bool &doChange, bool &doAccept, const double s){
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



ArrayXXd 
KS::intg(const ArrayXd &a0, const double h, const int Nt, const int skip_rate){
    
    assert(a0.size() == N-2);
    return constETD(a0, h, Nt, skip_rate, true);
}

std::pair<ArrayXXd, ArrayXXd>
KS::intgj(const ArrayXd &a0, const double h, const int Nt, const int skip_rate){
    
    assert(a0.size() == N-2);
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    ArrayXXd aa = constETD(v0, h, Nt, skip_rate, false);
    
    int m = aa.cols() / (N-1);
    ArrayXXd x(N-2, m);
    ArrayXXd xx(N-2, m*(N-2));
    for(int i = 0; i < m; i++){
	x.col(i) = aa.col(i*(N-1));
	xx.middleCols((N-2)*i, N-2) = aa.middleCols((N-1)*i+1, N-2);
    }

    return std::make_pair(x, xx);
}


std::pair<VectorXd, ArrayXXd>
KS::aintg(const ArrayXd &a0, const double h, const double tend, 
	  const int skip_rate){
    
    assert(a0.size() == N-2);
    return adaptETD(a0, h, tend, skip_rate, true);
}

std::tuple<VectorXd, ArrayXXd, ArrayXXd>
KS::aintgj(const ArrayXd &a0, const double h, const double tend, 
	   const int skip_rate){
    
    assert(a0.size() == N-2);
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    auto tmp = adaptETD(v0, h, tend, skip_rate, false);
    ArrayXXd &aa = tmp.second;

    int m = aa.cols() / (N-1);
    ArrayXXd x(N-2, m);
    ArrayXXd xx(N-2, m*(N-2));
    for(int i = 0; i < m; i++){
	x.col(i) = aa.col(i*(N-1));
	xx.middleCols((N-2)*i, N-2) = aa.middleCols((N-1)*i+1, N-2);
    }

    return std::make_tuple(tmp.first, x, xx);
}


/**
 * @brief Constant time step integrator
 */
ArrayXXd 
KS::constETD(const ArrayXXd a0, const double h, const int Nt, 
	     const int skip_rate, const bool onlyOrbit){

    int nc = 1;			// number of columns of a single state
    RFFT *f = F;
    if (!onlyOrbit)  {
	f = JF;
	nc = N-1;
    }
    
    const int M = Nt / skip_rate + 1;
    f[0].vc1 = R2C(a0);
    ArrayXXd aa(N-2, M*nc);
    aa.leftCols(nc) = a0;
    lte.resize(M-1);
    NCallF = 0;

    calCoe(h);

    double du;
    int num = 0;
    for(int i = 0; i < Nt; i++){
	oneStep(du, onlyOrbit);
	f[0].vc1 = f[4].vc1;	// update state
	NCallF += 5;
	if ( (i+1)%skip_rate == 0 ) {
	    aa.middleCols((num+1)*nc, nc) = C2R(f[4].vc1);
	    lte(num++) = du;  
	}
    }

    return aa;
}

/**
 * @brief time step adaptive integrator
 */
std::pair<VectorXd, ArrayXXd>
KS::adaptETD(const ArrayXXd &a0, const double h0, const double tend, 
	     const int skip_rate, const bool onlyOrbit){
    
    int nc = 1;			// number of columns of a single state
    RFFT *f = F;
    if (!onlyOrbit)  {
	f = JF;
	nc = N-1;
    }
    
    double h = h0; 
    calCoe(h);

    const int Nt = (int)round(tend/h);
    const int M = Nt /skip_rate + 1;
    f[0].vc1 = R2C(a0);

    ArrayXXd aa(N-2, M*nc);
    VectorXd tt(M);
    aa.leftCols(nc) = a0;
    tt(0) = 0;
    NCalCoe = 0;
    NReject = 0;
    NCallF = 0;    
    hs.resize(M-1);
    lte.resize(M-1);

    double t = 0;
    double du = 0;
    int num = 1;
    bool doChange, doAccept;

    int i = 0;
    while(t < tend){ 
	i++;

	if ( t + h > tend){
	    h = tend - t;
	    calCoe(h);
	    NCalCoe++;
	}

	oneStep(du, onlyOrbit);
	NCallF += 5;		
	double s = nu * std::pow(rtol/du, 1.0/4);
	double mu = adaptTs(doChange, doAccept, s);
	
	if (doAccept){
	    t += h;
	    f[0].vc1 = f[4].vc1;
	    if ( (i+1) % skip_rate == 0 ) {
		if (num >= tt.size() ) {
		    int m = tt.size();
		    tt.conservativeResize(m+cellSize);
		    aa.conservativeResize(Eigen::NoChange, (m+cellSize)*nc); // rows not change, just extend cols
		    hs.conservativeResize(m-1+cellSize);
		    lte.conservativeResize(m-1+cellSize);
		}
		hs(num-1) = h;
		lte(num-1) = du;
		aa.middleCols(num*nc, nc) = C2R(f[4].vc1);
		tt(num) = t;
		num++;
	    }
	}
	else {
	    NReject++;
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
    return std::make_pair(tt.head(num), aa.leftCols(num*nc));
}

#if 0
/** @brief intg() integrate KS system without calculating Jacobian
 *
 *  @param[in] a0 Initial condition of the orbit
 *  @param[in] nstp Number of steps to integrate
 *  @return the orbit, each column is one point
 */
ArrayXXd KS::intg(const ArrayXd &a0, size_t nstp, size_t np){
    if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);}  
    Fv.vc1 = R2C(a0);  
    ArrayXXd aa(N-2, nstp/np + 1);
    aa.col(0) = a0;

    for(size_t i = 1; i < nstp +1; i++){
	NL(Fv); 
	Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3; 

	NL(Fa); 
	Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;

	NL(Fb); 
	Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);

	NL(Fc);

	Fv.vc1 = E*Fv.vc1 + Fv.vc3*f1 + 2.0*(Fa.vc3+Fb.vc3)*f2 + Fc.vc3*f3;
    
	if( 0 == i%np ) aa.col(i/np) = C2R(Fv.vc1);
    }
  
    return aa;
}


/** @brief intg() integrate KS system without calculating Jacobian
 *
 *  @param[in] a0 Initial condition of the orbit
 *  @param[in] nstp Number of steps to integrate
 *  @return the orbit, each column is one point
 */
std::tuple<ArrayXXd, VectorXd, VectorXd> 
KS::intgDP(const ArrayXd &a0, size_t nstp, size_t np){
    if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);}  
    Fv.vc1 = R2C(a0);  
    ArrayXXd aa(N-2, nstp/np + 1);
    aa.col(0) = a0;
  
    /* define the total dissipation and pumping */
    VectorXd DD(nstp/np + 1), PP(nstp/np + 1);
    double D, P, D1, D2, D3, D4, P1, P2, P3, P4;
    D = 0; P = 0;
    DD(0) = D; PP(0) = P;


    for(size_t i = 1; i < nstp +1; i++){
	D1 = disspation(Fv.vc1); P1 = pump(Fv.vc1);
	NL(Fv); Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3; 

	D2 = disspation(Fa.vc1); P2 = pump(Fa.vc1);
	NL(Fa); Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;

	D3 = disspation(Fb.vc1); P3 = pump(Fb.vc1);
	NL(Fb); Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);

	D4 = disspation(Fc.vc1); P4 = pump(Fc.vc1);
	NL(Fc);

	Fv.vc1 = E*Fv.vc1 + Fv.vc3*f1 + 2.0*(Fa.vc3+Fb.vc3)*f2 + Fc.vc3*f3;
	D = D + h/6 * (D1 + 2*D2 + 2*D3 + D4); 
	P = P + h/6 * (P1 + 2*P2 + 2*P3 + P4);
    
	if( 0 == i%np ) {
	    aa.col(i/np) = C2R(Fv.vc1);
	    DD(i/np) = D;
	    PP(i/np) = P;
	}
    }
  
    return std::make_tuple(aa, DD, PP);
}


/** @brief intgj() integrate KS system and calculate Jacobian along this orbit.
 *
 * @param[in] a0 Initial condition of the orbit. Size [N-2,1]
 * @param[in] nstp Number of steps to integrate.
 * @return Pair value. The first element is the trajectory, dimension [N, nstp/np+1]
 *         The second element is the Jacobian along the trajectory,
 *         dimension [N*N, nstp/nqr].
 */
std::pair<ArrayXXd, ArrayXXd>
KS::intgj(const ArrayXd &a0, size_t nstp, size_t np, size_t nqr){
    assert( N-2 == a0.rows() );
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    jFv.vc1 = R2C(v0);
    ArrayXXd aa(N-2, nstp/np+1); aa.col(0) = a0;
    ArrayXXd daa((N-2)*(N-2), nstp/nqr);
    for(size_t i = 1; i < nstp + 1; i++){ // diagonal trick 
	jNL(jFv); 
	jFa.vc1 = E2.matrix().asDiagonal()*jFv.vc1.matrix() + Q.matrix().asDiagonal()*jFv.vc3.matrix();

	jNL(jFa); 
	jFb.vc1 = E2.matrix().asDiagonal()*jFv.vc1.matrix() + Q.matrix().asDiagonal()*jFa.vc3.matrix();

	jNL(jFb); 
	jFc.vc1 = E2.matrix().asDiagonal()*jFa.vc1.matrix() + Q.matrix().asDiagonal()*(2.0*jFb.vc3 - jFv.vc3).matrix();

	jNL(jFc);

	jFv.vc1 = E.matrix().asDiagonal()*jFv.vc1.matrix() 
	    + f1.matrix().asDiagonal()*jFv.vc3.matrix() 
	    + (2.0*f2).matrix().asDiagonal()*(jFa.vc3+jFb.vc3).matrix()
	    + f3.matrix().asDiagonal()*jFc.vc3.matrix();
    
	if ( 0 == i%np ) aa.col(i/np) = C2R(jFv.vc1.col(0));
	if ( 0 == i%nqr){
	    ArrayXXd tmp = C2R(jFv.vc1.rightCols(N-2)); tmp.resize((N-2)*(N-2), 1);
	    daa.col(i/nqr-1) = tmp;
	    jFv.vc1.rightCols(N-2) = R2C(MatrixXd::Identity(N-2, N-2));
	}
    }
  
    return std::make_pair(aa, daa);
}

#endif

void KS::NL(const int k, const bool onlyOrbit){
    if(onlyOrbit){
	F[k].ifft();
	F[k].vr2 = F[k].vr2 * F[k].vr2;
	F[k].fft();
	F[k].vc3 *= G;
    }
    else {
	JF[k].ifft(); 
	ArrayXd tmp = JF[k].vr2.col(0);	// in case of aliasing
	JF[k].vr2 = JF[k].vr2.colwise() * tmp;
	JF[k].fft();
	JF[k].vc3 *= jG;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/* @brief complex matrix to the corresponding real matrix.
 * [N/2+1, M] --> [N-2, M]
 * Since the Map is not address continous, the performance is
 * not good enough.
 */
ArrayXXd KS::C2R(const ArrayXXcd &v){
    int n = v.rows();
    int m = v.cols();
    ArrayXXcd vt = v.middleRows(1, n-2);
    ArrayXXd vp(2*(n-2), m);
    vp = Map<ArrayXXd>((double*)&vt(0,0), 2*(n-2), m);

    return vp;
}

ArrayXXcd KS::R2C(const ArrayXXd &v){
    int n = v.rows();
    int m = v.cols();
    assert( 0 == n%2);
    
    ArrayXXcd vp = ArrayXXcd::Zero(n/2+2, m);
    vp.middleRows(1, n/2) = Map<ArrayXXcd>((dcp*)&v(0,0), n/2, m);

    return vp;
}

/*************************************************** 
 *           stability ralated                     *
 ***************************************************/

/** @brief calculate the velocity 
 *
 * @param[in] a0 state vector
 * @return velocity field at a0
 */
VectorXd 
KS::velocity(const Ref<const ArrayXd> &a0){
    assert(a0.rows() == N-2);
    F[0].vc1 = R2C(a0);
    NL(0, true); 
    F[0].vc1 = L * F[0].vc1 + F[0].vc3;
  
    return C2R(F[0].vc1);
}

/* return v(x) + theta *t(x) */
VectorXd
KS::velg(const Ref<const VectorXd> &a0, const double theta){
    return velocity(a0) + theta * gTangent(a0);
}

/** @brief calculate the stability matrix 
 *    A = (qk^2 - qk^4) * v  - i*qk* F( F^{-1}(a0) * F^{-1}(v))
 */
MatrixXd KS::stab(const Ref<const ArrayXd> &a0){
    assert( N-2 == a0.rows() );
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    JF[0].vc1 = R2C(v0);
    NL(0, false);
    JF[0].vc1 = PROD(L, JF[0].vc1).array() + JF[0].vc3;
    
    return C2R(JF[0].vc1.rightCols(N-2));
}

/* the stability matrix of a relative equilibrium */
MatrixXd KS::stabReq(const Ref<const VectorXd> &a0, const double theta){
    return stab(a0) + theta * gGenerator();
}

/* Eigenvalues/Eigenvectors of equilibrium */
std::pair<VectorXcd, MatrixXcd>
KS::stabEig(const Ref<const VectorXd> &a0){
    MatrixXd A = stab(a0);
    return denseRoutines::evEig(A);
}

/* Eigenvalues/Eigenvectors of equilibrium */
std::pair<VectorXcd, MatrixXcd>
KS::stabReqEig(const Ref<const VectorXd> &a0, const double theta){
    MatrixXd A = stabReq(a0, theta);
    return denseRoutines::evEig(A);
}

/*************************************************** 
 *           energe ralated                        *
 ***************************************************/
double KS::pump(const ArrayXcd &vc){
    VectorXcd tmp = vc * K;
    return tmp.squaredNorm();
}

double KS::disspation(const ArrayXcd &vc){
    VectorXcd tmp = vc * K * K;
    return tmp.squaredNorm();
}

/*************************************************** 
 *           Multishooting related                 *
 ***************************************************/

/**
 * @brief calculate the multishooting orbit and Jacobian from
 *        several ponts along the orbit.
 * @param[in]  aa0 a set of points dimension [N, M]
 * @param[in]  nstp integration step for each short segment
 * @return     [orbit, Jacobian] pair, dimension [N, M*nstp/np+1]
 *             and [N*N, M*nstp/nqr] respectively
 */
std::pair<ArrayXXd, ArrayXXd>
KS::intgjMulti(const MatrixXd aa0, size_t nstp, size_t np, size_t nqr){
    // nq and nqr should divide nstp
    assert (nstp % np == 0 && nstp % nqr == 0);
    int M = aa0.cols();
    int N = aa0.rows(); 
    
    // set aa the dim = nstp/np + 1 to make it consistent with
    // the original integrator
    MatrixXd aa(N, M*nstp/np+1);
    MatrixXd daa(N*N, M*nstp/nqr);
    for (int i = 0; i < M; i++) {
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(aa0.col(i), nstp, np, nqr);
	aa.middleCols(i*nstp/np, nstp/np) = tmp.first.leftCols(nstp/np);
	daa.middleCols(i*nstp/nqr, nstp/nqr) = tmp.second;
	// the last color included
	if(i == M-1) aa.rightCols(1) = tmp.first.rightCols(1);
    }
    
    return std::make_pair(aa, daa);
}


/* try to form the Jacobian for finding the relative equilibirum
 *
 *      | A+wT , tx|
 *  J = | tx   ,  0|
 *
 *  @param[in] x  [a0, omega]
 *  return J^T*J, diag(J^T*J), J^T * F
 */
std::tuple<MatrixXd, MatrixXd, VectorXd>
KS::calReqJJF(const Ref<const VectorXd> &x){
    assert(x.size() == N-1);
    double omega = x(N-2); 

    MatrixXd A = stabReq(x.head(N-2), omega); 
    VectorXd tx = gTangent(x.head(N-2)); 
    
    MatrixXd J(N-1, N-1);
    J << 
	A, tx, 
	tx.transpose(), 0; 

    VectorXd F(N-1);
    F << velg(x.head(N-2), omega), 0 ;

    MatrixXd JJ = J.transpose() * J; 
    MatrixXd DJJ = JJ.diagonal().asDiagonal(); 
    VectorXd JF = J.transpose() * F; 
    
    return std::make_tuple(JJ, DJJ, JF); 
}
/**
 * @see calReqJJF
 */
std::tuple<MatrixXd, MatrixXd, VectorXd>
KS::calEqJJF(const Ref<const VectorXd> &x){
    assert(x.size() == N-2);
    
    MatrixXd J = stab(x);
    VectorXd F = velocity(x);

    MatrixXd JJ  = J.transpose() * J;
    MatrixXd DJJ = JJ.diagonal().asDiagonal(); 
    VectorXd JF = J.transpose() * F; 
    
    return std::make_tuple(JJ, DJJ, JF);
}

/* find reqs in KS  */
std::tuple<VectorXd, double, double>
KS::findReq(const Ref<const VectorXd> &x, const double tol, 
	    const int maxit, const int innerMaxit){

    auto fx = [&](const VectorXd &x){
	VectorXd F(N-1);
	F << velg(x.head(N-2), x(N-2)), 0; 
	return F;
    };
    
    KSReqJJF<MatrixXd> jj(*this);    
    ColPivHouseholderQR<MatrixXd> solver; 

    auto result = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    if(std::get<2>(result) != 0) fprintf(stderr, "Req not converged ! \n");
    
    VectorXd at = std::get<0>(result);
    return std::make_tuple(at.head(N-2), at(N-2) , std::get<1>(result).back() );
}

/* find eq in KS */
std::pair<VectorXd, double>
KS::findEq(const Ref<const VectorXd> &x, const double tol,
	   const int maxit, const int innerMaxit){
    
    auto fx = [&](const VectorXd &x){
	return velocity(x);
    };
    
    KSEqJJF<MatrixXd> jj(*this);
    ColPivHouseholderQR<MatrixXd> solver;
    
    auto result = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);
    if(std::get<2>(result) != 0) fprintf(stderr, "Req not converged ! \n");
    
    return std::make_pair(std::get<0>(result), std::get<1>(result).back() );   
}

/*************************************************** 
 *           Symmetry related                      *
 ***************************************************/

/** @brief apply reflection on each column of input  */
ArrayXXd KS::Reflection(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2 );
  
    ArrayXd R(n);
    for(size_t i = 0; i < n/2; i++) {
	R(2*i) = -1;
	R(2*i+1) = 1;
    }
  
    ArrayXXd Raa = aa.colwise() * R;
    return Raa;
}


ArrayXXd KS::half2whole(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
  
    ArrayXXd raa = Reflection(aa);
    ArrayXXd aaWhole(n, 2*m);
    aaWhole << aa, raa;
  
    return aaWhole;
}

/** @brief apply rotation to each column of input with the
 *    same angle specified
 *
 */
ArrayXXd KS::Rotation(const Ref<const ArrayXXd> &aa, const double th){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2);
    ArrayXXd raa(n, m);
  
    for(size_t i = 0; i < n/2; i++){
	Matrix2d R;
	double c = cos(th*(i+1));
	double s = sin(th*(i+1));
	R << 
	    c, -s,
	    s,  c;  // SO(2) matrix

	raa.middleRows(2*i, 2) = R * aa.middleRows(2*i, 2).matrix();
    }
  
    return raa;
}

/** @brief group tangent of SO(2)
 *
 *  x=(b1, c1, b2, c2, ...) ==> tx=(-c1, b1, -2c2, 2b2, ...)
 */
MatrixXd KS::gTangent(const MatrixXd &x){
    int n = x.rows();
    int m = x.cols();
    assert( 0 == n%2);
    MatrixXd tx(n, m);
    for(int i = 0; i < n/2; i++){
	tx.row(2*i) = -(i+1) * x.row(2*i+1).array();
	tx.row(2*i+1) = (i+1) * x.row(2*i).array(); 
    }

    return tx;
}

/* group generator matrix T */
MatrixXd KS::gGenerator(){
    MatrixXd T(MatrixXd::Zero(N-2, N-2));
    for (int i = 0; i < N/2-1; i++ ){
	T(2*i, 2*i+1) = -(i+1);
	T(2*i+1, 2*i) = i+1;
    }

    return T;
}

/** @brief rotate the KS trajectory to the 1st mode slice.
 *
 * @return the angle is the transform angle from slice to orginal point.
 */
std::pair<MatrixXd, VectorXd> KS::orbitToSlice(const Ref<const MatrixXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2);
    MatrixXd raa(n, m);
    VectorXd ang(m);

    for(size_t i = 0; i < m; i++){
	double th = atan2(aa(1,i), aa(0,i));
	ang(i) = th;
	raa.col(i) = Rotation(aa.col(i), -th);
    }
    return std::make_pair(raa, ang);
}

/** @brief project covariant vectors to 1st mode slice
 *
 * projection matrix is h= (I - |tx><tp|/<tx|tp>) * g(-th), so eigenvector |ve> is
 * projected to |ve> - |tx>*(<tp|ve>|/<tx|tp>), before which, g(-th) is 
 * performed.
 *
 * In 1st mode slice, template point is |xp>=(1,0,0,...,0) ==> |tp>=(0,1,0,0,...,0)
 * <tp|ve> = ve.row(1) // second row
 * group tangent is |tx>=(-c1, b1, -2c2, 2b2,...) ==> <tx|tp> = b1
 *
 * @note vectors are not normalized
 */
MatrixXd KS::veToSlice(const MatrixXd &ve, const Ref<const VectorXd> &x){
    std::pair<MatrixXd, VectorXd> tmp = orbitToSlice(x);
    MatrixXd &xhat =  tmp.first; // dimension [N, 1]
    double th = tmp.second(0);
    VectorXd tx = gTangent(xhat);

    MatrixXd vep = Rotation(ve, -th);
    MatrixXd dot = vep.row(1)/xhat(0); //dimension [1, N]
    vep = vep - tx * dot;
  
    return vep;
}

/** @beief project the sequence of Floquet vectors to 1st mode slice
 *
 *  Usaully, aa has one more column the the Floquet vectors, so you can
 *  call this function like:
 *  \code
 *      veToSliceAll(eigVecs, aa.leftCols(aa.cols()-1))
 *  \endcode
 *  
 *  @param[in] eigVecs Floquet vectors along the orbit. Dimension: [N*Trunc, M]
 *  @param[in] aa the orbit
 *  @return projected vectors on the slice with dimension [N, M*Trunc]
 *
 *  @note vectors are not normalized
 */
MatrixXd KS::veToSliceAll(const MatrixXd &eigVecs, const MatrixXd &aa,
			  const int trunc /* = 0*/){
    int Trunc = trunc;
    if(trunc == 0) Trunc = sqrt(eigVecs.rows());

    assert(eigVecs.rows() % Trunc == 0);
    const int n = eigVecs.rows() / Trunc ;  
    const int m = eigVecs.cols();
    const int n2 = aa.rows();
    const int m2 = aa.cols();

    assert(m == m2 && n == n2);
    MatrixXd newVe(n, Trunc*m);
    for(size_t i = 0; i < m; i++){
	MatrixXd ve = eigVecs.col(i);
	ve.resize(n, Trunc);
	newVe.middleCols(i*Trunc, Trunc) = veToSlice(ve, aa.col(i));
    }

    return newVe;
}


/**
 * @brief get the full orbit and full set of Fvs
 * 
 * Given the inital point and the set of Fvs, the whole set is
 * twice bigger for ppo, but for rpo, it is the single piece.
 *
 * @param[in] a0       the inital condition of the ppo/rpo
 * @param[in] ve       the set of Fvs. Dimension [(N-2)*NFV, M]
 * @param[in] nstp     number of integration steps
 * @param[in] ppTpe    ppo/rpo
 * @return             the full orbit and full set of Fvs.
 */
std::pair<ArrayXXd, ArrayXXd>
KS::orbitAndFvWhole(const ArrayXd &a0, const ArrayXXd &ve,
		    const double h,
		    const size_t nstp, const std::string ppType
		    ){
    assert(N-2 == a0.rows());
    const int M = ve.cols();
    assert(nstp % M == 0);
    const int space = nstp / M;
    
    ArrayXXd aa = intg(a0, h, nstp, space);
    if(ppType.compare("ppo") == 0) 
	return std::make_pair(
			      half2whole(aa.leftCols(aa.cols()-1)),
			      half2whole(ve) // this is correct. Think carefully.
			      );
    
    else 
	return std::make_pair(
			      aa.leftCols(aa.cols()-1), // one less
			      ve
			      );
}

/**
 * @brief get rid of the marginal direction of the Fvs
 * 
 * @param[in] ve        dimension [N, M*trunc]
 * @param[in] pos       the location of the group tangent margianl Floquet vector.
 *                      pos starts from 0.
 * @param[in] return    the clean set of Fvs
 */
MatrixXd KS::veTrunc(const MatrixXd ve, const int pos, const int trunc /* = 0 */){
    int Trunc = trunc;
    if(trunc == 0) Trunc = ve.rows();

    const int N = ve.rows();
    const int M = ve.cols() / Trunc;
    assert(ve.cols()%M == 0);
  
    MatrixXd newVe(N, (Trunc-1)*M);
    for(size_t i = 0; i < M; i++){
	newVe.middleCols(i*(Trunc-1), pos) = ve.middleCols(i*Trunc, pos);
	newVe.middleCols(i*(Trunc-1)+pos, Trunc-1-pos) = 
	    ve.middleCols(i*Trunc+pos+1, Trunc-1-pos);
    }
    return newVe;
}


/**
 * @brief get the full orbit and full set of Fvs on the slice
 *
 * @return             the full orbit and full set of Fvs on slice
 * @see                orbitAndFvWhole(),  veTrunc()
 */
std::pair<ArrayXXd, ArrayXXd>
KS::orbitAndFvWholeSlice(const ArrayXd &a0, const ArrayXXd &ve,
			 const double h,
			 const size_t nstp, const std::string ppType,
			 const int pos
			 ){
    assert(ve.rows() % (N-2) == 0);
    const int NFV = ve.rows() / (N-2);
    auto tmp = orbitAndFvWhole(a0, ve, h, nstp, ppType);
    auto tmp2 = orbitToSlice(tmp.first);
    MatrixXd veSlice = veToSliceAll(tmp.second, tmp.first, NFV);
    MatrixXd ve_trunc = veTrunc(veSlice, pos, NFV);

    return std::make_pair(tmp2.first, ve_trunc);
}


/**
 * @brief calculate the changed indices under reflection symmetry
 *
 *        1, 2, 5, 6, 9, 10, ....
 */
std::vector<int> KS::reflectIndex(){
    int n = N - 2;
    std::vector<int> index;
    index.push_back(2);
    for(int i = 5; i < n; i+=4){
	index.push_back(i);
	if(i+1 < n) index.push_back(i+1); /* be carefule here */
    }
    return index;
}

/**
 * @brief reduce the reflection symmetry in the 1st mode slice
 *
 *  p2 = ()\sqrt{b_2^2 + c_3^2}
 *  p3 = b_2 \cdot c_3 / \sqrt{b_2^2 + c_3^2}
 *  p4 = ...
 *  
 * @param[in] aaHat        states in the 1st mode slice
 * @return                 reflection symmetry reduced states
 */
ArrayXXd KS::reduceReflection(const Ref<const ArrayXXd> &aaHat){
    int n = aaHat.rows();
    int m = aaHat.cols();
    assert( n == N-2 );
    MatrixXd aaTilde(aaHat);

    std::vector<int> index = reflectIndex();
    //for(auto it : index) cout << it << endl;
    
    // p2
    ArrayXd x = aaHat.row(2).square();
    ArrayXd y = aaHat.row(5).square();
    aaTilde.row(2) = (x - y) / (x + y).sqrt();
    // p3, p4, p5, ...
    for(int i = 1; i < index.size(); i++){
	ArrayXd x = aaHat.row(index[i-1]);
	ArrayXd y = aaHat.row(index[i]);
	aaTilde.row(index[i]) = x * y / (x.square() + y.square()).sqrt();
    }

    return aaTilde;
}


/**
 * @brief transform covariant vectors (stability vector or Floquet vector)
 *        into the reflection reduced
 *
 *        Denote the reflectiong transform as y = h(x), then according to
 *        ChaosBook, Jacobian is transformed as J' = \Gamma J \Gamma^{-1}
 *        Here \Gamma = \partial h(x) / \partial x : the Jacobian of h(x).
 *        So, covariant vectors are transformed by \Gamma
 */
MatrixXd KS::GammaMat(const Ref<const ArrayXd> &xHat){
    int n = xHat.rows();
    assert( n == N-2);
    
    MatrixXd Gamma(MatrixXd::Identity(n, n));
    std::vector<int> index = reflectIndex();
    double denom = sqrt(xHat(2)*xHat(2) + xHat(5)*xHat(5));
    double denom3 = denom * denom * denom;
    Gamma(2, 2) = xHat(2) * (xHat(2)*xHat(2) + 3*xHat(5)*xHat(5)) / denom3;
    Gamma(2, 5) = -xHat(5) * (xHat(5)*xHat(5) + 3*xHat(2)*xHat(2)) / denom3;

    for(int i = 1; i < index.size(); i++){
	double denom = sqrt( xHat(index[i])*xHat(index[i]) +
			     xHat(index[i-1])*xHat(index[i-1]) );
	double denom3 = denom * denom *denom;
	
	Gamma(index[i], index[i]) = xHat(index[i-1]) / denom -
	    xHat(index[i-1]) * xHat(index[i]) * xHat(index[i]) / denom3;
	Gamma(index[i], index[i-1]) = xHat(index[i]) / denom -
	    xHat(index[i]) * xHat(index[i-1]) * xHat(index[i-1]) / denom3;
    }

    return Gamma;
} 

MatrixXd KS::reflectVe(const MatrixXd &ve, const Ref<const ArrayXd> &xHat){
    MatrixXd Gamma = GammaMat(xHat);
    return Gamma * ve;
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
 *  @param[in] aaHat   the orbit in the 1st mode slice
 *  @param[in] trunc   the number of vectors at each orbit point.
 *                     trunc = 0 means full set of vectors
 *  @return            transformed to the reflection invariant space.
 *                     Dimension [N, M*Trunc]
 *
 *  @note vectors are not normalized
 */
MatrixXd KS::reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
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


////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<MatrixXd, VectorXd>
KS::redSO2(const Ref<const MatrixXd> &aa){
    int p = 2;			// index of Fourier mode

    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2);
    MatrixXd raa(n, m);
    VectorXd ang(m);

    for(size_t i = 0; i < m; i++){
	double th = (atan2(aa(2*p-1, i), aa(2*p-2, i)) - M_PI/2 )/ p;
	ang(i) = th;
	raa.col(i) = Rotation(aa.col(i), -th);
    }
    return std::make_pair(raa, ang);
}

MatrixXd KS::redR1(const Ref<const MatrixXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2);
    
    int p = 2;
    
    // step 1
    MatrixXcd F = a2f(aa);
    MatrixXcd bb(n/2, m);
    bb.row(0) = F.row(0).array().pow(p);
    for(int i = 1; i < n/2; i++){
	int k = (n/2 * p - i - 1) % p;
	bb.row(i) = F.row(i).array() * F.row(0).array().pow(k);
    }
    MatrixXd cc = f2a(bb);
    
    return cc;
}

MatrixXd KS::redR2(const Ref<const MatrixXd> &cc){
    int n = cc.rows();
    int m = cc.cols();
    assert( 0 == n%2);

    // step 2
    MatrixXd raa(cc);
    raa.row(1) = cc.row(1).array().square();
    for(int i = 1; i < n/2; i++){
	if(i % 2 == 0 )
	    raa.row(2*i + 1) = cc.row(2*i + 1).array() * cc.row(1).array();
	else 
	    raa.row(2*i) = cc.row(2*i).array() * cc.row(1).array();
    }
    
    return raa;
}

MatrixXd KS::redRef(const Ref<const MatrixXd> &aa){
    return redR2(redR1(aa));
}

std::pair<MatrixXd, VectorXd>
KS::redO2(const Ref<const MatrixXd> &aa){
    auto tmp = redSO2(aa);
    return std::make_pair(redRef(tmp.first), tmp.second);
}


MatrixXd KS::Gmat1(const Ref<const VectorXd> &x){
    int n = x.size();
    assert( n == N-2);
    
    // step 1
    MatrixXd G(MatrixXd::Identity(n, n));
    G(0, 0) = 2*x(0);
    G(1, 1) = 2*x(0);
    G(0, 1) = -2*x(1);
    G(1, 0) = 2*x(1);
    for (int i = 2; i < n/2; i += 2){
	G(2*i, 2*i) = x(0);
	G(2*i+1, 2*i+1) = x(0);
	G(2*i, 2*i+1) = -x(1);
	G(2*i+1, 2*i) = x(1);

	G(2*i, 0) = x(2*i);
	G(2*i, 1) = -x(2*i+1);
	G(2*i+1, 0) = x(2*i+1);
	G(2*i+1, 1) = x(2*i);
    }

    return G;
}
 
MatrixXd KS::Gmat2(const Ref<const VectorXd> &x){
    int n = x.size();
    assert( n == N-2 );

    // step 2 
    MatrixXd G(MatrixXd::Identity(n, n));
    G(1, 1) = 2*x(1);
    for (int i = 1; i < n/2; i++){
	if( i % 2 == 1){
	    G(2*i, 2*i) = x(1);
	    G(2*i, 1) = x(2*i);
	}
	else {
	    G(2*i+1, 2*i+1) = x(1);
	    G(2*i+1, 1) = x(2*i+1); 
	}
    }

    return G;
} 

std::pair<VectorXd, MatrixXd>
KS::redV(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a){
    auto tmp = redSO2(a);
    MatrixXd &aH = tmp.first;
    double th = tmp.second(0);
    VectorXd tx = gTangent(aH);

    int p = 2;

    MatrixXd vep = Rotation(v, -th);
    MatrixXd dot = vep.row(2*p-2) / (-p * aH(2*p-1));  
    vep = vep - tx * dot;

    return std::make_pair(aH, vep);
}

MatrixXd KS::redV2(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a){
    auto tmp = redV(v, a);    
    VectorXd &aH = tmp.first;
    MatrixXd &vp = tmp.second;
    
    VectorXd aH1 = redR1(aH);
    
    MatrixXd G1 = Gmat1(aH);
    MatrixXd G2 = Gmat2(aH1);
    
    return G2*G1*vp;
}

/*************************************************** 
 *                  Others                         *
 ***************************************************/

/* calculate mode modulus */
MatrixXd KS::calMag(const Ref<const MatrixXd> &aa){
    int m = aa.rows();
    int n = aa.cols();
    assert(m % 2 == 0);
    
    MatrixXd r(m/2, n);
    for(int i = 0; i < m/2; i++){
	r.row(i) = (aa.row(2*i).array().square() + 
		    aa.row(2*i+1).array().square()
		    ).sqrt();
    }

    return r;
}

std::pair<MatrixXd, MatrixXd>
KS::toPole(const Ref<const MatrixXd> &aa){
    int m = aa.rows();
    int n = aa.cols();
    assert(m % 2 == 0);
    
    MatrixXd r(m/2, n);
    MatrixXd th(m/2, n);
    for(int i = 0; i < m/2; i++){
	r.row(i) = (aa.row(2*i).array().square() + 
		    aa.row(2*i+1).array().square()
		    ).sqrt();
	for(int j = 0; j < n; j++){
	    th(i, j) = atan2(aa(2*i+1, j), aa(2*i, j)); 
	}

    }

    return std::make_pair(r, th);

}


MatrixXcd
KS::a2f(const Ref<const MatrixXd> &aa){
    int m = aa.rows();
    int n = aa.cols();
    assert(m % 2 == 0);
    
    MatrixXcd F(m/2, n);
    for(int i = 0; i < m/2; i++){
	F.row(i).real() = aa.row(2*i);
	F.row(i).imag() = aa.row(2*i+1);
    }
    
    return F;
}

MatrixXd
KS::f2a(const Ref<const MatrixXcd> &f){
    int m = f.rows();
    int n = f.cols();
    
    MatrixXd a(m*2, n);
    for(int i = 0; i < m; i++){
	a.row(2*i) = f.row(i).real();
	a.row(2*i+1) = f.row(i).imag();
    }
    
    return a;
}

