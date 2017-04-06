#include "CQCGL1dSub.hpp"

using namespace denseRoutines;

// implementation notes
// 1) not use fft.fwd(MatrixBase, MatrixBase) but use
//    fft.fwd(Complex * dst, const Complex * src, Index nfft)
//    becuase the former cannot work with Map<VectorXcd> type.
// 2) In C++, initialization list will be evaluated in the order they were
//    declared in the class definition

//////////////////////////////////////////////////////////////////////
//                     Inner Class CQCGL1dSub                          //
//////////////////////////////////////////////////////////////////////
CQCGL1dSub::NL::NL(){}
CQCGL1dSub::NL::NL(CQCGL1dSub *cgl, int cols) : 
    cgl(cgl), N(cgl->N), Ne(cgl->Ne), B(cgl->Br, cgl->Bi), G(cgl->Gr, cgl->Gi) 
{
    modes.resize(cgl->N, cols);
    field.resize(cgl->N, cols);
}
CQCGL1dSub::NL::~NL(){}

void CQCGL1dSub::NL::operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t){
    int cs = x.cols();
    assert(cs == dxdt.cols() && Ne == x.rows() && Ne == dxdt.rows());
    
    modes << x, ArrayXXcd::Zero(N-2*Ne+1, cs), x.bottomRows(Ne-1).colwise().reverse();    
    for(int i = 0; i < cs; i++)
	cgl->fft.inv(field.data() + i*N, modes.data() + i*N, N);
    ArrayXXcd AA = field.topRows(N/2 + 1); // [A_0, A_1..., A_N/2, A_N/2-1,,. A_1]

    ArrayXcd A = AA.col(0); 
    ArrayXcd aA2 = A * A.conjugate();

    if (cgl->IsQintic)
	AA.col(0) = B * A * aA2 + G * A * aA2.square();
    else 
	AA.col(0) = B * A * aA2;

    if(cs > 1){
	int M = cs - 1;
	ArrayXcd A2 = A.square();
	if (cgl->IsQintic)
	    AA.rightCols(M) = AA.rightCols(M).conjugate().colwise() *  ((B+G*2.0*aA2) * A2) +
		AA.rightCols(M).colwise() * ((2.0*B+3.0*G*aA2)*aA2);
	else
	    AA.rightCols(M) = AA.rightCols(M).conjugate().colwise() *  (B * A2) +
		AA.rightCols(M).colwise() * (2.0*B*aA2);	
    }
    
    field << AA, AA.middleRows(1, N/2-1).colwise().reverse();
    for(int i = 0; i < cs; i++)
	cgl->fft.fwd(modes.data() + i*N, field.data() + i*N, N);
    
    dxdt = modes.topRows(Ne);
}

//////////////////////////////////////////////////////////////////////
//                        Class CQCGL1dSub                             //
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
 * @param[in] dimTan       the dimension of tangent space
 *                         dimTan > 0 => dimTan
 *                         dimTan = 0 => Ndim
 *                         dimTan < 0 => 0
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
 *        Ne                    N - 2*Ne + 1           Ne - 1
 */
CQCGL1dSub::CQCGL1dSub(int N, double d,
		       double Mu, double Dr, double Di, double Br, double Bi,
		       double Gr, double Gi, int dimTan)
    : N(N), d(d), Mu(Mu), Dr(Dr), Di(Di), Br(Br), Bi(Bi), Gr(Gr), Gi(Gi),
      Ne(N/3),	
      Ndim(2 * Ne),
      DimTan(dimTan == 0 ? Ndim : (dimTan > 0 ? dimTan : 0)),
      nl(this, 1), nl2(this, DimTan+1)
{
    // calculate the Linear part
    K = ArrayXd::LinSpaced(Ne, 0, Ne-1);      
    QK = 2*M_PI/d * K; 
    L = dcp(Mu, -Omega) - dcp(Dr, Di) * QK.square(); 

    int nYN0 = eidc.names.at(eidc.scheme).nYN; // do not call setScheme here. Different.
    for(int i = 0; i < nYN0; i++){
	Yv[i].resize(Ne, 1);
	Nv[i].resize(Ne, 1);
	Yv2[i].resize(Ne, DimTan+1);
	Nv2[i].resize(Ne, DimTan+1);
    }
    eidc.init(&L, Yv, Nv);
    eidc2.init(&L, Yv2, Nv2);
}

/**
 * Constructor of cubic quintic complex Ginzburg-Landau equation
 * A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
 */
CQCGL1dSub::CQCGL1dSub(int N, double d, 
		       double b, double c, double dr, double di, 
		       int dimTan)
    : CQCGL1dSub::CQCGL1dSub(N, d, -1, 1, b, 1, c, -dr, -di, dimTan) {}


CQCGL1dSub::CQCGL1dSub(int N, double d,
		       double delta, double beta, double D, double epsilon,
		       double mu, double nu, int dimTan)
    : CQCGL1dSub::CQCGL1dSub(N, d, delta, beta, D/2, epsilon, 1, mu, nu) {}

CQCGL1dSub::~CQCGL1dSub(){}

CQCGL1dSub & CQCGL1dSub::operator=(const CQCGL1dSub &x){
    return *this;
}

/* ------------------------------------------------------ */
/* ----                 Integrator                ------- */
/* ------------------------------------------------------ */

void CQCGL1dSub::setScheme(std::string x){
    int nYN0 = eidc.names.at(eidc.scheme).nYN;
    eidc.scheme = x;
    int nYN1 = eidc.names.at(eidc.scheme).nYN;
    for (int i = nYN0; i < nYN1; i++) {
	Yv[i].resize(Ne, 1);
	Nv[i].resize(Ne, 1);
	Yv2[i].resize(Ne, DimTan+1);
	Nv2[i].resize(Ne, DimTan+1);
    }
}

void CQCGL1dSub::changeOmega(double w){
    Omega = w;
    L = dcp(Mu, -Omega) - dcp(Dr, Di) * QK.square();
}

/** @brief Integrator of 1d cqCGL equation.
 *
 *  The intial condition a0 should be coefficients of the intial state:
 *  [b0, c0 ,b1, c1,...] where bi, ci the real and imaginary parts of Fourier modes.
 *  
 * @param[in] a0 initial condition of Fourier coefficents. Size : [2*Ne,1]
 * @return state trajectory. Each column is the state followed the previous column. 
 */
ArrayXXd 
CQCGL1dSub::intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert( Ndim == a0.size());
    
    ArrayXXcd u0 = R2C(a0); 
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(Ne, M);
    lte.resize(M);
    int ks = 0; 
    auto ss = [this, &ks, &aa](ArrayXXcd &x, double t, double h, double err){
	aa.col(ks) = x;
	lte(ks++) = err;
    }; 
	
    eidc.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
    return C2R(aa);
}

std::pair<ArrayXXd, ArrayXXd>
CQCGL1dSub::intgjC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    
    assert(a0.size() == Ndim && DimTan == Ndim);
    ArrayXXd v0(Ndim, Ndim+1); 
    v0 << a0, MatrixXd::Identity(Ndim, Ndim);
    ArrayXXcd u0 = R2C(v0);
    
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(Ne, M), daa(Ne, Ndim*M);
    lte.resize(M);
    int ks = 0;
    auto ss = [this, &ks, &aa, &daa](ArrayXXcd &x, double t, double h, double err){
	aa.col(ks) = x.col(0);
	daa.middleCols(ks*Ndim, Ndim) = x.rightCols(Ndim);
	lte(ks++) = err;
	x.rightCols(Ndim) = R2C(MatrixXd::Identity(Ndim, Ndim));
    };
    
    eidc2.intgC(nl2, ss, 0, u0, tend, h, skip_rate);
    
    return std::make_pair(C2R(aa), C2R(daa));
}

/**
 * @brief integrate the state and a subspace in tangent space
 */
ArrayXXd
CQCGL1dSub::intgvC(const ArrayXd &a0, const ArrayXXd &v, const double h,
		const double tend){
    assert( Ndim == a0.size() && Ndim == v.rows() && DimTan == v.cols());
    ArrayXXd v0(Ndim, DimTan+1);
    v0 << a0, v;
    ArrayXXcd u0 = R2C(v0);

    const int Nt = (int)round(tend/h);
    ArrayXXcd aa(Ne, DimTan+1);
    auto ss = [this, &aa](ArrayXXcd &x, double t, double h, double err){
	aa = x;
    };
    
    eidc2.intgC(nl2, ss, 0, u0, tend, h, Nt);
    
    return C2R(aa);	   //both the final orbit and the perturbation
}

ArrayXXd
CQCGL1dSub::intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert( Ndim == a0.size());
    
    ArrayXXcd u0 = R2C(a0); 
    const int Nt = (int)round(tend/h);
    const int M = (Nt+skip_rate-1)/skip_rate;
    ArrayXXcd aa(Ne, M);
    Ts.resize(M);
    hs.resize(M);
    lte.resize(M);
    int ks = 0;
    auto ss = [this, &ks, &aa](ArrayXXcd &x, double t, double h, double err){
	int m = Ts.size();
	if (ks >= m ) {
	    Ts.conservativeResize(m+cellSize);
	    aa.conservativeResize(Eigen::NoChange, m+cellSize); // rows not change, just extend cols
	    hs.conservativeResize(m+cellSize);
	    lte.conservativeResize(m+cellSize);
	}
	hs(ks) = h;
	lte(ks) = err;
	aa.col(ks) = x;
	Ts(ks++) = t;
    };
    
    eidc.intg(nl, ss, 0, u0, tend, h, skip_rate);
	
    hs.conservativeResize(ks);
    lte.conservativeResize(ks);
    Ts.conservativeResize(ks);
    aa.conservativeResize(Eigen::NoChange, ks);
    
    return C2R(aa);
}

std::pair<ArrayXXd, ArrayXXd>
CQCGL1dSub::intgj(const ArrayXd &a0, const double h, const double tend, 
	       const int skip_rate){
    
    assert(a0.size() == Ndim && DimTan == Ndim);
    ArrayXXd v0(Ndim, Ndim+1); 
    v0 << a0, MatrixXd::Identity(Ndim, Ndim);
    ArrayXXcd u0 = R2C(a0);
    
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(Ne, M), daa(Ne, Ndim*M);
    Ts.resize(M);
    hs.resize(M);
    lte.resize(M);
    int ks = 0;
    auto ss = [this, &ks, &aa, &daa](ArrayXXcd &x, double t, double h, double err){
	int m = Ts.size();
	if (ks >= m ) {
	    Ts.conservativeResize(m+cellSize);	    
	    hs.conservativeResize(m+cellSize);
	    lte.conservativeResize(m+cellSize);
	    aa.conservativeResize(Eigen::NoChange, m+cellSize); // rows not change, just extend cols
	    daa.conservativeResize(Eigen::NoChange, (m+cellSize)*Ndim);
	}
	hs(ks) = h;
	lte(ks) = err;
	Ts(ks) = t;
	aa.col(ks) = x.col(0);
	daa.middleCols(ks*Ndim, Ndim) = x.rightCols(Ndim);
	x.rightCols(Ndim) = R2C(MatrixXd::Identity(Ndim, Ndim));
	ks++;
    };

    eidc2.intg(nl2, ss, 0, u0, tend, h, skip_rate);
    return std::make_pair(C2R(aa), C2R(daa));
}
 

ArrayXXd 
CQCGL1dSub::intgv(const ArrayXXd &a0, const ArrayXXd &v, const double h,
	       const double tend){
    assert( Ndim == a0.size() && Ndim == v.rows() && DimTan == v.cols());
    ArrayXXd v0(Ndim, DimTan+1);
    v0 << a0, v;
    ArrayXXcd u0 = R2C(v0);

    ArrayXXcd aa(Ne, DimTan+1); 
    auto ss = [this, &aa](ArrayXXcd &x, double t, double h, double err){
	aa = x;
    };

    eidc2.intg(nl2, ss, 0, u0, tend, h, 1000000);
    return C2R(aa);
}


ArrayXXd CQCGL1dSub::C2R(const ArrayXXcd &v){
    return Map<ArrayXXd>((double*)&v(0,0), 2*v.rows(), v.cols());
}

ArrayXXcd CQCGL1dSub::R2C(const ArrayXXd &v){
    assert( 0 == v.rows() % 2);
    return Map<ArrayXXcd>((dcp*)&v(0,0), v.rows()/2, v.cols());
}


/* -------------------------------------------------- */
/* -------  Fourier/Configure transformation -------- */
/* -------------------------------------------------- */

/**
 * @brief back Fourier transform of the states. 
 */
ArrayXXcd CQCGL1dSub::Fourier2Config(const Ref<const ArrayXXd> &aa){
    int cs = aa.cols(), rs = aa.rows();
    assert(Ndim == rs);
    ArrayXXcd aac = R2C(aa);
    ArrayXXcd modes(N, cs);
    modes << aac, ArrayXXcd::Zero(N-2*Ne+1, cs), aac.bottomRows(Ne-1).colwise().reverse();
    ArrayXXcd field(N, cs);
    
    for(size_t i = 0; i < cs; i++)
	fft.inv(field.data() + i*N, modes.data() + i*N, N);
    
    return field;
}


/**
 * @brief Fourier transform of the states. Input and output are both real.
 */
ArrayXXd CQCGL1dSub::Config2Fourier(const Ref<const ArrayXXcd> &field){
    int cs = field.cols(), rs = field.rows();
    assert(N == rs);
    ArrayXXcd modes(N, cs);
    
    for(size_t i = 0; i < cs; i++)
	fft.fwd(modes.data() + i*N, field.data() + i*N, N);
    
    return C2R(modes.topRows(Ne));
}
#if 0
ArrayXXd CQCGL1dSub::Fourier2ConfigMag(const Ref<const ArrayXXd> &aa){
    return Fourier2Config(aa).abs();
}

ArrayXXd CQCGL1dSub::calPhase(const Ref<const ArrayXXcd> &AA){
    int m = AA.cols();
    int n = AA.rows();
    assert(N == n);
    ArrayXXd phase(n, m);
    for(size_t i = 0; i < m; i++)
	for(size_t j =0; j < n; j++)
	    phase(j, i) = atan2(AA(j, i).imag(), AA(j, i).real());

    return phase;
}

ArrayXXd CQCGL1dSub::Fourier2Phase(const Ref<const ArrayXXd> &aa){
    return calPhase(Fourier2Config(aa));
}

/**
 * @brief calculate the mean energy density
 * \frac{\int |A|^2  dx}{\int dx} = 1/N \sum |A_i|^2 = 1/N^2 \sum |a_i|^2
 * = 1/N^2 \sum (a_r^2 + a_i^2)
 */
VectorXd 
CQCGL1dSub::calQ(const Ref<const ArrayXXd> &aa){
    const int n = aa.cols();
    VectorXd Q(n);
    for (int i = 0; i < n; i++){
	Q(i) = aa.col(i).matrix().squaredNorm() / (N*N);
    }

    return Q;
}

VectorXd
CQCGL1dSub::calMoment(const Ref<const ArrayXXd> &aa, const int p){
    const int n = aa.cols();
    VectorXd mom(n);
    
    ArrayXd x = ArrayXd::LinSpaced(N, 0, N-1) * d / N;
    ArrayXd xp = x;
    for(int i = 0; i < p-1; i++) xp *= x;

    ArrayXXcd AA = Fourier2Config(aa);
    for (int i = 0; i < n; i++){
	ArrayXd A2 = (AA.col(i) * AA.col(i).conjugate()).real();
	mom(i) = (A2 * xp).sum() / A2.sum();
    }

    return mom;
}


/* -------------------------------------------------- */
/* --------            velocity field     ----------- */
/* -------------------------------------------------- */

/**
 * @brief velocity field
 */
ArrayXd CQCGL1dSub::velocity(const ArrayXd &a0){
    assert( Ndim == a0.rows() );
    ArrayXXcd A = R2C(a0);
    ArrayXXcd v(N, 1);
    nl(A, v, 0);
    return C2R(L*A + v);
}

/**
 * @brief the generalized velociyt for relative equilibrium
 *
 *   v(x) + \omega_\tau * t_\tau(x) + \omega_\rho * t_\rho(x)
 */
ArrayXd CQCGL1dSub::velocityReq(const ArrayXd &a0, const double wth,
			     const double wphi){
    return velocity(a0) + wth*transTangent(a0) + wphi*phaseTangent(a0);    
}

/**
 * velocity in the slice
 *
 * @param[in]  aH  state in the slice
 */
VectorXd CQCGL1dSub::velSlice(const Ref<const VectorXd> &aH){
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

VectorXd CQCGL1dSub::velPhase(const Ref<const VectorXd> &aH){
    VectorXd v = velocity(aH);
 
    VectorXd tp = phaseTangent(aH);   
    double c = v(Ndim-1) / aH(Ndim-2);

    return v - c * tp;
}

/* -------------------------------------------------- */
/* --------          stability matrix     ----------- */
/* -------------------------------------------------- */
MatrixXd CQCGL1dSub::stab(const ArrayXd &a0){
    assert(a0.size() == Ndim && DimTan == Ndim);
    ArrayXXd v0(Ndim, Ndim+1); 
    v0 << a0, MatrixXd::Identity(Ndim, Ndim);
    ArrayXXcd u0 = R2C(v0);
    
    ArrayXXcd v(N, Ndim+1);
    nl2(u0, v, 0);
	
    ArrayXXcd j0 = R2C(MatrixXd::Identity(Ndim, Ndim));
    MatrixXcd Z = j0.colwise() * L + v.rightCols(Ndim);
  
    return C2R(Z);
}

/**
 * @brief stability for relative equilbrium
 */
MatrixXd CQCGL1dSub::stabReq(const ArrayXd &a0, double wth, double wphi){
    MatrixXd z = stab(a0);
    return z + wth*transGenerator() + wphi*phaseGenerator();
}

/**
 * @brief stability exponents of req
 */
VectorXcd CQCGL1dSub::eReq(const ArrayXd &a0, double wth, double wphi){
    return eEig(stabReq(a0, wth, wphi), 1);
}

/**
 * @brief stability vectors of req
 */
MatrixXcd CQCGL1dSub::vReq(const ArrayXd &a0, double wth, double wphi){
    return vEig(stabReq(a0, wth, wphi), 1);
}

/**
 * @brief stability exponents and vectors of req
 */
std::pair<VectorXcd, MatrixXcd>
CQCGL1dSub::evReq(const ArrayXd &a0, double wth, double wphi){
    return evEig(stabReq(a0, wth, wphi), 1);
}


/* -------------------------------------------------- */
/* ------           symmetry related           ------ */
/* -------------------------------------------------- */

/**
 * @brief reflect the states
 *
 * Reflection : a_k -> a_{-k}. so a_0 keeps unchanged
 */
ArrayXXd CQCGL1dSub::reflect(const Ref<const ArrayXXd> &aa){
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
inline ArrayXd CQCGL1dSub::rcos2th(const ArrayXd &x, const ArrayXd &y){
    ArrayXd x2 = x.square();
    ArrayXd y2 = y.square();
    return (x2 - y2) / (x2 + y2).sqrt();
}

/**
 * @ brief calculate x * y / \sqrt{x^2 + y^2}
 */
inline ArrayXd CQCGL1dSub::rsin2th(const ArrayXd &x, const ArrayXd &y){
    return x * y / (x.square() + y.square()).sqrt();
}

/**
 * @brief calculate the gradient of rcos2th()
 *
 *        partial derivative over x :   (x^3 + 3*x*y^2) / (x^2 + y^2)^{3/2}
 *        partial derivative over y : - (y^3 + 3*y*x^2) / (x^2 + y^2)^{3/2}
 */
inline double CQCGL1dSub::rcos2thGrad(const double x, const double y){
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
inline double CQCGL1dSub::rsin2thGrad(const double x, const double y){
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
ArrayXXd CQCGL1dSub::reduceRef1(const Ref<const ArrayXXd> &aaHat){
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

ArrayXXd CQCGL1dSub::reduceRef2(const Ref<const ArrayXXd> &step1){
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
std::vector<int> CQCGL1dSub::refIndex3(){
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
ArrayXXd CQCGL1dSub::reduceRef3(const Ref<const ArrayXXd> &aa){

    ArrayXXd aaTilde(aa);
    aaTilde.row(0) = rcos2th(aa.row(0), aa.row(1));
    aaTilde.row(1) = rsin2th(aa.row(0), aa.row(1));
    
    std::vector<int> index = refIndex3();
    for(size_t i = 1; i < index.size(); i++){
	aaTilde.row(index[i]) = rsin2th(aa.row(index[i-1]), aa.row(index[i]));
    }

    return aaTilde;
}

ArrayXXd CQCGL1dSub::reduceReflection(const Ref<const ArrayXXd> &aaHat){
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
MatrixXd CQCGL1dSub::refGrad1(){
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
MatrixXd CQCGL1dSub::refGrad2(const ArrayXd &x){
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
MatrixXd CQCGL1dSub::refGrad3(const ArrayXd &x){
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
MatrixXd CQCGL1dSub::refGradMat(const ArrayXd &x){
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
MatrixXd CQCGL1dSub::reflectVe(const MatrixXd &veHat, const Ref<const ArrayXd> &xHat){
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
MatrixXd CQCGL1dSub::reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
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
ArrayXXd CQCGL1dSub::transRotate(const Ref<const ArrayXXd> &aa, const double th){
    ArrayXcd R = ( dcp(0,1) * th * K2 ).exp(); // e^{ik\theta}
    ArrayXXcd raa = r2c(aa); 
    raa.colwise() *= R;
  
    return c2r(raa);
}

/** @brief group tangent in angle unit.
 *
 *  x=(b0, c0, b1, c1, b2, c2 ...) ==> tx=(0, 0, -c1, b1, -2c2, 2b2, ...)
 */
ArrayXXd CQCGL1dSub::transTangent(const Ref<const ArrayXXd> &aa){
    ArrayXcd R = dcp(0,1) * K2;
    ArrayXXcd raa = r2c(aa);
    raa.colwise() *= R;
  
    return c2r(raa);
}

/** @brief group generator. */
MatrixXd CQCGL1dSub::transGenerator(){
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
ArrayXXd CQCGL1dSub::phaseRotate(const Ref<const ArrayXXd> &aa, const double phi){
    return c2r( r2c(aa) * exp(dcp(0,1)*phi) ); // a0*e^{i\phi}
}

/** @brief group tangent.  */
ArrayXXd CQCGL1dSub::phaseTangent(const Ref<const ArrayXXd> &aa){
    return c2r( r2c(aa) * dcp(0,1) );
}

/** @brief group generator  */
MatrixXd CQCGL1dSub::phaseGenerator(){
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
ArrayXXd CQCGL1dSub::Rotate(const Ref<const ArrayXXd> &aa, const double th,
			 const double phi){
    ArrayXcd R = ( dcp(0,1) * (th * K2 + phi) ).exp(); // e^{ik\theta + \phi}
    ArrayXXcd raa = r2c(aa); 
    raa.colwise() *= R;
  
    return c2r(raa);
}

/**
 * @brief rotate the whole orbit with different phase angles at different point
 */
ArrayXXd CQCGL1dSub::rotateOrbit(const Ref<const ArrayXXd> &aa, const ArrayXd &th,
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
 * @brief reduce the continous symmetries
 * 
 * @param[in] aa       states in the full state space
 * @return    aaHat, theta, phi
 * 
 * @note  g(theta, phi)(x+y) is different from gx+gy. There is no physical
 *        meaning to transform the sum/subtraction of two state points.
 */
std::tuple<ArrayXXd, ArrayXd, ArrayXd>
CQCGL1dSub::orbit2slice(const Ref<const ArrayXXd> &aa, const int method){
    int n = aa.rows();
    int m = aa.cols();
    assert(Ndim == n);
    ArrayXXd raa(n, m);
    ArrayXd th(m);
    ArrayXd phi(m);
    
    switch (method){

    case 1: {
	// a0 -> positive real. a1 -> positive real
	for(int i = 0; i < m; i++){
	    double x = atan2(aa(1, i), aa(0, i));
	    double y = atan2(aa(3, i), aa(2, i));
	    phi(i) = x;
	    th(i) = y - x;
	}
	break;
    }

    case 2: {
	// a0 -> positive imag. a1 -> positive real
	for (int i = 0; i < m; i++){
	    double x = atan2(aa(1, i), aa(0, i));
	    double y = atan2(aa(3, i), aa(2, i));
	    phi(i) = x - M_PI/2;
	    th(i) = y - x + M_PI/2;
	}
	break;
    }

    case 3: {
	// a0 -> positive real. a1 -> positive imag
	for (int i = 0; i < m; i++){
	    double x = atan2(aa(1, i), aa(0, i));
	    double y = atan2(aa(3, i), aa(2, i));
	    phi(i) = x;
	    th(i) = y - x - M_PI/2;
	}
	break;
    }
	
    case 4: {
	// a0 -> positive imag. a1 -> positive imag
	for (int i = 0; i < m; i++){
	    double x = atan2(aa(1, i), aa(0, i));
	    double y = atan2(aa(3, i), aa(2, i));
	    phi(i) = x - M_PI/2;
	    th(i) = y - x;
	}
	break;
    }

    case 5: {
	// a1 -> positive real. a-1 -> positive real
	// phase is wrapped. 
	for(size_t i = 0; i < m; i++){
	    double am1 = atan2(aa(n-1, i), aa(n-2, i));
	    double a1 = atan2(aa(3, i), aa(2, i));
	    phi(i) = 0.5 * (a1 + am1);
	    th(i) = 0.5 * (a1 - am1);
	    raa.col(i) = Rotate(aa.col(i), -th(i), -phi(i));
	}
	break;
    }

    case 6: {
	// a1 -> positive real. a-1 -> positive real
	// phase is unwrapped. 
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
	break;
    }
		
    default:
	fprintf(stderr, "orbit to slice error\n");
    }
	

    for(size_t i = 0; i < m; i++){
	raa.col(i) = Rotate(aa.col(i), -th(i), -phi(i));
    }

    return std::make_tuple(raa, th, phi);
}


/** @brief project covariant vectors to 1st mode slice
 *
 * @note vectors are not normalized
 */
MatrixXd CQCGL1dSub::ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag){
    int n = x.size();
    ArrayXXd xhat;
    ArrayXd th, phi;
    std::tie(xhat, th, phi) = orbit2slice(x, flag);
    VectorXd tx_rho = phaseTangent(xhat);
    VectorXd tx_tau = transTangent(xhat);
    VectorXd x0_rho(n), x0_tau(n);
    x0_rho.setZero();
    x0_tau.setZero();
    
    switch (flag){
    case 1 : {
	x0_rho(0) = 1;
	x0_tau(2) = 1;
	break;
    }
	
    case 2 : {
	x0_rho(1) = 1;
	x0_tau(2) = 1;
	break;
    }
    
    case 3 : {
	x0_rho(0) = 1;
	x0_tau(3) = 1;
	break;
    }
	
    case 4 : {
	x0_rho(1) = 1;
	x0_tau(3) = 1;
	break;
    }
	
    case 5 : {
	x0_rho(n-2) = 1;
	x0_tau(3) = 1;
	break;
    }
	
    case 6 : {
	x0_rho(n-2) = 1;
	x0_tau(2) = 1;
	break;
    }

    default:
	fprintf(stderr, "orbit to slice error\n");
    
    }

    VectorXd tp_rho = phaseTangent(x0_rho);
    VectorXd tp_tau = transTangent(x0_tau);

    Matrix2d L;
    L << tx_rho.dot(tp_rho), tx_tau.dot(tp_rho),
	tx_rho.dot(tp_tau), tx_tau.dot(tp_tau);

    MatrixXd tx(n, 2);
    tx.col(0) = tx_rho;
    tx.col(1) = tx_tau;

    MatrixXd t0(n, 2);
    t0.col(0) = tp_rho;
    t0.col(1) = tp_tau;

    MatrixXd vep = Rotate(ve, -th(0), -phi(0));
    vep = vep - tx * L.inverse() * t0.transpose() * vep;

    return vep;

}

/**
 * @brief a wrap function => reduce all symmetries of an orbit
 */
std::tuple<ArrayXXd, ArrayXd, ArrayXd>
CQCGL1dSub::reduceAllSymmetries(const Ref<const ArrayXXd> &aa, int flag){
    std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(aa, flag);
    return std::make_tuple(reduceReflection(std::get<0>(tmp)),
			   std::get<1>(tmp), std::get<2>(tmp));
}

/**
 * @brief a wrap function => reduce all the symmetries of covariant vectors
 */
MatrixXd CQCGL1dSub::reduceVe(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag){
    std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(x, flag);
    return reflectVe(ve2slice(ve, x, flag), std::get<0>(tmp).col(0));
}

#endif
