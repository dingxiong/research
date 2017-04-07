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

// the dimension should be [Ne, DimTan+1]
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
    ArrayXXcd v(Ne, 1);
    nl(A, v, 0);
    return C2R(L*A + v);
}

/**
 * @brief the generalized velociyt for relative equilibrium
 *
 *   v(x) + \omega_\tau * t_\tau(x) + \omega_\rho * t_\rho(x)
 */
ArrayXd CQCGL1dSub::velocityReq(const ArrayXd &a0, const double wphi){
    return velocity(a0) + wphi*phaseTangent(a0);    
}

/* -------------------------------------------------- */
/* --------          stability matrix     ----------- */
/* -------------------------------------------------- */
MatrixXd CQCGL1dSub::stab(const ArrayXd &a0){
    assert(a0.size() == Ndim && DimTan == Ndim);
    ArrayXXd v0(Ndim, Ndim+1); 
    v0 << a0, MatrixXd::Identity(Ndim, Ndim);
    ArrayXXcd u0 = R2C(v0);
    
    ArrayXXcd v(Ne, Ndim+1);
    nl2(u0, v, 0);
	
    ArrayXXcd j0 = R2C(MatrixXd::Identity(Ndim, Ndim));
    MatrixXcd Z = j0.colwise() * L + v.rightCols(Ndim);
    
    return C2R(Z);
}

/**
 * @brief stability for relative equilbrium
 */
MatrixXd CQCGL1dSub::stabReq(const ArrayXd &a0, double wphi){
    return stab(a0) + wphi*phaseGenerator();
}

/**
 * @brief stability exponents of req
 */
VectorXcd CQCGL1dSub::eReq(const ArrayXd &a0, double wphi){
    return eEig(stabReq(a0, wphi), 1);
}

/**
 * @brief stability vectors of req
 */
MatrixXcd CQCGL1dSub::vReq(const ArrayXd &a0, double wphi){
    return vEig(stabReq(a0, wphi), 1);
}

/**
 * @brief stability exponents and vectors of req
 */
std::pair<VectorXcd, MatrixXcd>
CQCGL1dSub::evReq(const ArrayXd &a0, double wphi){
    return evEig(stabReq(a0, wphi), 1);
}


/* -------------------------------------------------- */
/* ------           symmetry related           ------ */
/* -------------------------------------------------- */

ArrayXXd CQCGL1dSub::phaseRotate(const Ref<const ArrayXXd> &aa, const double phi){
    return C2R( R2C(aa) * exp(dcp(0,1)*phi) ); // a0*e^{i\phi}
}

ArrayXXd CQCGL1dSub::phaseTangent(const Ref<const ArrayXXd> &modes){
    return C2R( R2C(modes) * dcp(0,1) );
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
