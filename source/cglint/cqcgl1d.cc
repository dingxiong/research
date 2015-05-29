#include "cqcgl1d.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
using std::cout;
using std::endl;
using namespace sparseRoutines;
/*      ----------------------------------------------------
 *                        Class Cqcgl1d
 *      ----------------------------------------------------
 */

/* ------------------------------------------------------ */
/* ----                constructor/destructor     ------- */
/* ------------------------------------------------------ */
Cqcgl1d::Cqcgl1d(int N, double d, double h,
		 double Mu, double Br, double Bi,
		 double Dr, double Di, double Gr,
		 double Gi)
    : N(N), d(d), h(h),
      Mu(Mu), Br(Br), Bi(Bi),
      Dr(Dr), Di(Di), Gr(Gr),
      Gi(Gi)
{
    CGLInit(); // calculate coefficients.
  
    // initialize fft/ifft plan
#ifdef TFFT  // mutlithread fft.
    if(!fftw_init_threads()){
	printf("error create MultiFFT.\n");
	exit(1);
    }
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif	/* TFFT */

    initFFT(Fv, 1); initFFT(Fa, 1); initFFT(Fb, 1); initFFT(Fc, 1);
    initFFT(jFv, 2*N+1); initFFT(jFa, 2*N+1); initFFT(jFb, 2*N+1); initFFT(jFc, 2*N+1);
}

Cqcgl1d::Cqcgl1d(const Cqcgl1d &x) : N(x.N), d(x.d), h(x.h),
				     Mu(x.Mu), Br(x.Br), Bi(x.Bi),
				     Dr(x.Dr), Di(x.Di), Gr(x.Gr),
				     Gi(x.Gi)
{}

Cqcgl1d::~Cqcgl1d(){
    // destroy fft/ifft plan
    freeFFT(Fv); freeFFT(Fa); freeFFT(Fb); freeFFT(Fc);
    freeFFT(jFv); freeFFT(jFa); freeFFT(jFb); freeFFT(jFc);
    //fftw_cleanup();
  
#ifdef TFFT
    fftw_cleanup_threads();
#endif	/* TFFT */
}

Cqcgl1d & Cqcgl1d::operator=(const Cqcgl1d &x){
    return *this;
}

/* -------------------------------------------------- */
/* ---------        integrator         -------------- */
/* -------------------------------------------------- */

/** @brief Integrator of 1d cqCGL equation.
 *
 *  The intial condition a0 should be coefficients of the intial state:
 *  [b0, c0 ,b1, c1,...] where bi, ci the real and imaginary parts of Fourier modes.
 *  
 * @param[in] a0 initial condition of Fourier coefficents. Size : [2*N,1]
 * @param[in] nstp number of integration steps.
 * @param[in] np spacing of saving the trajectory.
 * @return state trajectory. Each column is the state followed the previous column. 
 *         Size : [2*N, nst/np+1] 
 */
ArrayXXd Cqcgl1d::intg(const ArrayXd &a0, const size_t nstp, const size_t np){
    assert( 2*N == a0.rows() ); // check the dimension of initial condition.
    Fv.v1 = R2C(a0);
    ArrayXXd uu(2*N, nstp/np+1); uu.col(0) = a0;  
  
    for(size_t i = 1; i < nstp+1; i++) {    
	NL(Fv);  Fa.v1 = E2*Fv.v1 + Q*Fv.v3;
	NL(Fa);  Fb.v1 = E2*Fv.v1 + Q*Fa.v3;
	NL(Fb);  Fc.v1 = E2*Fa.v1 + Q*(2.0*Fb.v3-Fv.v3);
	NL(Fc); 
	Fv.v1 = E*Fv.v1 + Fv.v3*f1 + (Fa.v3+Fb.v3)*f2 + Fc.v3*f3;
      
	if( i%np == 0 ) uu.col(i/np) = C2R(Fv.v1) ;
    }

    return uu;
}

pair<ArrayXXd, ArrayXXd>
Cqcgl1d::intgj(const ArrayXd &a0, const size_t nstp,
	       const size_t np, const size_t nqr){
    assert( 2*N == a0.rows() ); // check the dimension of initial condition.

    // Kronecker product package is not available right now.
    ArrayXXcd J0 = ArrayXXcd::Zero(N,2*N);
    for(size_t i = 0; i < N; i++) {
	J0(i,2*i) = dcp(1,0);
	J0(i,2*i+1) = dcp(0,1);
    }

    jFv.v1 << R2C(a0), J0;
    ArrayXXd uu(2*N, nstp/np+1); uu.col(0) = a0;  
    ArrayXXd duu((2*N)*(2*N), nstp/nqr);
  
    for(size_t i = 1; i < nstp + 1; i++){
	jNL(jFv); jFa.v1 = jFv.v1.colwise() * E2 + jFv.v3.colwise() * Q;
	jNL(jFa); jFb.v1 = jFv.v1.colwise() * E2 + jFv.v3.colwise() * Q;
	jNL(jFb); jFc.v1 = jFa.v1.colwise() * E2 + (2.0*jFb.v3 - jFv.v3).colwise() * Q;
	jNL(jFc);
    
	jFv.v1 = jFv.v1.colwise() * E + jFv.v3.colwise() * f1 +
	    (jFa.v3 + jFb.v3).colwise() * f2 + jFc.v3.colwise() * f3;
    
	if ( 0 == i%np ) uu.col(i/np) = C2R(jFv.v1.col(0));
	if ( 0 == i%nqr){
	    Map<ArrayXcd> tmp(&jFv.v1(0,1), N*2*N, 1); 
	    duu.col(i/nqr - 1) = C2R(tmp); 
	    jFv.v1.rightCols(2*N) = J0;
	}    
    }
  
    return make_pair(uu, duu);
}

void Cqcgl1d::CGLInit(){
    // calculate the ETDRK4 coefficients
    Kindex.resize(N,1);
    Kindex << ArrayXd::LinSpaced(N/2+1, 0, N/2), ArrayXd::LinSpaced(N/2-1, -N/2+1, -1);
    K = 2*M_PI/d * Kindex;
    L = Mu - dcp(Dr, Di) * K.square();
    E = (h*L).exp(); 
    E2 =(h/2*L).exp();
  
    ArrayXd tmp = ArrayXd::LinSpaced(M, 1, M);
    ArrayXXcd r = (tmp/M*dcp(0,2*M_PI)).exp().transpose(); // row array.

    ArrayXXcd LR = h*L.replicate(1, M) + r.replicate(N, 1);
    ArrayXXcd LR2 = LR. square();
    ArrayXXcd LR3 = LR.cube();
    ArrayXXcd LRe = LR.exp();

    Q = h * ( ((LR/2.0).exp() - 1)/LR ).rowwise().mean();
    f1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
    f2 = h * ( (4.0 + 2.0*LR + LRe*(-4.0 + 2.0*LR)) / LR3 ).rowwise().mean();
    f3 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();

}



void Cqcgl1d::NL(CGLfft &f){
    ifft(f);
    ArrayXcd A2 = f.v2 * f.v2.conjugate();
    f.v2 =  dcp(Br, Bi) * f.v2 * A2 + dcp(Gr, Gi) * f.v2 * A2.square();
    fft(f);
}

void Cqcgl1d::jNL(CGLfft &f){
    ifft(f); 
    ArrayXcd A = f.v2.col(0);
    ArrayXcd aA2 = A * A.conjugate();
    ArrayXcd A2 = A.square();
    dcp B(Br, Bi);
    dcp G(Gr, Gi);
    f.v2.col(0) = dcp(Br, Bi) * A * aA2 + dcp(Gr, Gi) * A * aA2.square();

    f.v2.rightCols(2*N) = f.v2.rightCols(2*N).conjugate().colwise() *  ((B+G*2.0*aA2) * A2) +
    	f.v2.rightCols(2*N).colwise() * ((2.0*B+3.0*G*aA2)*aA2);

    fft(f);
}

void Cqcgl1d::fft(CGLfft &f){
  fftw_execute(f.p);  
}

void Cqcgl1d::ifft(CGLfft &f){
  fftw_execute(f.rp);
  f.v2 /= N;
}

void Cqcgl1d::initFFT(CGLfft &f, int M){
  f.c1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
  f.c2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
  f.c3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
  //build the maps.
  new (&(f.v1)) Map<ArrayXXcd>( (dcp*)&(f.c1[0][0]), N, M );
  new (&(f.v2)) Map<ArrayXXcd>( (dcp*)&(f.c2[0][0]), N, M );
  new (&(f.v3)) Map<ArrayXXcd>( (dcp*)&(f.c3[0][0]), N, M );

  if (1 == M){
    f.p = fftw_plan_dft_1d(N, f.c2, f.c3, FFTW_FORWARD, FFTW_MEASURE);
    f.rp = fftw_plan_dft_1d(N, f.c1, f.c2, FFTW_BACKWARD, FFTW_MEASURE);
  } else{
    int n[] = { N };
    f.p = fftw_plan_many_dft(1, n, M, f.c2, n, 1, N,
			     f.c3, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    f.rp = fftw_plan_many_dft(1, n, M, f.c1, n, 1, N,
			      f.c2, n, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
  }
}

void Cqcgl1d::freeFFT(CGLfft &f){
  fftw_destroy_plan(f.p);
  fftw_destroy_plan(f.rp);
  fftw_free(f.c1);
  fftw_free(f.c2);
  fftw_free(f.c3);
  /* releae the map */
  new (&(f.v1)) Map<ArrayXXcd>(NULL, 0, 0);
  new (&(f.v2)) Map<ArrayXXcd>(NULL, 0, 0);
  new (&(f.v3)) Map<ArrayXXcd>(NULL, 0, 0);
}

/** @brief transform conjugate matrix to its real form */
ArrayXXd Cqcgl1d::C2R(const ArrayXXcd &v){
  // allocate memory for new array, so it will not change the original array.
  return Map<ArrayXXd>((double*)&v(0,0), 2*v.rows(), v.cols());
}

ArrayXXcd Cqcgl1d::R2C(const ArrayXXd &v){
  if(0 != v.rows()%2 ) { printf("R2C dimension wrong.\n"); exit(1); }
  return Map<ArrayXXcd>((dcp*)&v(0,0), v.rows()/2, v.cols());
}

/* -------------------------------------------------- */
/* --------            velocity field     ----------- */
/* -------------------------------------------------- */
ArrayXd Cqcgl1d::velocity(const ArrayXd &a0){
  assert( 2*N == a0.rows() );
  Fv.v1 = R2C(a0);
  NL(Fv);
  ArrayXcd vel = L*Fv.v1 + Fv.v3;
  return C2R(vel);
}

/* -------------------------------------------------- */
/* --------          stability matrix     ----------- */
/* -------------------------------------------------- */
MatrixXd Cqcgl1d::stab(const ArrayXd &a0){
  ArrayXXcd j0 = MatrixXcd::Zero(N, 2*N);
  for(size_t i = 0; i < N; i++) {
    j0(i,2*i) = dcp(1,0);
    j0(i,2*i+1) =dcp(0,1);
  }
  jFv.v1 << R2C(a0), j0;
  jNL(jFv);
  MatrixXcd Z = j0.colwise() * L + jFv.v3.rightCols(2*N);
  
  return C2R(Z);
}

MatrixXd Cqcgl1d::stabReq(const ArrayXd &a0, double w1, double w2){
    MatrixXd z = stab(a0);
    return z + w1*transGenerator() + w2*phaseGenerator();
}
/* -------------------------------------------------- */
/* ------           symmetry related           ------ */
/* -------------------------------------------------- */

/** @brief group rotation for spatial translation of set of arrays.
 *  th : rotation angle
 *  */
ArrayXXd Cqcgl1d::transRotate(const Ref<const ArrayXXd> &aa, const double th){
    ArrayXcd R = ( dcp(0,1) * th * Kindex ).exp(); // e^{ik\theta}
    ArrayXXcd raa = R2C(aa); 
    raa.colwise() *= R;
  
    return C2R(raa);
}

/** @brief group tangent in angle unit.
 *
 *  x=(b0, c0, b1, c1, b2, c2 ...) ==> tx=(0, 0, -c1, b1, -2c2, 2b2, ...)
 */
ArrayXXd Cqcgl1d::transTangent(const Ref<const ArrayXXd> &aa){
    ArrayXcd R = dcp(0,1) * Kindex;
    ArrayXXcd raa = R2C(aa);
    raa.colwise() *= R;
  
    return C2R(raa);
}

/** @brief group generator. */
MatrixXd Cqcgl1d::transGenerator(){
  MatrixXd T = MatrixXd::Zero(2*N, 2*N);
  for(size_t i = 0; i < N; i++){
    T(2*i, 2*i+1) = -Kindex(i);
    T(2*i+1, 2*i) = Kindex(i);
  }
  return T;
}


/** @brief group transform for complex rotation
 * phi: rotation angle
 * */
ArrayXXd Cqcgl1d::phaseRotate(const Ref<const ArrayXXd> &aa, const double phi){
  
  return C2R( R2C(aa) * exp(dcp(0,1)*phi) ); // a0*e^{i\phi}
}

/** @brief group tangent.  */
ArrayXXd Cqcgl1d::phaseTangent(const Ref<const ArrayXXd> &aa){
  return C2R( R2C(aa) * dcp(0,1) );
}

/** @brief group generator  */
MatrixXd Cqcgl1d::phaseGenerator(){
  MatrixXd T = MatrixXd::Zero(2*N, 2*N);
  for(size_t i = 0; i < N; i++){
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
ArrayXXd Cqcgl1d::Rotate(const Ref<const ArrayXXd> &aa, const double th,
			 const double phi){
    ArrayXcd R = ( dcp(0,1) * th * Kindex + phi).exp(); // e^{ik\theta + \phi}
    ArrayXXcd raa = R2C(aa); 
    raa.colwise() *= R;
  
    return C2R(raa);
    return phaseRotate( transRotate(aa, th), phi);
}

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
VectorXd Cqcgl1d::multiF(const ArrayXXd &x, const int nstp, const double th, const double phi){
    int m = x.cols();
    int n = x.rows();
    assert( 2*N == n );
    
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
pair<Cqcgl1d::SpMat, VectorXd>
Cqcgl1d::multishoot(const ArrayXXd &x, const int nstp, const double th,
		    const double phi, bool doesPrint /* = false*/){
    int m = x.cols();		/* number of shooting points */
    int n = x.rows();
    assert( 2*N == n );
  
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
	J.resize(n, n);
	
	if(i < m-1){
	    // J
	    std::vector<Tri> triJ = triMat(J, i*n, i*n);
	    nz.insert(nz.end(), triJ.begin(), triJ.end());
	    // velocity
	    std::vector<Tri> triv = triMat(velocity(aa.col(1)), i*n, m*n);
	    nz.insert(nz.end(), triv.begin(), triv.end());
	    // f(x_i) - x_{i+1}
	    F.segment(i*n, n) = aa.col(1) - x.col(i+1);
	} else {
	    ArrayXd gfx = Rotate(aa.col(1), th, phi); /* gf(x) */
	    // g*J
	    std::vector<Tri> triJ = triMat(Rotate(J, th, phi), i*n, i*n);     
	    nz.insert(nz.end(), triJ.begin(), triJ.end());
	    // R*velocity
	    std::vector<Tri> triv = triMat(Rotate(velocity(aa.col(1)), th, phi), i*n, m*n);
	    nz.insert(nz.end(), triv.begin(), triv.end());
	    // T_\tau * g * f(x_{m-1})
	    VectorXd tx_trans = transTangent(gfx) ;
	    std::vector<Tri> tritx_trans = triMat(tx_trans, i*n, m*n+1);
	    nz.insert(nz.end(), tritx_trans.begin(), tritx_trans.end());
	    // T_\phi * g * f(x_{m-1})
	    ArrayXd tx_phase = phaseTangent( gfx );
	    std::vector<Tri> tritx_phase = triMat(tx_phase, i*n, m*n+2);
	    nz.insert(nz.end(), tritx_phase.begin(), tritx_phase.end());
	    // g*f(x_{m-1}) - x_0
	    F.segment(i*n, n) = gfx  - x.col((i+1)%m);
	}
	std::vector<Tri> triI = triDiag(n, -1, i*n, ((i+1)%m)*n);
	nz.insert(nz.end(), triI.begin(), triI.end());
    }
    if(doesPrint) printf("\n");
    
    DF.setFromTriplets(nz.begin(), nz.end());

    return make_pair(DF, F);
}
