#include "ksint.hpp"
#include <cmath>
#include <iostream>
using std::cout; using std::endl;

/*============================================================
 *                       Class : KS integrator
 *============================================================*/

/*-------------------- constructor, destructor -------------------- */
KS::KS(int N, double h, double d) : N(N), h(h), d(d) {
  /* calcute various coefficients used for the integrator */
  ksInit();
  /* initialize the FFTW holders */
  initFFT(Fv, 1); initFFT(Fa, 1); initFFT(Fb, 1); initFFT(Fc, 1);
  initFFT(jFv, N-1); initFFT(jFa, N-1); initFFT(jFb, N-1); initFFT(jFc, N-1);
}

KS::KS(const KS &x) : N(x.N), d(x.d), h(x.h){};

KS & KS::operator=(const KS &x){
  return *this;
}

KS::~KS(){
  freeFFT(Fv); freeFFT(Fa); freeFFT(Fb); freeFFT(Fc);
  freeFFT(jFv); freeFFT(jFa); freeFFT(jFb); freeFFT(jFc);
  fftw_cleanup();
}

/*------------------- member methods ------------------ */
void KS::ksInit(){
  K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d; //2*PI/d*[0, 1, 2,...,N/2]
  K(N/2) = 0;
  L = K*K - K*K*K*K;
  E = (h*L).exp();
  E2 = (h/2*L).exp();
  
  ArrayXd tmp = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 
  ArrayXXcd r = ((tmp-0.5)/M * dcp(0,M_PI)).exp().transpose();
  ArrayXXcd Lc = ArrayXXcd::Zero(N/2+1, 1); 
  Lc.real() = L;
  ArrayXXcd LR = h*Lc.replicate(1, M) + r.replicate(N/2+1, 1);
  ArrayXXcd LR2 = LR.square();
  ArrayXXcd LR3 = LR.cube();
  ArrayXXcd LRe = LR.exp();
  
  Q = h * ( ((LR/2.0).exp() - 1)/LR ).rowwise().mean().real(); 
  f1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean().real();
  f2 = h * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean().real();
  f3 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean().real();
  G = 0.5 * dcp(0,1) * K * N; 
  
  jG = ArrayXXcd::Zero(G.rows(), N-1); jG << G, 2.0*G.replicate(1, N-2); 
}

ArrayXXd KS::intg(const ArrayXd &a0, size_t nstp, size_t np){
  if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);}  
  Fv.vc1 = R2C(a0);  
  ArrayXXd aa(N-2, nstp/np + 1);
  aa.col(0) = a0;

  for(size_t i = 1; i < nstp +1; i++){
    NL(Fv); Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3; 
    NL(Fa); Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;
    NL(Fb); Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);
    NL(Fc);
    Fv.vc1 = E*Fv.vc1 + Fv.vc3*f1 + 2.0*(Fa.vc3+Fb.vc3)*f2 + Fc.vc3*f3;
    
    if( 0 == i%np ) aa.col(i/np) = C2R(Fv.vc1);
  }
  
  return aa;
}

/** @brief intgj() integrate KS system and calculate Jacobian along this orbit.
 *
 * @param[in] a0 Initial condition of the orbit. Size [N-2,1]
 * @param[in] nstp Number of steps to integrate.
 * @return Pair value. The first element is the trajectory.
 *         The second element is the Jacobian along the trajectory.
 */
std::pair<ArrayXXd, ArrayXXd>
KS::intgj(const ArrayXd &a0, size_t nstp, size_t np, size_t nqr){
  if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);}  
  ArrayXXd v0(N-2, N-1); 
  v0 << a0, MatrixXd::Identity(N-2, N-2);
  jFv.vc1 = R2C(v0);
  ArrayXXd aa(N-2, nstp/np+1); aa.col(0) = a0;
  ArrayXXd daa((N-2)*(N-2), nstp/nqr);
  for(size_t i = 1; i < nstp + 1; i++){ // diagonal trick 
    jNL(jFv); jFa.vc1 = E2.matrix().asDiagonal()*jFv.vc1.matrix() +
		Q.matrix().asDiagonal()*jFv.vc3.matrix();
    jNL(jFa); jFb.vc1 = E2.matrix().asDiagonal()*jFv.vc1.matrix() + 
		Q.matrix().asDiagonal()*jFa.vc3.matrix();
    jNL(jFb); jFc.vc1 = E2.matrix().asDiagonal()*jFa.vc1.matrix() + 
		Q.matrix().asDiagonal()*(2.0*jFb.vc3 - jFv.vc3).matrix();
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

void KS::NL(KSfft &f){
  ifft(f);
  f.vr2 = f.vr2 * f.vr2;
  fft(f);
  f.vc3 *= G;
}

void KS::jNL(KSfft &f){
  ifft(f); 
  ArrayXd tmp = f.vr2.col(0);
  // f.vr2.colwise() *= tmp;
  f.vr2 = tmp.matrix().asDiagonal()*f.vr2.matrix();
  fft(f);
  f.vc3 *= jG;
}

void KS::initFFT(KSfft &f, int M) {
  f.r2 = (double*) fftw_malloc(sizeof(double) * N * M);
  f.c1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1) * M);
  f.c3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1) * M);
  
  //build the maps.
  new (&(f.vr2)) Map<ArrayXXd>( &(f.r2[0]), N, M);
  new (&(f.vc1)) Map<ArrayXXcd>( (dcp*)&(f.c1[0][0]), N/2+1, M );
  new (&(f.vc3)) Map<ArrayXXcd>( (dcp*)&(f.c3[0][0]), N/2+1, M );
  
  if (1 == M){
    f.p = fftw_plan_dft_r2c_1d(N, f.r2, f.c3, FFTW_MEASURE);
    f.rp = fftw_plan_dft_c2r_1d(N, f.c1, f.r2, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
  } else{
    int n[]={N};
    f.p = fftw_plan_many_dft_r2c(1, n, N-1, f.r2, n, 1, N, 
				 f.c3, n, 1, N/2+1, FFTW_MEASURE);
    f.rp = fftw_plan_many_dft_c2r(1, n, N-1, f.c1, n, 1, N/2+1,
				  f.r2, n, 1, N, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
  }
      
}

void KS::freeFFT(KSfft &f){
  /* free the memory */
  fftw_destroy_plan(f.p);
  fftw_destroy_plan(f.rp);
  fftw_free(f.c1);
  fftw_free(f.r2);
  fftw_free(f.c3);
  /* release the maps */
  new (&(f.vc1)) Map<ArrayXXcd>(NULL, 0, 0);
  new (&(f.vr2)) Map<ArrayXXd>(NULL, 0, 0);
  new (&(f.vc3)) Map<ArrayXXcd>(NULL, 0, 0);
}

void KS::fft(KSfft &f){
  fftw_execute(f.p); 
}

void KS::ifft(KSfft &f){
  fftw_execute(f.rp); 
  f.vr2 /= N;
}

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
  if(0 != n%2) { 
    printf("The dimension of r2c is wrong !\n");
    exit(1);
  }
  ArrayXXcd vp = ArrayXXcd::Zero(n/2+2, m);
  vp.middleRows(1, n/2) = Map<ArrayXXcd>((dcp*)&v(0,0), n/2, m);

  return vp;
}

/*************************************************** 
 *           Symmetry related                      *
 ***************************************************/
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
