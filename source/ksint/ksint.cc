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
  // comment out when trying to compile interface for Matlab/Python 
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
    NL(Fv); Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3; 
    NL(Fa); Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;
    NL(Fb); Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);
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

/** @brief calculate the velocity 
 *
 * @param[in] a0 state vector
 * @return velocity field at a0
 */
VectorXd 
KS::velocity(const Ref<const ArrayXd> &a0){
  assert(a0.rows() == N-2);
  Fv.vc1 = R2C(a0);
  NL(Fv); 
  Fv.vc1 = L * Fv.vc1 + Fv.vc3;
  
  return C2R(Fv.vc1);
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
