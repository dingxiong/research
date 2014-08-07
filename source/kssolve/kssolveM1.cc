#include "kssolveM1.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <boost/numeric/odeint.hpp>

using boost::numeric::odeint::integrate;

/*===============             Class KsM1            =================*/

/*----------               constructors and destructor            ------------*/
KsM1::KsM1(int N, double d, double h) : Ks(N, d, h), int2p(*this){}
KsM1::KsM1(const KsM1 &x) :  Ks(x), int2p(x.int2p){};
//KsM1::~KsM1() : ~Ks(){}
KsM1 & KsM1::operator=(const KsM1 &x){
  return *this;
}

/*-----------               member functions                     -------------*/

void
KsM1::kssolve(double *a0, int nstp, int np, int nqr, double *aa, double *daa, double *tt){
  
  FFT rp[N-1], p[N-1];
  dcomp v[(N/2+1) * (N-1)]; 	/* the last row vector is zero. */
  initKs(a0, v, aa, rp, p);
 
  tt[0] = 0;	
  int ix1 = 1; // index counter for aa.
  // the size of daa should be (N-2)^2 * nstp/nqr
  int ix2 = 0; // index counter for daa.  
  for (int i = 0; i < nstp; ++i){    
    onestep( v, p, rp);    
    if( (i+1)%np == 0 && i != nstp - 1) {
      aa[ix1*(N-3)] = v[1].real();
      for(int i = 0; i< N/2 - 2; i++) {
	aa[ix1*(N-3)+ 2*i + 1] = v[i+2].real();
	aa[ix1*(N-3)+ 2*i + 2] = v[i+2].imag();
      }
      tt[ix1] = tt[ix1-1] + (aa[ix1 * (N-3)] + aa[(ix1-1) * (N-3)]) * h / 2;
      ix1++;
    }

    if((i+1)%nqr == 0){
      for(int i = 0; i < N - 3; i++){ // only store N - 3 arrays. 
	for(int j = 0; j < N/2 -1; j++){
	  if(j == 0) 
	    daa[ix2*(N-3)*(N-3) + i * (N - 3) ] = v[(N/2+1)*(i+1) + 1].real();
	  else{
	    // reading from v starts from second row  and second column. 
	    daa[ix2*(N-3)*(N-3) + i * (N - 3) + 2*j - 1] = v[(N/2+1)*(i+1) + j+1].real();
	    daa[ix2*(N-3)*(N-3) + i * (N - 3) + 2*j] = v[(N/2+1)*(i+1) + j+1].imag();
	  }
	}
      }
      ix2++;
      initJ(v); // Initialize Jacobian again.
    }
  }
  cleanKs(rp, p);
}


void 
KsM1::kssolve(double *a0, int nstp, int np, double *aa, double *tt){

  FFT rp, p; 
  dcomp v[N/2+1]; 
  initKs(a0, v, aa, rp, p);

  tt[0] = 0;			/* initialize time in the full state space. */
  int ix = 1;
  
  for(int i=0; i<nstp; ++i){
    onestep(v, p, rp);
    if((i+1)%np ==0 && i != nstp - 1){
      aa[ix*(N-3)] = v[1].real();
      for(int i = 0; i < N/2 - 2; i++){
	aa[ix*(N-3) + 2*i + 1] = v[i+2].real();
	aa[ix*(N-3) + 2*i + 2] = v[i+2].imag();
      }
      /* update time. */
      tt[ix] = tt[ix-1] + (aa[ix * (N-3)] + aa[(ix-1) * (N-3)]) * h / 2;
      ix++;
    }
  }
  cleanKs(rp, p);
}


/** @brief calculte the nonlinear term for the 1st mode slice without Jacobian.
 * hide calNL() in parent class.
 */
void KsM1::calNL(dcomp*u, dcomp *Nv, const FFT &p, const FFT &rp){

  irfft(u, rp);
  for(int i = 0; i < N; i++) rp.r[i] = rp.r[i] * rp.r[i];
  rfft(rp.r, p);
  
  double a1 = u[1].real();
  for(int i = 0; i < N/2 + 1; i++)
    Nv[i] = (a1-1) * coe.L[i] * u[i] + a1 * coe.g[i] * dcomp(p.c[i][0], p.c[i][1])
      - p.c[1][0] * coe.g[i] * u[i];

}

/** @brief calculate the nonlinear term when the Jabobian is also desired.
 * 
 * @parameter[out] Nv returned nonliear term. [N-1, N/2+1] row-wise complex array.
 *                 The last row is all zeros.
 * @parameter[in]  u  input data. [N-1, N/2+1] row-wise complex array
 *                 The last row is all zeros.        
 */
void KsM1::calNL(dcomp *u, dcomp *Nv, const FFT *p, const FFT *rp ){

  
  /* only calculate the first N - 2 vectors. */
  for(int i = 0; i < N - 2; i++) irfft(u+i*(N/2+1), rp[i]);
	
  // make a copy of the first transformed vector since the square 
  // operation is inline, it will destroy rp[0].r[j].
  double tmp[N];
  memcpy(tmp, rp[0].r, N * sizeof(double));

  /* only calculate the converlustion of the first N - 2 vectors. */
  for(int i = 0; i < N - 2; i++){    
    for(int j = 0; j < N; j++)
      rp[i].r[j] = tmp[j] * rp[i].r[j];    
    rfft(rp[i].r, p[i]);  // get the converlution terms.
  }
  
  /* calculate the nonlinear term. */
  double a1 = u[1].real();
  for(int i = 0; i < N - 2; i++){
    for(int j = 0; j < N/2 + 1; j++){

      if(i == 0) {
	Nv[i*(N/2+1) + j] =  (a1-1) * coe.L[j] * u[j] + a1 * coe.g[j] * 
	  dcomp(p[i].c[j][0], p[i].c[j][1]) - p[i].c[1][0] * coe.g[j] * u[j];
      }
      else {
	Nv[i*(N/2+1) + j] = (a1 - 1) * coe.L[j] * u[i*(N/2+1) + j] + 
	  2.0 * a1 * coe.g[j] * dcomp(p[i].c[j][0], p[i].c[j][1])
	  - coe.g[j] * u[i*(N/2+1) + j] * p[0].c[1][0]
	  + u[i*(N/2+1) + 1] * 
	  ( coe.L[j] * u[j] + coe.g[j] * dcomp(p[0].c[j][0], p[0].c[j][1]))
	  - 2.0 * coe.g[j] * u[j] * p[i].c[1][0]; 
      }
    }
  }
}

/** @brief Initialize the KS system fon the 1st slice without Jacobian
 *  Hide initKs() in parent class
 */

void KsM1::initKs(double *a0, dcomp *v, double *aa, FFT &rp, FFT &p){
   // the size of aa should be (N-3)*(nstp/np)
  for(int i = 0; i<N-3; ++i) aa[i] = a0[i];
  
  // initialize initial data.
  v[0] =dcomp(0,0); 
  v[N/2] = dcomp(0,0); 
  v[1] = dcomp(a0[0], 0);
  for(int i = 0; i<N/2-2; i++) v[i+2] = dcomp(a0[2*i+1], a0[2*i+2]);
 
  /* FFT initiate. */
  initFFT(rp, -1);
  initFFT(p, 1);
  
}

/** @brief Initialize the KS system with calculating Jacobian */
void KsM1::initKs(double *a0, dcomp *v, double *aa, FFT *rp, FFT *p){
  // the size of aa should be (N-3)*(nstp/np)
  for(int i = 0; i < N-3; i++) aa[i] = a0[i];

  //initialize initial data.
  v[0] = dcomp(0, 0);
  v[N/2] = dcomp(0, 0);
  v[1] = dcomp(a0[0], 0);
  for(int i = 0; i<N/2-2; i++) v[i+2] = dcomp(a0[2*i+1], a0[2*i+2]);

  initJ(v);
  
  for(int i = 0; i < N-1; i++) initFFT(rp[i], -1);
  for(int i = 0; i < N-1; i++) initFFT(p[i], 1);
}

/** @brief initialize the Jacobian matrix for KS.
 * 
 * @param[out] v [N-1, N/2+1] dimensional complex array.
 * the last row will not be used, so it is all zero. */
void KsM1::initJ(dcomp *v){
  memset(v + N/2 + 1, 0, (N-2) * (N/2+1)*sizeof(dcomp)); //set the 2nd to (N-1)th row to be zero.
  v[N/2+2] = dcomp(1.0, 0);
  for(int i = 0; i < N/2 - 2; i++){
    v[(2*i+2)*(N/2+1) + i + 2] = dcomp(1.0, 0);
    v[(2*i+3)*(N/2+1) + i + 2] = dcomp(0, 1.0);
  }
}

KsM1::dvec KsM1::velo(dvec &a0){
  dvec aa(N-3); 		/* not used, just in orde to use function initKs() */
  dcvec v(N/2+1), Nv(N/2+1);
  FFT rp, p;
  initKs(&a0[0], &v[0], &aa[0], rp, p);
  calNL(&v[0], &Nv[0], p, rp); 	/* calculate the nonlinear term */
  
  for(int i = 0; i < Nv.size(); i++) Nv[i] += coe.L[i] * v[i]; /* add the linear term */
  
  dvec velo(cv2rv(Nv));

  cleanKs(rp, p);
  
  return velo;
}

/** @brief convert the complex vector to real and imaginary part for M1 slice.
 *  
 *  @param[in] a0 state complex vector of size N/2+1
 *  @return real vector of size N-3
 * */
KsM1::dvec KsM1::cv2rv(const dcvec &a0){
  dvec v(N-3);
  v[0] = a0[1].real();
  for(int i = 0; i < N/2 - 2; i++){
    v[2*i+1] = a0[i+2].real();
    v[2*i+2] = a0[i+2].imag();
  }
  
  return v;
}

/**
 * @param[in] x0 template point chosen for Poincare section.
 * @param[in] a0 staring point, should be close to poincare section.
 */
KsM1::dvec KsM1::ks2poinc(dvec &x0, dvec &a0){
  dvec v0(velo(x0));
  int2p.v0 = v0;
  dvec a(a0);
  a.push_back(0.0);
  double u0 = 0;
  for(int i = 0; i < a0.size(); i++) u0 += v0[i] * (a0[i] - x0[i]);
  size_t steps = integrate(int2p, a, u0, 0.0, fabs(u0)/100);
  a.push_back(steps);
  return a;
}
/* --------------------------------------------------------- */


void KsM1::Int2p::operator() (const dvec &x, dvec &dxdt, const double /* u */){
  dvec a(x.begin(), x.end()-1); // the first N-3 elements.
  dvec va(ks.velo(a)); // velocity of state point.
  double nor = 0;
  for(int i = 0; i < va.size(); i++) nor += va[i] * v0[i];
  for(int i = 0; i< va.size(); i++) dxdt[i] = va[i]/nor;
  dxdt[ks.N-3] = 1.0/nor;
}
