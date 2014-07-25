#include "kssolve.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
//#include <complex>
//#include <fftw3.h>
//#include <omp.h>

  

/* ==========================================================================
 *                              Class Ks
 * ==========================================================================*/
/*------------------          constructors and destructor  ---------------- */

Ks::Ks(int N, double d, double h) : N(N), d(d), h(h){
  /* allocation of coefficients and initialize them to be zero. */
  coe.E = new double[N/2+1]();
  coe.E2 = new double[N/2+1]();
  coe.k = new double[N/2+1]();
  coe.L = new double[N/2+1]();
  coe.Q = new double[N/2+1]();
  coe.f1 = new double[N/2+1]();
  coe.f2 = new double[N/2+1]();
  coe.f3 = new double[N/2+1]();
  coe.g = new dcomp[N/2+1](); 
  /* calculate coefficients */
  calcoe();
}

Ks::Ks(const Ks &x) : N(x.N), d(x.d), h(x.h){}

Ks::~Ks(){
  delete[] coe.E;
  delete[] coe.E2;
  delete[] coe.k;
  delete[] coe.L;
  delete[] coe.Q;
  delete[] coe.f1;
  delete[] coe.f2;
  delete[] coe.f3;
  delete[] coe.g;
}

/* assignment changes nothing. */
Ks & Ks::operator=(const Ks &x){
  return *this;
}

/*---------------              member functions         --------------------*/

void Ks::kssolve(double *a0, int nstp, int np, double *aa){

  FFT rp, p; 
  dcomp v[N/2+1]; // wave mode 0 and N/2 are both zero.   
  initKs(a0, v, aa, rp, p);

  int ix = 1; 
  for (int i=0; i<nstp; ++i){
    onestep(v, p, rp);
    if( (i+1)%np == 0 && i != nstp - 1) {
      for(int i = 0; i< N/2 - 1; i++) {
	aa[ix*(N-2)+ 2*i] = v[i+1].real();
	aa[ix*(N-2)+ 2*i + 1] = v[i+1].imag();
      }
      ix++;
    }
  }
  cleanKs(rp, p);
}

void
Ks::kssolve(double *a0, int nstp, int np, int nqr, double *aa, double *daa){

  FFT rp[N-1], p[N-1];
  dcomp v[(N/2+1) * (N-1)]; // wave mode 0 and N/2 are both zero.
  initKs(a0, v, aa, rp, p);

  int ix1 = 1; // index counter for aa.
  // the size of daa should be (N-2)^2 * nstp/nqr
  int ix2 = 0; // index counter for daa.
  for (int i = 0; i < nstp; ++i){
    onestep(v, p, rp);
    if( (i+1)%np == 0 && i != nstp - 1) {
      for(int i = 0; i< N/2 - 1; i++) {
	aa[ix1*(N-2)+ 2*i] = v[i+1].real();
	aa[ix1*(N-2)+ 2*i + 1] = v[i+1].imag();
      }
      ix1++;
    }
    
    if((i+1)%nqr == 0){
      for(int i = 0; i < N - 2; i++){
	for(int j = 0; j < N/2 -1; j++){
	  // reading from v starts from second row  and second column. 
	  daa[ix2*(N-2)*(N-2) + i * (N - 2) + 2*j] = v[(N/2+1)*(i+1) + j+1].real();
	  daa[ix2*(N-2)*(N-2) + i * (N - 2) + 2*j + 1] = v[(N/2+1)*(i+1) + j+1].imag();
	}
      }
      ix2++;
      initJ(v); // Initialize Jacobian again.
    }
  }
  cleanKs(rp, p);
}


/** @brief Initialize the KS system without Jacobian
 *  
 */

void Ks::initKs(double *a0, dcomp *v, double *aa, FFT &rp, FFT &p){
  // the size of aa should be (N-2)*(nstp/np)
  for(int i = 0; i < N-2; i++) aa[i] = a0[i];
  
  // initialize initial data.
  v[0] = dcomp(0, 0);
  v[N/2] = dcomp(0, 0);
  for(int i = 0; i < N/2 - 1; i++) v[i+1] = dcomp(a0[2*i], a0[2*i+1]);
  
  /* FFT initiate. */
  initFFT(rp, -1);
  initFFT(p, 1);
  
}

/** @brief Initialize the KS system with calculating Jacobian */
void Ks::initKs(double *a0, dcomp *v, double *aa, FFT *rp, FFT *p){
  // the size of aa should be (N-2)*(nstp/np)
  for(int i = 0; i < N-2; i++) aa[i] = a0[i];

  //initialize initial data.
  v[0] = dcomp(0, 0);
  v[N/2] = dcomp(0, 0);
  for(int i = 0; i < N/2 - 1; i++) v[i+1] = dcomp(a0[2*i], a0[2*i+1]);
  initJ(v);
  
  for(int i = 0; i < N-1; i++) initFFT(rp[i], -1);
  for(int i = 0; i < N-1; i++) initFFT(p[i], 1);
}


/** @brief initialize the Jacobian matrix for KS.
 * 
 * @param[out] v [N-1, N/2+1] dimensional complex array. */
void Ks::initJ(dcomp *v){
  memset(v + N/2 + 1, 0, (N-2) * (N/2+1)*sizeof(dcomp)); //set the 2nd to (N-1)th row to be zero.
  for(int i = 0; i < N/2 - 1; i++){
    v[(2*i+1)*(N/2+1) + i + 1] = dcomp(1.0, 0);
    v[(2*i+2)*(N/2+1) + i + 1] = dcomp(0, 1.0);
  }
}

void Ks::cleanKs(FFT &rp, FFT &p){
  /* free the FFT resource. */
  freeFFT(p);
  freeFFT(rp);
  fftw_cleanup();
}

void Ks::cleanKs(FFT *rp, FFT *p){
  for(int i = 0; i < N - 1; i++) freeFFT(p[i]);
  for(int i = 0; i < N - 1; i++) freeFFT(rp[i]);
  
  //fftw_cleanup frees heap reservation of FFTW. It may not affect this C++
  //program, but it can cause unpredictable behavior in MEX and Ctypes,
  //so just keep it.
  fftw_cleanup();
}

void Ks::calcoe(const int M /* = 16 */){

  for(int i = 0; i < N/2 + 1; i++){
    coe.k[i] = i * 2 * PI / d;
    coe.L[i] = coe.k[i]*coe.k[i] - coe.k[i]*coe.k[i]*coe.k[i]*coe.k[i];    
    coe.E[i] = exp(coe.L[i] * h);
    coe.E2[i] = exp(coe.L[i] * h / 2);    
    coe.g[i] = 0.5 * N * dcomp(0, 1) * coe.k[i] ;
  }
  coe.g[N/2] = dcomp(0, 0); // the N/2 mode has zero first derivative at grid point.
  dcomp r[M];
  for(int i = 0; i < M; i++) r[i] = exp(dcomp(0, PI*(i+0.5)/M)); 
    
  dcomp LR, LR3, cf1, cf2, cf3, cQ;
  // In the equiverlent matlab code, LR is a matrix, and then 
  // Q, f1, f2, f3 are all column vectors after averaging over matrix rows. 
  for(int i = 0; i < N/2 + 1; ++i){ // iteration over column
    for(int j = 0; j < M; ++j){ // iteration over row
      LR = h * coe.L[i] + r[j]; // each row element of LR;
      LR3 = LR * LR * LR;
      // each element of the matrix
      cQ = h * ( exp(LR/2.0) - 1.0 ) / LR;
      cf1 = h * (-4.0 - LR + exp(LR)*(4.0 - 3.0 * LR + LR*LR)) / LR3;
      cf2 = h * (2.0 + LR + exp(LR)*(LR-2.0)) / LR3;
      cf3 = h * (-4.0 - 3.0*LR -LR*LR + exp(LR)*(4.0 - LR)) / LR3;
      // sum over the row
      coe.Q[i] += cQ.real();
      coe.f1[i] += cf1.real();
      coe.f2[i] += cf2.real();
      coe.f3[i] += cf3.real();
    }
  }
  
  // calculate the averarge.
  for(int i = 0; i < N/2 +1; ++i){
    coe.Q[i] = coe.Q[i] / M;
    coe.f1[i] = coe.f1[i] / M;
    coe.f2[i] = coe.f2[i] / M;
    coe.f3[i] = coe.f3[i] / M;
  }
}

/**
   @parameter[in, out] v state vector, [N/2+1,1] complex array
*/
void Ks::onestep(dcomp *v, FFT &p, FFT &rp){

  dcomp Nv[N/2+1], a[N/2+1], Na[N/2+1], b[N/2+1], Nb[N/2+1], c[N/2+1], Nc[N/2+1];
	
  calNL( v, Nv, p, rp);	
  for (int i = 0; i < N/2 + 1; i++) a[i] = coe.E2[i]*v[i] + coe.Q[i]*Nv[i];
  calNL( a, Na, p, rp);	
	
  for (int i = 0; i < N/2 + 1; i++) b[i] = coe.E2[i]*v[i] + coe.Q[i]*Na[i];
  calNL( b, Nb, p, rp);
	
  for (int i = 0; i < N/2 + 1; i++) c[i] = coe.E2[i]*a[i] + coe.Q[i]*(Nb[i]*2.0-Nv[i]);
  calNL( c, Nc, p, rp);
	
  for (int i = 0; i < N/2 + 1; i++) v[i] = coe.E[i]*v[i] + Nv[i]*coe.f1[i] + 2.0*coe.f2[i]
				      *(Na[i]+Nb[i]) + Nc[i]*coe.f3[i];
  
}


/** @brief one time step integration when Jacobian is also desired.
 * 
 * @parameter[in,out] v [N-1, N/2+1] row-wise complex array. */
void Ks::onestep(dcomp *v, FFT *p, FFT *rp){
  const int size = (N - 1) * (N/2 + 1); 
  dcomp Nv[size], Na[size], Nb[size], Nc[size];
  dcomp a[N-1][N/2+1], b[N-1][N/2+1],  c[N-1][N/2+1];
	
  calNL( v, Nv, p, rp);
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      a[i][j] = coe.E2[j]*v[i*(N/2+1) + j] + coe.Q[j]*Nv[i*(N/2+1) + j];
  calNL( a[0], Na, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      b[i][j] = coe.E2[j]*v[i*(N/2+1) + j] + coe.Q[j]*Na[i*(N/2+1) + j];
  calNL( b[0], Nb, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      c[i][j] = coe.E2[j]*a[i][j] + coe.Q[j]*(2.0*Nb[i*(N/2+1) + j]
				      - Nv[i*(N/2+1) + j]);
  calNL(c[0], Nc, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j =0; j< N/2+1; j++)
      v[i*(N/2+1) + j] = coe.E[j]*v[i*(N/2+1) + j] + Nv[i*(N/2+1) + j]*coe.f1[j] + 2.0*coe.f2[j]
	*(Na[i*(N/2+1) + j]+Nb[i*(N/2+1) + j]) + Nc[i*(N/2+1) + j]*coe.f3[j];
}

/** @brief calculate the nonliear term for one dimensional array.
 * 
 * @parameter[out] Nv [N/2+1,1] complex array */ 
void Ks::calNL(dcomp *u, dcomp *Nv, const FFT &p, const FFT &rp){
  
  irfft(u, rp); // F^{-1}
  for(int i = 0; i < N; i++) rp.r[i] = rp.r[i] * rp.r[i]; // get (F^{-1})^2 
  rfft(rp.r, p); 
  
  for (int i = 0; i < N/2 + 1; i++) 
    Nv[i] =  coe.g[i] * dcomp(p.c[i][0], p.c[i][1]);
  
}


/* @brief calculate the nonlinear term when the Jabobian is also desired.
 * 
 * @parameter[out] Nv returned nonliear term. [N-1, N/2+1] row-wise complex array.
 * @parameter[in]  u  input data. [N-1, N/2+1] row-wise complex array
 */
void Ks::calNL(dcomp *u, dcomp *Nv, const FFT *p, const FFT *rp ){

  // split the irfft part since operations that follows depend on 
  // result of rp[0]. 
  for(int i = 0; i < N - 1; i++) irfft(u+i*(N/2+1), rp[i]); /* only N-2 vectors.
							     */
	
  // make a copy of the first transformed vector since the square 
  // operation is inline, it will destroy rp[0].r[j].
  double tmp[N];
  memcpy(tmp, rp[0].r, N * sizeof(double));

  for(int i = 0; i < N - 1; i++){
    for(int j = 0; j < N; j++)
      rp[i].r[j] = tmp[j] * rp[i].r[j];
    
    rfft(rp[i].r, p[i]);
    for(int j = 0; j < N/2 + 1; j++){
      if(i == 0) Nv[i*(N/2+1) + j] = coe.g[j] * dcomp(p[i].c[j][0], p[i].c[j][1]); 
      else Nv[i*(N/2+1) + j] = 2.0 * coe.g[j] * dcomp(p[i].c[j][0], p[i].c[j][1]); 
    }
  }
}

/**
   inverse real fft for one dimensional array.
   inverse transform will overwirte the input data.
   FFTW for inverse transform doesn't normlize the outcome,
   so we need to normalize the result explicitly.
*/
void 
Ks::irfft(const dcomp *u, const FFT &rp){
  memcpy(rp.c, u, (N/2+1) * sizeof(fftw_complex));

  fftw_execute(rp.p);
  for(int i = 0; i < N; i++) rp.r[i] /= N;
}


/** r2c transform will destroy input data, so must assign new array double in[N] */
void Ks::rfft(double *u, const FFT &p){
  
  memcpy(p.r, u, N * sizeof(double));
  fftw_execute(p.p);
}

/** @brief construction of rfft/irfft plan.
 *    
 * @parameter a indicator the direction of fft. a = 1 : forward; a = -1 : backward. */

void Ks::initFFT(FFT &p, int a){
  p.r = (double*) fftw_malloc(sizeof(double) * N);
  p.c= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1));
  if(a == 1)
    p.p = fftw_plan_dft_r2c_1d(N, p.r, p.c, FFTW_MEASURE);
  else if(a == -1)
    p.p = fftw_plan_dft_c2r_1d(N, p.c, p.r, FFTW_MEASURE);
  else{ 
    printf("please indicate the correct FFT direction.\n");
    exit(1);
  }
}

/**
 * @brief destroy rfft/irfft plan. 
 */

void
Ks::freeFFT(FFT &p){
  fftw_destroy_plan(p.p);
  fftw_free(p.c);
  fftw_free(p.r);
}


