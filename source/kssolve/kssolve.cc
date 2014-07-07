#include "kssolve.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <fftw3.h>
//#include <omp.h>

/****    global variable definition.   *****/
const double PI=3.14159265358979323;

/*****    structure definition.        ******/
typedef std::complex<double> dcomp;

/**
   Structure for convenience of rfft.
   For forward real Fourier transform, 'in' is real
   and 'out' is complex.
   For inverse real Fourier transform, the situation reverses.
*/
typedef struct{
  fftw_plan p;
  fftw_complex *c; // complex array.
  double *r; // real array
} FFT;

/**************         function declaration       *****************/
void
initFFT(FFT &p, const int N, int a);
void
freeFFT(FFT &p);
void
dot(double* a, double* b, int N);
void 
rfft(double *u, int N, const FFT &p);
void 
irfft(const dcomp *u, int N, const FFT &rp);
void 
calNL(dcomp *g, dcomp *u, dcomp *Nv, int N, const FFT &p, const FFT &rp);
void
calNL(dcomp *g, dcomp *u, dcomp *Nv, int N, const FFT *p, const FFT *rp );
void  
onestep(dcomp *g, dcomp *v, double *E,  double *E2,  double *Q, double *f1, double *f2, double *f3, int N, FFT &p, FFT &rp);
void  
onestep(dcomp *g, dcomp *v, double *E, double *E2, double *Q, double *f1, double *f2, double *f3, int N, FFT *p, FFT *rp);
void
calcoe(double h, double d,  double *E, double *E2, dcomp *g, double *Q, double *f1, double *f2, double *f3, int N, const int M = 16);
void
initJ(dcomp *v, int N);

//////////////////////////////////////////////////////////////////////
void
ksfj(double *a0, double d, double h, int nstp, int np, int nqr, double *aa, double *daa){

  double E[N/2+1], E2[N/2+1], Q[N/2+1],f1[N/2+1],f2[N/2+1],f3[N/2+1];
  dcomp g[N/2+1];
  calcoe(h,d,E,E2,g,Q,f1,f2,f3,N);
  
  FFT rp[N-1], p[N-1];
  for(int i = 0; i < N-1; i++) initFFT(rp[i], N, -1);
  for(int i = 0; i < N-1; i++) initFFT(p[i], N, 1);
    
  // the size of aa should be (N-2)*(nstp/np)
  for(int i = 0; i < N-2; ++i) aa[i]=a0[i];
  int ix1 = 1; // index counter for aa.
  // the size of daa should be (N-2)^2 * nstp/nqr
  int ix2 = 0; // index counter for daa.

  // initialize initial data.
  dcomp v[(N/2+1) * (N-1)]; // wave mode 0 and N/2 are both zero.
  v[0] = dcomp(0, 0);
  v[N/2] = dcomp(0, 0);
  for(int i = 0; i < N/2 - 1; i++) v[i+1] = dcomp(a0[2*i], a0[2*i+1]);
  initJ(v, N);

  for (int i = 0; i < nstp; ++i){
    
    onestep(g, v, E, E2, Q, f1, f2, f3, N, p, rp);
    
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
      initJ(v, N); // Initialize Jacobian again.
    }
  }
  for(int i = 0; i < N - 1; i++) freeFFT(p[i]);
  for(int i = 0; i < N - 1; i++) freeFFT(rp[i]);

}


void
ksf(double *a0, double d, double h, int nstp, int np, double *aa){

  double E[N/2+1], E2[N/2+1], Q[N/2+1],f1[N/2+1],f2[N/2+1],f3[N/2+1];
  dcomp g[N/2+1];
  calcoe(h,d,E,E2,g,Q,f1,f2,f3,N);
  
  FFT rp, p;
  initFFT(rp, N, -1);
  initFFT(p, N, 1);
    
  // the size of aa should be (N-2)*(nstp/np+1)
  for(int i = 0; i < N-2; ++i) aa[i]=a0[i];
  int ix = 1;
  
  // initialize initial data.
  dcomp v[N/2+1]; // wave mode 0 and N/2 are both zero.
  v[0] = dcomp(0, 0);
  v[N/2] = dcomp(0, 0);
  for(int i = 0; i < N/2 - 1; i++) v[i+1] = dcomp(a0[2*i], a0[2*i+1]);
  
  for (int i=0; i<nstp; ++i){
    onestep(g, v, E, E2, Q, f1, f2, f3, N, p, rp);
    if( (i+1)%np == 0 ) {
      for(int i = 0; i< N/2 - 1; i++) {
	aa[ix*(N-2)+ 2*i] = v[i+1].real();
	aa[ix*(N-2)+ 2*i + 1] = v[i+1].imag();
      }
      ix++;
    }
  }
  freeFFT(p);
  freeFFT(rp);

}

/**
   @brief initialize the Jacobian matrix for KS.

   @parameter[out] v [N-1, N/2+1] dimensional complex array.
*/
void
initJ(dcomp *v, int N){
  memset(v + N/2 + 1, 0, (N-2) * (N/2+1)*sizeof(dcomp)); //set the 2nd to (N-1)th row to be zero.
  for(int i = 0; i < N/2 - 1; i++){
    v[(2*i+1)*(N/2+1) + i + 1] = dcomp(1.0, 0);
    v[(2*i+2)*(N/2+1) + i + 1] = dcomp(0, 1.0);
  }
}

void
calcoe(double h, double d,  double *E, double *E2, dcomp *g, double *Q, double *f1, double *f2, double *f3, int N, const int M /* = 16 */){

  double k[N/2+1], L[N/2+1];
  for(int i = 0; i < N/2 + 1; i++){
    k[i] = i * 2 * PI / d;
    L[i] = k[i]*k[i] - k[i]*k[i]*k[i]*k[i];    
    E[i] = exp(L[i] * h);
    E2[i] = exp(L[i] * h / 2);    
    g[i] = 0.5 * N * dcomp(0, 1) * k[i] ;
  }
  g[N/2] = dcomp(0, 0); // the N/2 mode has zero first derivative at grid point.
  dcomp r[M];
  for(int i = 0; i < M; i++) r[i] = exp(dcomp(0, PI*(i+0.5)/M)); 
    
  dcomp LR, LR3, cf1, cf2, cf3, cQ;
  // In the equiverlent matlab code, LR is a matrix, and then 
  // Q, f1, f2, f3 are all column vectors after averaging over matrix rows. 
  for(int i = 0; i < N/2 + 1; ++i){ // iteration over column
    for(int j = 0; j < M; ++j){ // iteration over row
      LR = h * L[i] + r[j]; // each row element of LR;
      LR3 = LR * LR * LR;
      // each element of the matrix
      cQ = h * ( exp(LR/2.0) - 1.0 ) / LR;
      cf1 = h * (-4.0 - LR + exp(LR)*(4.0 - 3.0 * LR + LR*LR)) / LR3;
      cf2 = h * (2.0 + LR + exp(LR)*(LR-2.0)) / LR3;
      cf3 = h * (-4.0 - 3.0*LR -LR*LR + exp(LR)*(4.0 - LR)) / LR3;
      // sum over the row
      Q[i] += cQ.real();
      f1[i] += cf1.real();
      f2[i] += cf2.real();
      f3[i] += cf3.real();
    }
  }
  
  // calculate the averarge.
  for(int i = 0; i < N/2 +1; ++i){
    Q[i] = Q[i] / M;
    f1[i] = f1[i] / M;
    f2[i] = f2[i] / M;
    f3[i] = f3[i] / M;
  }
}

/**
   @brief one time step integration when Jacobian is also desired.

   @parameter[in,out] v [N-1, N/2+1] row-wise complex array.
*/
void  
onestep(dcomp *g, dcomp *v, double *E, double *E2, double *Q, double *f1, double *f2, double *f3, int N, FFT *p, FFT *rp){
  const int size = (N - 1) * (N/2 + 1); 
  dcomp Nv[size], Na[size], Nb[size], Nc[size];
  dcomp a[N-1][N/2+1], b[N-1][N/2+1],  c[N-1][N/2+1];
	
  calNL(g, v, Nv, N, p, rp);
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      a[i][j] = E2[j]*v[i*(N/2+1) + j] + Q[j]*Nv[i*(N/2+1) + j];
  calNL(g, a[0], Na, N, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      b[i][j] = E2[j]*v[i*(N/2+1) + j] + Q[j]*Na[i*(N/2+1) + j];
  calNL(g, b[0], Nb, N, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j = 0; j < N/2+1; j++)
      c[i][j] = E2[j]*a[i][j] + Q[j]*(2.0*Nb[i*(N/2+1) + j]
					       - Nv[i*(N/2+1) + j]);
  calNL(g, c[0], Nc, N, p, rp);
  
  for( int i = 0; i < N - 1; i++)
    for(int j =0; j< N/2+1; j++)
      v[i*(N/2+1) + j] = E[j]*v[i*(N/2+1) + j] + Nv[i*(N/2+1) + j]*f1[j] + 2.0*f2[j]
	*(Na[i*(N/2+1) + j]+Nb[i*(N/2+1) + j]) + Nc[i*(N/2+1) + j]*f3[j];
}



/**
   @parameter[in, out] v state vector, [N/2+1,1] complex array
*/
void  
onestep(dcomp *g, dcomp *v, double *E, double *E2, double *Q, double *f1, double *f2, double *f3, int N, FFT &p, FFT &rp){
  dcomp Nv[N/2+1], a[N/2+1], Na[N/2+1], b[N/2+1], Nb[N/2+1], c[N/2+1], Nc[N/2+1];
	
  calNL(g, v, Nv, N, p, rp);	
  for (int i = 0; i < N/2 + 1; i++) a[i] = E2[i]*v[i] + Q[i]*Nv[i];
  calNL(g, a, Na, N, p, rp);	
	
  for (int i = 0; i < N/2 + 1; i++) b[i] = E2[i]*v[i] + Q[i]*Na[i];
  calNL(g, b, Nb, N, p, rp);
	
  for (int i = 0; i < N/2 + 1; i++) c[i] = E2[i]*a[i] + Q[i]*(Nb[i]*2.0-Nv[i]);
  calNL(g, c, Nc, N, p, rp);
	
  for (int i = 0; i < N/2 + 1; i++) v[i] = E[i]*v[i] + Nv[i]*f1[i] + 2.0*f2[i]
				      *(Na[i]+Nb[i]) + Nc[i]*f3[i];
  
}

/**
   @brief calculate the nonliear term for one dimensional array.
   
   @parameter[out] Nv [N/2+1,1] complex array
*/
void 
calNL(dcomp *g, dcomp *u, dcomp *Nv, int N, const FFT &p, const FFT &rp){
  
  irfft(u, N, rp); // F^{-1}
  for(int i = 0; i < N; i++) rp.r[i] = rp.r[i] * rp.r[i]; // get (F^{-1})^2 
  rfft(rp.r, N, p); 
  
  for (int i = 0; i < N/2 + 1; i++) 
    Nv[i] =  g[i] * dcomp(p.c[i][0], p.c[i][1]);
  
}

/**
   @brief calculate the nonlinear term when the Jabobian is also desired.
   
   @parameter[out] Nv returned nonliear term. [N-1, N/2+1] row-wise complex array.
   @parameter[in]  u  input data. [N-1, N/2+1] row-wise complex array
*/
void
calNL(dcomp *g, dcomp *u, dcomp *Nv, int N, const FFT *p, const FFT *rp ){

	// split the irfft part since operations that follows depend on 
	// result of rp[0].
	for(int i = 0; i < N - 1; i++) irfft(u+i*(N/2+1), N, rp[i]);
	
	// make a copy of the first transformed vector since the square 
	// operation is inline, it will destroy rp[0].r[j].
	double tmp[N];
	memcpy(tmp, rp[0].r, N * sizeof(double));

	for(int i = 0; i < N - 1; i++){
	  for( int j = 0; j < N; j++) 
	    rp[i].r[j] = tmp[j] *rp[i].r[j];
		
	  rfft(rp[i].r, N, p[i]);
	  for(int j = 0; j < N/2 + 1; j++){
	    if(i == 0) Nv[i*(N/2+1) + j] = g[j] * dcomp(p[i].c[j][0], p[i].c[j][1]); 
	    else Nv[i*(N/2+1) + j] = 2.0 * g[j] * dcomp(p[i].c[j][0], p[i].c[j][1]); 
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
irfft(const dcomp *u, int N, const FFT &rp){
  
  memcpy(rp.c, u, (N/2+1) * sizeof(fftw_complex));
  fftw_execute(rp.p);
  for(int i = 0; i < N; i++) rp.r[i] /= N;
}

/**
   r2c transform will destroy input data, so must assign new array double in[N]
   
*/
void 
rfft(double *u, int N, const FFT &p){
  
  memcpy(p.r, u, N * sizeof(double));
  fftw_execute(p.p);
}

/**
   dot product of two arrays.
*/
void
dot(double* a, double* b, int N){
	for(int i=0; i<N; ++i) b[i] *= a[i];
}

/**
   @brief construction of rfft/irfft plan.
   
   @parameter a indicator the direction of fft. a = 1 : forward; a = -1 : backward.
*/
void
initFFT(FFT &p, const int N, int a){
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
   destroy rfft/irfft plan.
*/
void
freeFFT(FFT &p){
  fftw_destroy_plan(p.p);
  fftw_free(p.c);
  fftw_free(p.r);
}
