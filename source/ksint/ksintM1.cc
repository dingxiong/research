#include "ksintM1.hpp"
#include <cmath>

using Eigen::NoChange;

/* ================================================== 
 *                    Class : KS 1st mode integrator
 * ==================================================*/

/* --------------------    constructors  -------------------- */
KSM1::KSM1(int N, double h, double d) : KS(N, h, d) {}
KSM1::KSM1(const KSM1 &x) : KS(x) {}
KSM1 & KSM1::operator=(const KSM1 &x) { return *this; }


/* -------------------- member functions -------------------- */

/** @brief calculate the nonlinear term
 * (a_1 - 1)*(q_k^2 - q_k^4)a_k - iq_k/2 fft(ifft^2(a)) + iq_k/2*a*real(fft(ifft(a))_1) 
 **/
void KSM1::NL(KSfft &f){
  double a1hat = f.vc1(1).real();  
  ifft(f);
  f.vr2 = f.vr2 * f.vr2;
  fft(f);
  f.vc3 = (a1hat - 1)*L*f.vc1 + a1hat*G*f.vc3 - f.vc3(1).real()*G*f.vc1;
}

void KSM1::jNL(KSfft &f){
  double a1 = f.vc1(1,0).real();
  ArrayXd y1 = f.vc1.block(1,1,1,N-2).real();
  ifft(f);
  ArrayXd tmp = f.vr2.col(0);
  f.vr2 = tmp.matrix().asDiagonal()*f.vr2.matrix();
  fft(f);
  double fv1 = f.vc3.row(1).real();
  /* (q_k^2-q_k^4)*a - iq_k/2*\sum a_m a_{k-m} */
  f.vc3.col(0) = L*f.vc1.col(0) + G*f.vc3.col(0);
  /* (q_k^2-q_k^4)*y - iq_k/2*\sum a_m y_{k-m} */
  f.vc3.rightCols(N-2) = L*f.vc1.rightCols(N-2) + 2.0*G*f.vr3.rightCols(N-2);
  
  f.vc3.rightCols(N-2) = (a1-1)*f.vc3.rightCols(N-2) - G * f.vc1.rightCols(N-2)*fv1(0)
    +y1*f.vr3.col(0) - 2.0*G*fv1.tail(N-2);
  f.vc3.col(0) = a1*f.vc3.col(0) - L*f.vc1.col(0) -fv1(0)*G*f.vc1;
}

KSM1::KSat KSM1::intg(const ArrayXd &a0, size_t nstp, size_t np){
  if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);} 
  Fv.vc1 = R2C(a0); 
  Fv.vc1(1) = dcp(Fv.vc1(1).real(), 0.0); //force the imag of 1st mode to be zero
  ArrayXXd aa(N-2, nstp/np+1);  aa.col(0) = a0;
  ArrayXd tt(nstp/np+1); tt(0) = 0.0;

  double t = 0;
  for(size_t i = 1; i < nstp +1; i++){
    NL(Fv); Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3;
    NL(Fa); Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;
    NL(Fb); Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);
    NL(Fc);
    Fv.vc1 = E*Fv.vc1 + Fv.vc3*f1 + 2.0*(Fa.vc3+Fb.vc3)*f2 + Fc.vc3*f3;    
    t += Fv.vc1(1).real() * h;

    if( 0 == i%np ) {
      aa.col(i/np) = C2R(Fv.vc1);
      tt(i/np) = t;
    }
  }
  
  KSat at; 
  at.aa = aa; at.tt = tt;

  return at;
}

KSM1::KSat KSM1::intg2(const ArrayXd &a0, double T, size_t np){
  const size_t cell = 1000;
  if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);} 
  Fv.vc1 = R2C(a0); 
  Fv.vc1(1) = dcp(Fv.vc1(1).real(), 0.0); //force the imag of 1st mode to be zero
  ArrayXXd aa(N-2, cell);  aa.col(0) = a0;
  ArrayXd tt(cell); tt(0) = 0.0;
  
  double t = 0;
  size_t i = 1; // record the integration steps.
  size_t ix = 1; // record the #columns of aa.
  double lastT = 0; // record the last time recorded in tt.
  while(lastT < T){
    NL(Fv); Fa.vc1 = E2*Fv.vc1 + Q*Fv.vc3;
    NL(Fa); Fb.vc1 = E2*Fv.vc1 + Q*Fa.vc3;
    NL(Fb); Fc.vc1 = E2*Fa.vc1 + Q*(2.0*Fb.vc3 - Fv.vc3);
    NL(Fc);
    Fv.vc1 = E*Fv.vc1 + Fv.vc3*f1 + 2.0*(Fa.vc3+Fb.vc3)*f2 + Fc.vc3*f3;    
    t += Fv.vc1(1).real() * h;


    if( 0 == i%np ) {
      int m = aa.cols();
      if(i/np > m - 1) {
	aa.conservativeResize(NoChange, m+cell);
	tt.conservativeResize(m+cell);
      }
      aa.col(i/np) = C2R(Fv.vc1);
      tt(i/np) = t; lastT = t;
      ix++;
    }
    
    i++;
  }
  
  KSat at; 
  at.aa = aa.leftCols(ix); at.tt = tt.head(ix);

  return at;
}

KSM1::intgj(const ArrayXd a0,  size_t nstp, size_t np, size_t nqr){
  if( N-2 != a0.rows() ) {printf("dimension error of a0\n"); exit(1);}
  ArrayXXd v0(N-2, N-1);
  v0 << a0, MatrixXd::Identity(N-2, N-2);
  jFv.vc1 = R2C(v0);
  ArrayXXd aa(N-2, nstp/np+1); aa.col(0) = a0;
  ArrayXXd daa((N-3)*(N-3), nstp/nqr);
  for(size_t i = 1; i < nstp + 1; i++){
    
  }
}
