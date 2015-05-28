/* to comiple:
 * g++ -O3 lyapunov.cc -lcqcgl1d -L../lib -I../include -I/usr/include/eigen3
 * -std=c++0x -lfftw3
 */
#include "cqcgl1d.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/FFT>
#include <complex>

using namespace std;
using namespace Eigen;
typedef std::complex<double> dcp;
const int N = 256; 
const double L = 50;
const double h = 0.01;
const int nstp = 1000;
const int nqr = 10;
const int trans = 50000;
const int Maxit = 800;
int main(){
  //==================================================
  #if 1
  Cqcgl1d cgl(N, L, h); 
  ArrayXd x = ArrayXd::LinSpaced(N, 1, N) / N * L - L/2.0;
  ArrayXcd a0(N); a0.real() = (-x*x/8.0).exp(); a0.imag() = ArrayXd::Zero(N);
  FFT<double> fft;
  ArrayXcd fa = (fft.fwd(a0.matrix())).array();
  ArrayXd v0 = cgl.C2R(fa);
  /* transient process */
  ArrayXXd aa = cgl.intg(v0, trans, trans);
  v0 = aa.rightCols(1);
  MatrixXd Q0 = MatrixXd::Identity(2*N, 2*N);
  ArrayXXd R(2*N, Maxit*nstp/nqr);

  HouseholderQR<MatrixXd> qr(2*N, 2*N);
  Cqcgl1d::CGLaj aj = cgl.intgj(v0, nstp, nstp, nqr);
  for(size_t i = 0; i < Maxit; i++){
    printf("---------- i = %zd ----------\n ", i);
    size_t m = aj.daa.cols(); cout << m << endl;
    for(size_t j = 0; j < m; j++){
      Map<MatrixXd> Jac(&(aj.daa(0,j)), 2*N, 2*N);
      qr.compute(Jac*Q0);
      Q0 = qr.householderQ(); 
      R.col(i*nstp/nqr+j) = qr.matrixQR().diagonal();
    }
    aj = cgl.intgj(aj.aa.rightCols(1), nstp, nstp, nqr);
  }
  R = R.abs().log()/(h*nqr);

  ofstream fp;
  fp.open("R.bin", ios::binary);
  if(fp.is_open()) fp.write((char*)&(R(0,0)), R.cols()*R.rows()*sizeof(double));
  fp.close();

  #endif  
  //==================================================

  return 0;
}
