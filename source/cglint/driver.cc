/* to comiple:
 * g++ -O3 driver.cc -lcqcgl1d -L../lib -I../include -I/usr/include/eigen3
 * -std=c++0x -lfftw3
 */
#include "cqcgl1d.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>

using namespace std;
using namespace Eigen;
typedef std::complex<double> dcp;
const int N = 256; 
const int L = 50;

int main(){
  //////////////////////////////////////////////////
  #if 0
  ArrayXcd A0(N) ;
  // prepare Gaussian curve initial condition
  for(int i = 0; i < 1024; i++) {
    double x = (double)i/N*L - L/2.0; 
    A0(i) = dc( exp(-x*x/8.0), 0 );
  }
  Cqcgl1d cgl;
  int nstp = 1000000; int nqr = 10;
  int npre = 100000;
  MatrixXcd At = cgl.intg(A0, npre, npre);
  MatrixXcd AA = cgl.intg(At.rightCols(1), nstp, nqr);

  ofstream fp;
  fp.open("aa.bin", ios::binary);
  if(fp.is_open()) fp.write( (char*)&(AA(0,0)),  AA.cols()*AA.rows()*sizeof(dc) );
  fp.close();
  
  cout << AA.rows() << 'x' << AA.cols() << endl << "--------------" << endl;
  //cout << AA.col(5).head(10) << endl;
  //cout << A0 << endl;
  #endif
  //////////////////////////////////////////////////
  
  //==================================================
  #if 1
  ArrayXd x = ArrayXd::LinSpaced(N, 1, N) / N * L - L/2.0;
  ArrayXcd a0(N); a0.real() = (-x*x/8.0).exp(); a0.imag() = ArrayXd::Zero(N);
  //cout << a0 << endl;
  Cqcgl1d cgl; 
  ArrayXd v0 = cgl.C2R(a0);
  //ArrayXXd aa = cgl.intg(v0, 10, 1);
  //cout << aa.rows() << 'x' << aa.cols() << endl;
  //cout << cgl.R2C(aa.col(1)) << endl;
  Cqcgl1d::CGLaj aj = cgl.intgj(v0, 100, 1, 20);
  cout << aj.daa.rows() << 'x' << aj.daa.cols() << endl;
  #endif  
  //==================================================

  return 0;
}
