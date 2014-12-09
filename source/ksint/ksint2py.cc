/* Requirement: >= g++4.6, >=eigen3.1
 * compile command:
 *  g++ -o libks2py.so ksintM1.cc ksint.cc ksint2py.cc
 *  -shared -fpic -lm  -lfftw3 -std=c++0x -march=corei7 -O3 -msse2 -msse4
 *  -I../../include -I/path/to/eigen3
 */
#include "ksintM1.hpp"
#include <Eigen/Dense>
#include <cstring>
#include <cstdlib>

using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Map;

typedef struct {
  double *aa;
  double *tt;
} at;

const int N = 32;

extern "C"
void ksf(double *aa, double *a0, double d, double h, int nstp, int np){
  KS ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  ArrayXXd vv = ks.intg(v0, nstp, np);
  memcpy(aa, &vv(0,0), (nstp/np+1)*(N-2)*sizeof(double));
}

extern "C"
void ksfM1(double *aa, double *tt, double *a0, double d, double h,
	   int nstp, int np){
  KSM1 ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  std::pair<ArrayXXd, ArrayXd> vv = ks.intg(v0, nstp, np);
  memcpy(aa, &(vv.first(0,0)), (nstp/np+1)*(N-2)*sizeof(double));
  memcpy(tt, &(vv.second(0)), (nstp/np+1)*sizeof(double));
}

extern "C"
int ksf2M1(at *at, double *a0, double d, double h,
	    int nstp, int np){
  KSM1 ks(N, h, d);
  Map<ArrayXd> v0(a0, N-2);
  std::pair<ArrayXXd, ArrayXd> vv = ks.intg2(v0, nstp, np);
  int m = vv.first.cols();
  
  double *aa = new double[m*(N-2)];
  double *tt = new double[m];
  
  at->aa = aa;
  at->tt = tt;
  
  return m;
}

void freeks(at *at){
  delete[] at->aa;
  delete[] at->tt;
}
