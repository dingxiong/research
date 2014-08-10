#include "kssym.hpp"
#include <assert.h>
#include <cmath>
#include <complex>
#include <cstdlib>

typedef std::complex<double> dcomp;

MatrixXd Kssym::redSO2(const MatrixXd &aa, double *ang){
  size_t n = aa.rows(); 
  size_t m = aa.cols();
  MatrixXd raa(n, m);
  assert ( n%2 == 0); 
  
  for (int i = 0; i < m; i++ ){
    double th = arg(dcomp( aa(0,i), aa(1,i) )) / px;
    if(ang != NULL) ang[i] = th;
    raa.col(i) = rotate(aa.col(i), -th);
  }

  return raa;
}



MatrixXd Kssym::rotate(const MatrixXd &aa, const double th){
  size_t n = aa.rows();
  size_t m = aa.cols();
  MatrixXd raa(n, m);
  assert( n%2 == 0);
  
  for(int i = 0; i < n/2; i++){
    Matrix2d R;
    double c = cos(th*(i+1)); 
    double s = sin(th*(i+1));
    
    R << 
      c, -s, 
      s, c; // define the matrix

    raa.middleRows(2*i, 2) = R * aa.middleRows(2*i, 2);
    
  }
  
  return raa;
}
