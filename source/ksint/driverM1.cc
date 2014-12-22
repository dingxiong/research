#include "ksintM1.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
//using KSM1::KSat;
int main(){
  /* ------------------------------------------------------- */
  //#if 0
  ArrayXd a0 = ArrayXd::Ones(30) * 0.1; a0(1) = 0.0;
  KSM1 ks(32, 0.1, 22);
  std::pair<ArrayXXd, ArrayXd> at = ks.intg(a0, 20000, 1);
  cout << at.first.rows() << endl;
  cout << at.first.cols() << endl;
  //#endif

  return 0;
}
