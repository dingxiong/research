#include "ksint.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
int main(){
  /// -----------------------------------------------------
  #if 0
  ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
  KS ks(32, 0.1, 22);
  ArrayXXd aa;
  for(int i = 0; i < 100; i++) aa = ks.intg(a0, 20000,1000);
  //cout << aa.col(200) << endl << endl;
  #endif
  
  /* ------------------------------------------------------- */
  //#if 0
  ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
  KS ks(32, 0.25, 22);
  KS::KSaj aj = ks.intgj(a0, 10000,1000,1000);
  //cout << aj.daa << endl;
  //#endif

  return 0;
}
