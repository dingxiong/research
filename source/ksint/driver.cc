#include "ksint.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
int main(){
  /// -----------------------------------------------------
  //#if 0
  ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
  KS ks(32, 0.02, 22);
  ArrayXXd aa;
  aa = ks.intg(a0, 2000,1000);
  cout << aa << endl << endl;
  //#endif
  
  /* ------------------------------------------------------- */
  #if 0
  ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
  KS ks(32, 0.25, 22);
  KS::KSaj aj = ks.intgj(a0, 20,1,1);
  //cout << aj.daa << endl;
  #endif

  return 0;
}
