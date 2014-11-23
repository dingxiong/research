#include "ksint.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
int main(){
  /// -----------------------------------------------------
  switch (1){
    
  case 1:
    {
      ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
      KS ks(32, 0.1, 22);
      ArrayXXd aa;
      aa = ks.intg(a0, 20000,1);
      //cout << aa << endl << endl;
      break;
    }
  case 2 :
    {
      ArrayXd a0 = ArrayXd::Ones(30) * 0.1;
      KS ks(32, 0.1, 22);
      KS::KSaj aj = ks.intgj(a0, 20000,10000,10000);
      //cout << aj.aa << endl;
      break;
    }
  default :
    {
      cout << "please indicate the correct index." << endl;
    }
  }
  return 0;
}
