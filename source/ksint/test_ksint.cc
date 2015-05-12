/* to compile:
 * g++ test_ksint.cc -std=c++0x -lksint -lfftw3 -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3
 */

#include "ksintM1.hpp"
#include <Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
int main(){
  /// -----------------------------------------------------
  switch (5){
    
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
      KS ks(32, 0.01, 22);
      pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a0, 2000, 2000,2000);
      cout << tmp.second.col(0).tail(30) << endl;
      break;
    }
  case 3: // test velocity
    {
      
      ArrayXd a0(ArrayXd::Ones(30)*0.1);
      KS ks(32, 0.1, 22); 
      cout << ks.velocity(a0) << endl;
      /*
      ArrayXd a0(ArrayXd::Ones(62)*0.1);
      KS ks(64, 0.1, 22);
      cout << ks.velocity(a0) << endl;
      */
      break;
    }

  case 4: // test disspation
    {
      ArrayXd a0(ArrayXd::Ones(30)*0.1);
      KS ks(32, 0.1, 22);
      tuple<ArrayXXd, VectorXd, VectorXd> tmp = ks.intgDP(a0, 10000, 1);
      cout << get<1>(tmp).tail(1) << endl << endl;
      cout << get<2>(tmp).tail(1) << endl;

      break;
    }

  case 5: // test 1st mode integrator
    {
      ArrayXd a0(ArrayXd::Ones(62)*0.1);
      a0(1) = 0;
      KSM1 ks(64, 0.01, 22);
      pair<ArrayXXd, ArrayXd> tmp = ks.intg(a0, 10, 1);
      cout << tmp.first.cols() << endl << endl;
      // cout << get<2>(tmp).tail(1) << endl;

      break;
    }
  default :
    {
      cout << "please indicate the correct index." << endl;
    }
  }
  return 0;
}
