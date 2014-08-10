#include <iostream>
#include "kssym.hpp"
#include <eigen3/Eigen/Dense>
using namespace std;

int main(){
  MatrixXd x = MatrixXd::Random(30, 40000);
  Kssym ks;

  MatrixXd y = ks.redSO2(x, NULL);

  return 0;
}
