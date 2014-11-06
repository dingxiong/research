#include <iostream>
#include "ped.hpp"
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main(){
  //----------------------------------------

#if 0
  // small test of HessTrian.
  MatrixXd A, B, C;
  A = MatrixXd::Random(4,4);
  B = MatrixXd::Random(4,4);
  C = MatrixXd::Random(4,4); 
  //MatrixXd A(4,4), B(4,4), C(4,4);
  //A << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  //B = A.array() + 2;
  //C = B.array() + 16;
  cout << A << endl << endl;
  cout << B << endl << endl;
  cout << C << endl << endl;
  //cout << A*B*C << endl << endl;
  
  PED ped;
  MatrixXd G(4,12);
  G << A, B, C;
  MatrixXd Q = ped.HessTrian(G);

  cout << G.leftCols(4) << endl << endl;
  cout << G.middleCols(4,4) << endl << endl;
  cout << G.rightCols(4) << endl << endl;
  //cout << A*B*C << endl << endl;
# endif
  //----------------------------------------

#if 0
  /*  Systematic test of HessTrian()                   * 
   *  						       * 
   *  Sample output:				       * 
   *  1.9984e-15				       * 
   *  7.99361e-15				       * 
   *  						       * 
   * real	0m8.252s			       * 
   * user	0m7.984s			       * 
   * sys	0m0.208s                               *    
   */
  const int N = 100; 
  const int M = 1000;
  MatrixXd J = MatrixXd::Random(N, M*N);
  MatrixXd J0 = J;
  PED ped;
  MatrixXd Q = ped.HessTrian(J);
  
  double TriErr = 0; // error of the triangular form
  double QJQErr = 0; //error of the transform
  for(size_t i = 0; i < M; i++){
    // should be strict lower.
    MatrixXd tmp;
    if(0 == i){
      tmp = J.middleCols(i*N, N).bottomLeftCorner(N-1, N-1).triangularView<StrictlyLower>();

    }else{
      tmp = J.middleCols(i*N, N).triangularView<StrictlyLower>(); 
    }
    TriErr = max(TriErr, tmp.cwiseAbs().maxCoeff() );
    
    // error of transform QJQ - J
    MatrixXd dif = Q.middleCols(i*N, N).transpose() * J0.middleCols(i*N, N) 
      * Q.middleCols(((i+1)%M)*N, N) - J.middleCols(i*N, N);
    QJQErr = max(QJQErr, dif.cwiseAbs().maxCoeff() );
  }
  cout << TriErr << endl;
  cout << QJQErr << endl;

#endif
  
  /* -------------------------------------------------- */
#if 1
  /* small test of GivensOneRound/GivensOneIter/PeriodicQR */
  MatrixXd A, B, C;
  A = MatrixXd::Random(4,4);
  B = MatrixXd::Random(4,4);
  C = MatrixXd::Random(4,4); 

  PED ped;
  MatrixXd J(4,12);
  J << A, B, C;
  MatrixXd Q = ped.HessTrian(J);

  cout << J.leftCols(4) << endl << endl;
  cout << J.middleCols(4,4) << endl << endl;
  cout << J.rightCols(4) << endl << endl;
  
  Vector2d tmp = J.block(0, 0, 2, 1);
  // ped.GivensOneRound(J, Q, tmp, 0);
  // ped.GivensOneIter(J, Q, tmp, 0, 3);
  // for(int i = 0; i < 100; i++) ped.GivensOneIter(J, Q, tmp, 0, 3);
  ped.PeriodicQR(J, Q, 0, 3, 100, 1e-15, true);

  cout << J.leftCols(4) << endl << endl;
  cout << J.middleCols(4,4) << endl << endl;
  cout << J.rightCols(4) << endl << endl;
  
#endif
  
}
