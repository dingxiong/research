/** This class require c++0x or c++11 support.
 *
 */
#ifndef PED_H
#define PED_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector> 
#include <utility>

using std::vector;
using std::pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix2d; using Eigen::Vector2d;
using Eigen::Map; using Eigen::Ref;
using Eigen::EigenSolver;

/*============================================================ *
 *                 Class : periodic Eigendecomposition         *
 *============================================================ */
class PED{
  
public:
  VectorXd EigVals(MatrixXd &J, const int MaxN  = 100,
  		   const double tol = 1e-16 , bool Print  = true);
  pair<MatrixXd, vector<int> > PerSchur(MatrixXd &J, const int MaxN = 100,
		    const double tol = 1e-16, bool Print = true);
  MatrixXd HessTrian(MatrixXd &G);
  vector<int> PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
			 const int MaxN, const double tol, bool Print);
  //protected:
  void Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C,
	      const int k);
  void Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
	      const Vector2d &v, const int k);
  void HouseHolder(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, const int k,
		   bool subDiag = false);
  void GivensOneIter(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
		     const int L, const int U);
  void GivensOneRound(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
		      const int k);
  vector<int> checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
				  const int U,const double tol);
  vector<int> padEnds(const vector<int> &v, const int &left, const int &right);
  double deltaMat2(const Matrix2d &A);
  Vector2d vecMat2(const Matrix2d &A);
  int sgn(const double &num);
  pair<double, int> product1dDiag(const MatrixXd &J, const int k);
  pair<Matrix2d, double> product2dDiag(const MatrixXd &J, const int k);
};

#endif	/* PED_H */

