#ifndef PED_H
#define PED_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector> 
using std::vector;
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
  MatrixXd PerSchur(MatrixXd &J, const int MaxN, const double tol,
		    bool Print = True);
  MatrixXd HessTrian(MatrixXd &G);
  void PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
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
  
};

#endif	/* PED_H */

