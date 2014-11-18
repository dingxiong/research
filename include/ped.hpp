/** This class require c++0x or c++11 support.
 *
 */
#ifndef PED_H
#define PED_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>
#include <vector> 
#include <utility>
#include <tuple>


using std::vector;
using std::pair; using std::make_pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix2d; using Eigen::Vector2d;
using Eigen::Matrix2cd; using Eigen::Vector2cd;
using Eigen::MatrixXcd; using Eigen::VectorXcd;
using Eigen::Map; using Eigen::Ref;
using Eigen::EigenSolver;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Eigen::SparseLU; 
using Eigen::COLAMDOrdering;

/*============================================================ *
 *                 Class : periodic Eigendecomposition         *
 *============================================================ */
class PED{
  
public:
  MatrixXd 
  EigVals(MatrixXd &J, const int MaxN  = 100,
	  const double tol = 1e-16 , bool Print  = true);
  pair<MatrixXd, MatrixXd>
  EigVecs(MatrixXd &J, const int MaxN  = 100,
	  const double tol = 1e-16, bool Print = true);
  std::tuple<MatrixXd, vector<int>, MatrixXd> 
  eigenvalues(MatrixXd &J, const int MaxN = 100,
	      const double tol = 1e-16, bool Print = true);
  pair<MatrixXd, vector<int> >
  PerSchur(MatrixXd &J, const int MaxN = 100,
	   const double tol = 1e-16, bool Print = true);
  MatrixXd 
  HessTrian(MatrixXd &G);
  vector<int> 
  PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
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
  vector<int> 
  checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
		   const int U,const double tol);
  vector<int> 
  padEnds(const vector<int> &v, const int &left, const int &right);
  double 
  deltaMat2(const Matrix2d &A);
  Vector2d 
  vecMat2(const Matrix2d &A);
  pair<Vector2d, Matrix2d> 
  complexEigsMat2(const Matrix2d &A);
  int 
  sgn(const double &num);
  pair<double, int> 
  product1dDiag(const MatrixXd &J, const int k);
  pair<Matrix2d, double> 
  product2dDiag(const MatrixXd &J, const int k);
  vector<int> 
  realIndex(const vector<int> &complexIndex, const int N);
  vector<Triplet<double> >
  triDenseMat(const Ref<const MatrixXd> &A, const size_t M = 0, 
	      const size_t N = 0);
  vector<Triplet<double> > 
  triDenseMatKron(const size_t I, const Ref<const MatrixXd> &A, 
		  const size_t M = 0, const size_t N = 0);
  vector<Triplet<double> > 
  triDiagMat(const size_t n, const double x, 
	     const size_t M = 0, const size_t N = 0 );
  vector<Triplet<double> > 
  triDiagMatKron(const size_t I, const Ref<const MatrixXd> &A,
		 const size_t M = 0, const size_t N = 0 );
  pair<SparseMatrix<double>, VectorXd> 
  PerSylvester(const MatrixXd &J, const int &P, 
	       const bool &isReal, const bool &Print);
  MatrixXd
  oneEigVec(const MatrixXd &J, const int &P, 
	    const bool &isReal, const bool &Print);
  void
  fixPhase(MatrixXd &EigVecs, const VectorXd &realComplexIndex);

};

#endif	/* PED_H */

