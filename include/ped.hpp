/** @file
 *  @brief Header file for periodic eigendecomposition algorithm.
 */

/** @class PED
 *  @brief A class to calculate the periodic eigendecomposition of
 *         the product of a sequence of matrices.
 *  @note This class require c++0x or c++11 support.
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
  void 
  reverseOrder(MatrixXd &J);
  void 
  reverseOrderSize(MatrixXd &J);
};

#endif	/* PED_H */

/** \mainpage Periodic Eigendecomposition
 *
 * \section sec_intro Introduction 
 * This package contains the source file of implementing periodic 
 * eigendecomposition. 
 *
 * Suppose there are M matrices \f$J_M, J_{M-1},\cdots, J_1\f$, each of
 * which has dimension [N, N]. We are interested in the eigenvalues and
 * eigenvectors of their products:
 * \f[
 *  J_M J_{M-1}\cdots J_1\,, \quad, J_1J_M\cdots,J_2\,,\quad
 *  J_2J_1J_M\cdots J_3\,,\quad \cdots
 * \f]
 * Note all of these products have same eigenvalues and their
 * eigenvectors are related by similarity transformation.
 * This package is designed to solve this problem.
 * 
 * There is only one class PED which has only
 * member functions, no member variables. For the detailed usage,
 * please go to the documentation of two functions PED::EigVals()
 * and PED::EigVecs().
 * 
 * \section sec_usage How to compile
 * This package is developed under template library
 * <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page"><b>Eigen</b></a>.
 * In order to use this package, make sure you have Eigen3.2 or above,
 * and your C++ compiler support C++11.
 *
 * For example, the command line compilation in linux is
 *
 * \code
 *  g++ ped.cc yourcode.cc -std=c++0x -O3 -I/path/to/eigen/header
 * \endcode
 *
 * \section sec_ack Acknowledgment
 * This is one project for my PhD study. I sincerely thank my adviser
 * <a href="https://www.physics.gatech.edu/user/predrag-cvitanovic">
 * Prof. Predrag Cvitanovic </a>
 * for his patient guidance.
 */	
