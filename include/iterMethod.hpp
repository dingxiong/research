/** \mainpage iterative methods to solve linear problems
 *
 *  \section sec_intro Introduction
 *  This file contains the iterative methods to solve linear problems. It
 *  mainly contains the congugate gradient method w/wo preconditioning and
 *  provide SSOR as one choice of precondition method, which is thought to
 *  perform better than Jacobi and G-S. The reason that I insist writing
 *  these routines by myself is that the corresponding libraries in Eigen
 *  seem not to give the desired accuracy when I use them to refine POs for
 *  KS system. I suggest you using the libraries shipped with Eigen, but if
 *  you want to use my routines, I welcome !
 *
 *  \section sec_use usage
 *  This file is assembly of template functions, which means that all function
 *  implementations are in this header file, and there is no .cc file.
 *  You only need to include this file when compiling your code.
 *  Example:
 *  \code
 *  g++ yourfile.cc -I/path/to/iterMethod.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 */

#ifndef ITERMETHOD_H
#define ITERMETHOD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <fstream>

//////////////////////////////////////////////////
//              Interface                       //
//////////////////////////////////////////////////
namespace iterMethod {
  
  using namespace std;
  using namespace Eigen;
  typedef Eigen::SparseMatrix<double> SpMat;


  template <typename Mat>
  std::pair<Eigen::VectorXd, std::vector<double> >
  ConjGrad(const Mat &A, const Eigen::VectorXd &b, 
	   const Eigen::VectorXd &x0,
	   const int jmax, const double rtol);


  template <typename Mat, typename LinearSolver>
  std::pair<VectorXd, vector<double> >
  ConjGradPre(const Mat &A, const VectorXd &b, const Mat &M, 
	      LinearSolver &solver, 
	      const VectorXd &x0, const int jmax, const double rtol);

  template <typename Mat>
  Mat preSSOR(const Mat &A);

  template < typename Mat, typename LinearSolver>
  std::pair<VectorXd, vector<double> >
  ConjGradSSOR(const Mat &A, const VectorXd &b,
	       LinearSolver &solver, 
	       const VectorXd &x0, const int jmax, const double rtol);
}


//////////////////////////////////////////////////
//              Implementation                  //
//////////////////////////////////////////////////

namespace iterMethod {


  /** @brief perform conjugate gradient method on symmetry matrix A
   *         to solve Ax=b.
   *         Here the type of A is a template. permitted types are
   *         MatrixXd and SparseMatrix<double>.
   *
   *  @param[in] A symmetric (sparse) matrix
   *  @param[in] b vector
   *  @param[in] x0 initial guess of solution x
   *  @param[in] jmax maximal iteration number. This method is guranteed
   *                  to converge at most A's size iteration, so
   *                  jmax should be smaller than A.rows().
   *  @param[in] rtol relative tolerance for convergence
   */
  template <typename Mat>
  pair<VectorXd, vector<double> >
  ConjGrad(const Mat &A, const VectorXd &b, 
	   const VectorXd &x0, const int jmax,
	   const double rtol){
    assert( A.rows() == A.cols() );
    const int n = A.rows();
    VectorXd r1(b);
    VectorXd r0(b);
    VectorXd p(VectorXd::Zero(n));
    VectorXd x(x0);
    vector<double> res;
    res.reserve(jmax);
    double errInit = b.norm();

    for (size_t i = 0; i < jmax; i++) {
      double r1snorm = r1.squaredNorm(); 
      double mu = r1snorm / r0.squaredNorm(); 
      p = mu * p + r1;
      VectorXd ap = A * p;
      double sigma = r1snorm / (p.transpose() * ap);
      x = x + sigma * p;
      r0 = r1;
      r1 -= sigma * ap;
      double err = r1.norm();
      res.push_back(err);
      if(err <= rtol*errInit) break;
    }

    return make_pair(x, res);

  }

  /** @brief conjugate gradient with perconditioning
   *
   * Example usage:
   * 
   * for sparse matrix :
   * \code
   * typedef Eigen::SparseMatrix<double> SpMat;
   * // construction of A, M
   * // ...
   * SparseLU<SpMat> solver;
   * pair<VectorXd, vector<double> > cg = iterMethod::ConjGradPre<SpMat, SparseLU<SpMat> >
   * (A, b, M, solver, VectorXd::Zero(N), 100, 1e-6);
   * \endcode
   *
   * for dense matrix :
   * \code
   * PartialPivLU<MatrixXd> solver;
   * pair<VectorXd, vector<double> >
   * cg = iterMethod::ConjGradPre<MatrixXd, PartialPivLU<MatrixXd> >
   * (A, b, M, solver, VectorXd::Zero(N), 100, 1e-16);
   * \endcode
   * 
   * @param[in] M preconditioning matrix
   * @param[in] solver the linear solver
   * @see ConjGrad()
   * 
   */
  template <typename Mat, typename LinearSolver>
  std::pair<VectorXd, vector<double> >
  ConjGradPre(const Mat &A, const VectorXd &b, const Mat &M, 
	      LinearSolver &solver, 
	      const VectorXd &x0, const int jmax, const double rtol){

    assert( A.rows() == A.cols() );
    const int n = A.rows();
    VectorXd r1(b);
    VectorXd r0(b);
    VectorXd p(VectorXd::Zero(n));
    VectorXd x(x0); 
    solver.compute(M);
    VectorXd z0 = solver.solve(r0);
    vector<double> res;
    res.reserve(jmax);
    double errInit = b.norm();

    for (size_t i = 0; i < jmax; i++) {
      VectorXd z1 = solver.solve(r1);
      double r1z1 = r1.dot(z1);
      double mu = r1z1 / r0.dot(z0); 
      z0 = z1;
      p = mu * p + z1;
      VectorXd ap = A * p;
      double sigma = r1z1 / p.dot(ap);
      x = x + sigma * p;
      r0 = r1;
      r1 -= sigma * ap;
      double err = r1.norm();
      res.push_back(err);
      if(err <= rtol*errInit) break;
    }

    return make_pair(x, res);
  }
  
  /** @brief find the SSOR precondition matrix for CG method
   *  This works for both dense and sparse matrix.
   */
  template<typename Mat>
  Mat preSSOR(const Mat &A){
    Mat LD = A.template triangularView<Lower>();
    Mat UD = A.template triangularView<Upper>();

    return LD * ( A.diagonal().asDiagonal().inverse()) * UD;
  }
  

  /** @brief CG method with SSOR preconditioning
   *
   *  @see preSSOR
   *  @see ConjGradPre()
   */
  template < typename Mat, typename LinearSolver>
  std::pair<VectorXd, vector<double> >
  ConjGradSSOR(const Mat &A, const VectorXd &b,
	       LinearSolver &solver, 
	       const VectorXd &x0, const int jmax, const double rtol){
    Mat M = preSSOR<Mat> (A);
    return ConjGradPre<Mat, LinearSolver>(A, b, M, solver, x0, jmax, rtol);
  }

}

#endif	/* ITERMETHOD_H */
