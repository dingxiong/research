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
#include <tuple>
#include <iostream>
#include <fstream>
#include "denseRoutines.hpp"

//////////////////////////////////////////////////
//              Interface                       //
//////////////////////////////////////////////////
namespace iterMethod {
  
    using namespace std;
    using namespace Eigen;
    using namespace denseRoutines;
    
    typedef Eigen::SparseMatrix<double> SpMat;

    /* -------------------------------------------------- */
    extern bool CG_PRINT;
    extern int CG_PRINT_FREQUENCE;
    
    extern bool GMRES_OUT_PRINT;
    extern int GMRES_OUT_PRINT_FREQUENCE;

    extern bool GMRES_IN_PRINT;
    extern int GMRES_IN_PRINT_FREQUENCE;

    extern bool HOOK_PRINT;
    extern int HOOK_PRINT_FREQUENCE;

    extern bool INB_OUT_PRINT;
    extern int INB_OUT_PRINT_FREQUENCE;

    extern bool INB_IN_PRINT;
    extern int INB_IN_PRINT_FREQUENCE;

    extern bool LM_OUT_PRINT;
    extern int LM_OUT_PRINT_FREQUENCE;

    extern bool LM_IN_PRINT;
    extern int LM_IN_PRINT_FREQUENCE;

    extern double LM_LAMBDA_INCREASE_FACTOR; 
    extern double LM_LAMBDA_DECREASE_FACTOR;
    extern double LM_MAX_LAMBDA;
    /* -------------------------------------------------- */

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
    
    /* -------------------------------------------------- */
    void rotmat(const double &x, const double &y,
		double *c, double *s);

    template <class Adotx>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0(Adotx Ax , const VectorXd &b, const VectorXd &x0,
	   const int restart,
	   const int maxit, const double rtol);
    
    template <typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres(const Mat &A, const VectorXd &b, const VectorXd &x0, 
	  const int restart,
	  const int maxit, const double rtol);
    
    ArrayXd calz(const ArrayXd &D, const ArrayXd &p, const double mu);
    
    std::tuple<double, std::vector<double>, int>
    findTrustRegion(const ArrayXd &D, const ArrayXd &p, double delta,
		    const double tol = 1e-12,
		    const int maxit = 100,
		    const double mu0 = 0);

    /* -------------------------------------------------- */
    template <class Adotx>
    std::tuple<VectorXd, VectorXd, VectorXd, ArrayXd, MatrixXd, MatrixXd, std::vector<double>, int>
    Gmres0SVD(Adotx Ax , const VectorXd &b, const VectorXd &x0,
	      const int restart,
	      const int maxit, const double rtol);

    template<class Fx, class Jacv>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0Hook( Fx fx, Jacv jacv,
		const VectorXd &x0,
		const double tol,
		const double minRD,
		const int maxit,
		const int maxInnIt,
		const double GmresRtol,
		const int GmresRestart,
		const int GmresMaxit,
		const bool testT,
		const int Tindex);
    
    template<class Fx, class Jacv, class Precondition>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0HookPre( Fx fx, Jacv jacv, Precondition Pre,
		   const VectorXd &x0,
		   const double tol,
		   const double minRD,
		   const int maxit,
		   const int maxInnIt,
		   const double GmresRtol,
		   const int GmresRestart,
		   const int GmresMaxit,
		   const bool testT,
		   const int Tindex);

    template<typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    GmresHook( const Mat &A, const VectorXd &b,
	       const VectorXd &x0,
	       const double tol,
	       const double minRD,
	       const int maxit,
	       const int maxInnIt,
	       const double GmresRtol,
	       const int GmresRestart,
	       const int GmresMaxit,
	       const bool testT,
	       const int Tindex);

    template<typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    GmresHookPre( const Mat &A, const VectorXd &b,
		  const VectorXd &x0,
		  const double tol,
		  const double minRD,
		  const int maxit,
		  const int maxInnIt,
		  const double GmresRtol,
		  const int GmresRestart,
		  const int GmresMaxit,
		  const bool testT,
		  const int Tindex);
    /* -------------------------------------------------- */
    double chooseTheta(const double g0, const double g1, const double gp0,
		       const double theta_min, const double theta_max);

    template<class Fx, class Jacv>
    std::tuple<VectorXd, std::vector<double>, int>
    InexactNewtonBacktrack(Fx &fx, Jacv &jacv,
			   const VectorXd &x0,
			   const double tol = 1e-12,
			   const int btMaxIt = 20,
			   const int maxit = 100,
			   const double eta0 = 1e-4,
			   const double t = 1e-4,
			   const double theta_min = 0.1,
			   const double theta_max = 0.5,
			   const int GmresRestart = 30,
			   const int GmresMaxit = 100);
    

    /* --------------------------  LM  ------------------------ */

    template<class Fx, template<class> class CalJJ, class LinearSolver, class Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    LM0( Fx & fx, CalJJ<Mat> &JJF, LinearSolver &solver, 
	 const VectorXd &x0, const double tol,
	 const int maxit, const int innerMaxit);
    
    std::tuple<VectorXd, std::vector<double>, int>
    LM( const MatrixXd &A, const VectorXd &b,
	const VectorXd &x0, const double tol,
	const int maxit, const int innerMaxit);

}


///////////////////////////////////////////////////              Implementation                  ////////////////////////

namespace iterMethod {

    //////////////////////////////////////////////////////////////////////
    //                        CG related                                //
    //////////////////////////////////////////////////////////////////////
    
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
	res.push_back(errInit);
	
	for (size_t i = 0; i < jmax; i++) {

	    if(CG_PRINT && i % CG_PRINT_FREQUENCE == 0)
		fprintf(stderr, "CG : i= %zd/%d , r= %g\n", i, jmax, res.back());

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
	res.push_back(errInit);
	
	for (size_t i = 0; i < jmax; i++) {

	    if(CG_PRINT && i % CG_PRINT_FREQUENCE == 0)
		fprintf(stderr, "CG : i= %zd/%d , r= %g\n", i, jmax, res.back());

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


    //////////////////////////////////////////////////////////////////////
    //                        GMRES                                     //
    //////////////////////////////////////////////////////////////////////
    

    /**
     * @brief GMRES method to solve A*x = b.
     *
     *  This algorithm is described in
     *  "
     *  Saad Y, Schultz M 
     *  GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems
     *  SIAM Journal on Scientific and Statistical Computing, 1986
     *  "
     *  The implementation follows the free netlib implementation
     *  http://www.netlib.org/templates/matlab/gmres.m
     *  with some modifications. Also, preconditioning is not implemented right now.
     *
     * @param[in]  Ax      : a function take one argument and return Krylov vector:  Ax(x) = A*x
     * @param[in]  b       : right side of linear equation A*x = b
     * @param[in]  x0      : initial guess
     * @param[in]  restart : restart number
     * @param[in]  maxit   : maximum iteration number
     * @param[in]  rtol    : relative error tolerance 
     * @return [x, errVec, flag]
     *          x       : the sotion of Ax=b
     *          errVec  : errors in each iteration step
     *          flag    : 0 => converged, 1 => not
     *          
     * @note this routine does not require the explicit form of matrix A, but only a function
     *       which can return A*x.
     * @see  Gmres()
     */
    template <class Adotx>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0(Adotx Ax , const VectorXd &b, const VectorXd &x0,
	   const int restart,
	   const int maxit, const double rtol){

	/* initial setup */
	const int N = b.size();
	int M = restart;
	VectorXd x = x0;
	double bnrm2 = b.norm();
	if ( bnrm2 == 0.0 ) bnrm2 = 1.0;
	std::vector<double> errVec;	/* error at each iteration */

	/* initialize workspace */
	MatrixXd V(MatrixXd::Zero(N, M+1)); // orthogonal vectors in Arnoldi iteration
	MatrixXd H(MatrixXd::Zero(M+1, M)); // Hessenberg matrix in Arnoldi iteration
	
	ArrayXd C(ArrayXd::Zero(M+1));  // cos of Givens rotation
	ArrayXd S(ArrayXd::Zero(M+1));  // sin of Givens rotation
    
	/* outer iteration */
	for(size_t iter = 0; iter < maxit; iter++){
	    /* obtain residule */
	    VectorXd r = b - Ax(x); 
	    double rnorm = r.norm();
	    double err = rnorm / bnrm2;
	    
	    if(GMRES_OUT_PRINT && iter % GMRES_OUT_PRINT_FREQUENCE == 0)
		fprintf(stderr, "GMRES : out loop: i= %zd/%d , r= %g\n", iter, maxit, err);

	    if(err < rtol) return std::make_tuple(x, errVec, 0);
	
	    V.col(0) = r / rnorm;	// obtain V_1
	
	    VectorXd g(VectorXd::Zero(M+1));// vector g = ||r|| * e1
	    g(0) = rnorm;		

	    /* inner iteration : Arnoldi Iteration */
	    for(size_t i = 0; i < M; i++){
		// form the V_{i+1} and H(:, i)
		V.col(i+1) = Ax( V.col(i) );
		for(size_t j = 0; j <= i; j++){
		    H(j, i) = V.col(i+1).dot( V.col(j) );
		    V.col(i+1) -= H(j, i) * V.col(j);
		}
		H(i+1, i) = V.col(i+1).norm();
		if(H(i+1, i) != 0) V.col(i+1) /= H(i+1, i);
	    
		/* Givens Rotation
		 *  | c, s| * |x| = |r|
		 *  |-s, c|   |y|   |0|
		 */
		for(size_t j = 0; j < i; j++) {
		    double tmp = C(j) * H(j, i) + S(j) * H(j+1, i);
		    H(j+1, i) = -S(j) * H(j, i) + C(j) * H(j+1, i);
		    H(j, i) = tmp;
		}
	    
		/* rotate the last element */
		rotmat(H(i, i), H(i+1, i), &C(i), &S(i));
		H(i ,i) = C(i) * H(i, i) + S(i) * H(i+1, i);
		H(i+1, i) = 0;

		/* rotate g(i) and g(i+1)
		 *  | c, s| * |g_i| = |c * g_i |
		 *  |-s, c|   | 0 |   |-s * g_i|
		 */
		g(i+1) = -S(i) * g(i); // be careful about the order
		g(i) = C(i) * g(i);

		/* after all above operations, we obtain dimensions
		 *  H : [i+2, i+1]
		 *  V : [N,   i+2]
		 *  g : [i+2,   1]
		 *
		 * Here, it is better denote H as R since it is transfored
		 * into upper triangular form.
		 * residual = || \beta * e_1 - H * y|| = ||g - R*y||
		 * since the last row of R is zero, then residual =
		 * last element of g, namely g(i+1)
		 */
		double err = fabs(g(i+1)) / bnrm2 ; 
		errVec.push_back(err);

		if(GMRES_IN_PRINT && i % GMRES_IN_PRINT_FREQUENCE == 0)
		    fprintf(stderr, "GMRES : inner loop: i= %zd/%d , r= %g\n", i, M, err);

		if (err < rtol){
		    VectorXd y = H.topLeftCorner(i+1, i+1).lu().solve(g.head(i+1));
		    x += V.leftCols(i+1) * y; 
		    return std::make_tuple(x, errVec, 0);
		}
	    }

	    // the inner loop finishes => has not converged
	    // so need to update x, and go to outer loop
	    VectorXd y = H.topLeftCorner(M, M).lu().solve(g.head(M));
	    x += V.leftCols(M) * y; 
	}

	// if the outer loop finished => not converged
	return std::make_tuple(x, errVec, 1);
    
    }

    /**
     * @brief a wrapper of Gmres0.
     *
     * This function uses an explicit form of matrix A.
     */
    template <typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres(const Mat &A , const VectorXd &b, const VectorXd &x0,
	  const int restart,
	  const int maxit, const double rtol){

	return Gmres0([&A](const VectorXd &x){return A * x;}, b, x0, restart, maxit, rtol);
    }

    //////////////////////////////////////////////////////////////////////
    //                        GMRES Hook                                //
    //////////////////////////////////////////////////////////////////////

    /**
     * @brief GMRES method to solve A*x = b  w. r. t  ||x|| < delta
     *
     * This GMRES method adds a hook step to the original GMRES to restrict the update to
     * a trust region.
     * 
     * The implementation follows paper 
     * "Simple invariant solutions embedded in 2D Kolmogorov turbulence " by
     *             GARYJ. CHANDLER AND RICHR. KERSWELL
     * and online reference
     * http://channelflow.org/dokuwiki/doku.php?id=docs:math:newton_krylov_hookstep
     *
     * @param[in]  Ax            : a function take one argument and return Krylov vector:  Ax(x) = A*x
     * @param[in]  b             : right side of linear equation A*x = b
     * @param[in]  x0            : initial guess
     * @param[in]  restart       : restart number
     * @param[in]  maxit         : maximum iteration number
     * @param[in]  rtol          : relative error tolerance
     * @param[in]  delta         : radius of ball constriant
     * @param[in]  innerMaxit    : maximal inner iteration number 
     * @return [x, xold, p, D, V2, V, errVec, flag]
     *          x         the sotion of Ax=b
     *          xold      the old value just before the last update
     *          p         the residule vector without the last element
     *          D         SVD diagonal array
     *          V2        SVD right hand side orthogonal matrix
     *          V         the orthogonal matrix in the Arnolds iteration
     *          errVec    errors in each iteration step
     *          flag      0 => converged, 1 => not
     *          
     * @note this routine does not require the explicit form of matrix A, but only a function
     *       which can return A*x.
     *       The return values: p, D, V2, V can be used to reconstruct the update vector.
     * @see  Gmres0(), Gmres0Hook()
     */
    template <class Adotx>
    std::tuple<VectorXd, VectorXd, VectorXd, ArrayXd, MatrixXd, MatrixXd, std::vector<double>, int>
    Gmres0SVD(Adotx Ax , const VectorXd &b, const VectorXd &x0,
	      const int restart,
	      const int maxit, const double rtol){
	
	/* initial setup */
	const int N = b.size();
	int M = restart;
	VectorXd x = x0;
	double bnrm2 = b.norm();
	if ( bnrm2 == 0.0 ) bnrm2 = 1.0;
	std::vector<double> errVec;	/* error at each iteration */

	/* initialize workspace */
	MatrixXd V(MatrixXd::Zero(N, M+1)); // orthogonal vectors in Arnoldi iteration
	MatrixXd H(MatrixXd::Zero(M+1, M)); // Hessenberg matrix in Arnoldi iteration
    
	/* outer iteration */
	for(size_t iter = 0; iter < maxit; iter++){
	    /* obtain residule */
	    VectorXd r = b - Ax(x);
	    double rnorm = r.norm();
	    double err = rnorm / bnrm2;

	    if(GMRES_OUT_PRINT && iter % GMRES_OUT_PRINT_FREQUENCE == 0)
		fprintf(stderr, "**** GMRES : out loop: i= %zd/%d, r= %g\n", iter, maxit, err);

	    // if(err < rtol) return std::make_tuple(x, errVec, 0);
	
	    V.col(0) = r / rnorm;	// obtain V_1
	    
	    /* inner iteration : Arnoldi Iteration */
	    for(size_t i = 0; i < M; i++){
		// form the V_{i+1} and H(:, i)
		V.col(i+1) = Ax( V.col(i) );
		for(size_t j = 0; j <= i; j++){
		    H(j, i) = V.col(i+1).dot( V.col(j) );
		    V.col(i+1) -= H(j, i) * V.col(j);
		}
		H(i+1, i) = V.col(i+1).norm(); // cout << H.col(i).head(i+2) << endl;
		if(H(i+1, i) != 0) V.col(i+1) /= H(i+1, i);
		else fprintf(stderr, "H(i+i, i) = 0, Boss, what should I do ? \n");
		
		// conduct SVD decomposition
		// Here we must use the full matrix U
		// the residul is |p(i+1)|
		JacobiSVD<MatrixXd> svd(H.topLeftCorner(i+2, i+1), ComputeFullU | ComputeThinV);
	        ArrayXd D ( svd.singularValues() );
		MatrixXd U ( svd.matrixU() ); 
		MatrixXd V2 ( svd.matrixV() ); 
		VectorXd p = rnorm * U.row(0);
		double err = fabs(p(i+1)) / bnrm2;
		errVec.push_back(err); 

		if(GMRES_IN_PRINT && i%GMRES_IN_PRINT_FREQUENCE == 0)
		    fprintf(stderr, "** GMRES : inner loop: i= %zd/%d, r= %g\n", i, M, err);

		if (err < rtol){
		    VectorXd xold = x; 
 		    ArrayXd z = p.head(i+1).array() / D;
		    VectorXd y = V2 * z.matrix();
		    x += V.leftCols(i+1) * y;
		    return std::make_tuple(x, xold, p.head(i+1), D, V2, V.leftCols(i+1), errVec, 0);
		}
		if (i == M -1){ /* last one but has not converged */
		    VectorXd xold = x; 
		    ArrayXd z = p.head(i+1).array() / D;
		    VectorXd y = V2 * z.matrix();
		    x += V.leftCols(i+1) * y;
		    if (iter == maxit - 1){ // if the outer loop finished => not converged
			return std::make_tuple(x, xold, p.head(i+1), D, V2, V.leftCols(i+1), errVec, 1);
		    }
		}
	    }
	}
	
    }
        
    /**
     * @brief use GMRES HOOK method to fin the solution of A x = b
     *
     * @param[in] testT     whether test the updated period T is postive
     * @param[in] Tindex    the index of T in the state vector counting from the tail
     * @param[in] minRD      minimal relative descrease at each step
     */
    template<class Fx, class Jacv>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0Hook( Fx fx, Jacv jacv,
		const VectorXd &x0,
		const double tol,
		const double minRD,
		const int maxit,
		const int maxInnIt,
		const double GmresRtol,
		const int GmresRestart,
		const int GmresMaxit,
		const bool testT,
		const int Tindex){

	const int N = x0.size(); 
	VectorXd x(x0);
	std::vector<double> errVec;

	bool fail = false;
	for(size_t i = 0; i < maxit; i++){
	    VectorXd F = fx(x);
	    double Fnorm = F.norm();
	    
	    if(HOOK_PRINT && i % HOOK_PRINT_FREQUENCE == 0)
		fprintf(stderr, "\n+++++++++++ GHOOK: i = %zd/%d, r = %g ++++++++++ \n", i, maxit, Fnorm);

	    errVec.push_back(Fnorm);
	    if(Fnorm < tol) return std::make_tuple(x, errVec, 0);

	    // use GmresRPO to solve F' dx = -F
	    auto Ax = [&x, &jacv](const VectorXd &t){return jacv(x, t); };
	    auto tmp = Gmres0SVD(Ax, -F, VectorXd::Zero(N), GmresRestart, GmresMaxit, GmresRtol); 
	    if(std::get<7>(tmp) != 0) fprintf(stderr, "GMRES SVD not converged !\n");
	    VectorXd &s = std::get<0>(tmp); // update vector
	    VectorXd &sold = std::get<1>(tmp); // old update vector just before last change
	    VectorXd &p = std::get<2>(tmp);
	    ArrayXd &D = std::get<3>(tmp);
	    MatrixXd &V2 = std::get<4>(tmp);
	    MatrixXd &V = std::get<5>(tmp); 
	    
	    ArrayXd D2 = D * D;
	    ArrayXd pd = p.array() * D;
	    // ArrayXd mu = ArrayXd::Ones(p.size()) * 0.1; 
	    ArrayXd mu = ArrayXd::Ones(p.size()) * 0.1 * D2.minCoeff(); 
	    for(size_t j = 0; j < maxInnIt; j++){ 
		VectorXd newx = x + s; 
		double newT = newx(N - Tindex); 

		if(HOOK_PRINT && i % HOOK_PRINT_FREQUENCE == 0)	    
		    fprintf(stderr, " %zd, %g |", j, newT);

		if(!testT || newT > 0){
		    VectorXd newF = fx(newx);
		    if(newF.norm() < (1 - minRD)*Fnorm){
			x = newx;
			break;
		    }
		}
		ArrayXd z = pd / (D2 + mu);
		VectorXd y = V2 * z.matrix();
		s = sold + V * y;
		mu *= 2;
		
		if(j == maxInnIt-1) fail = true;
	    }
	    
	    if(fail) break; // if all inner loop finish, it means state not changed
			    // then no need to iterate more.
	}

	return std::make_tuple(x, errVec, 0);
	    
    }


    /**
     * Preconditioning version of GMRES.
     *
     * GMRES will converge in fewer steps if the eigenvalues are clustered. A 
     * matrix P^{-1} which makes AP^{-1} close to identiy is the goal of right
     * side preconditioning. The equation then becomes AP^{-1} (Px) = b. If the
     * solution is AP^{-1} y = b. then x = P^{-1} y. So only need a function
     * to calculate P^{-1} y.
     * See discussions in
     * ----------------------------------------------------------------------
     * "How Fast are Nonsymmetric Matrix Iterations?" by
     * Nachtigal, Noël; Reddy, Satish C.; and Trefethen, Lloyd N.
     * ----------------------------------------------------------------------
     *
     * @parame[in] Pre    the precondition funcion which return product P^{-1}x
     * @see Gmres0SVD
     * 
     */
    template<class Fx, class Jacv, class Precondition>
    std::tuple<VectorXd, std::vector<double>, int>
    Gmres0HookPre( Fx fx, Jacv jacv, Precondition Pre,
		   const VectorXd &x0,
		   const double tol,
		   const double minRD,
		   const int maxit,
		   const int maxInnIt,
		   const double GmresRtol,
		   const int GmresRestart,
		   const int GmresMaxit,
		   const bool testT,
		   const int Tindex) {
	
	const int N = x0.size(); 
	VectorXd x(x0);
	std::vector<double> errVec;

	bool fail = false;
	for(size_t i = 0; i < maxit; i++){
	    VectorXd F = fx(x);
	    double Fnorm = F.norm();
	    
	    if(HOOK_PRINT && i % HOOK_PRINT_FREQUENCE == 0)
		fprintf(stderr, "\n+++++++++++ GHOOK: i = %zd/%d, r = %g ++++++++++ \n", i, maxit, Fnorm);

	    errVec.push_back(Fnorm);
	    if(Fnorm < tol) return std::make_tuple(x, errVec, 0);

	    // use GmresRPO to solve F' dx = -F
	    VectorXd s, sold, p;
	    ArrayXd D;
	    MatrixXd V, V2;
	    std::vector<double> e;
	    int flag;
	    auto Ax = [&x, &jacv, &Pre](const VectorXd &t){ VectorXd y = Pre(x, t); 
							    VectorXd z = jacv(x, y);
							    return z; };
	    std::tie(s, sold, p, D, V2, V, e, flag) = 
		Gmres0SVD(Ax, -F, VectorXd::Zero(N), GmresRestart, GmresMaxit, GmresRtol); 
	    if(flag != 0) fprintf(stderr, "GMRES SVD not converged !\n");
	    
	    ArrayXd D2 = D * D;
	    ArrayXd pd = p.array() * D;
	    ArrayXd mu = ArrayXd::Ones(p.size()) * 0.1 * D2.minCoeff(); 
	    for(size_t j = 0; j < maxInnIt; j++){ 
		VectorXd newx = x + Pre(x, s);

		double newT = newx(N - Tindex); 

		if(HOOK_PRINT && i % HOOK_PRINT_FREQUENCE == 0)	    
		    fprintf(stderr, " %zd, %g |", j, newT);

		if(!testT || newT > 0){
		    VectorXd newF = fx(newx); 
		    if(newF.norm() < (1 - minRD)*Fnorm){
			x = newx;
			break;
		    }
		}
		ArrayXd z = pd / (D2 + mu);
		VectorXd y = V2 * z.matrix();
		s = sold + V * y;
		mu *= 2;
		
		if(j == maxInnIt-1) fail = true;
	    }
	    
	    if(fail) break; // if all inner loop finish, it means state not changed
			    // then no need to iterate more.
	}

	return std::make_tuple(x, errVec, 0);
	    
    }

    template<typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    GmresHook( const Mat &A, const VectorXd &b,
	       const VectorXd &x0,
	       const double tol,
	       const double minRD,
	       const int maxit,
	       const int maxInnIt,
	       const double GmresRtol,
	       const int GmresRestart,
	       const int GmresMaxit,
	       const bool testT,
	       const int Tindex){
	return Gmres0Hook([&A, &b](const VectorXd &x){return A * x - b;},
			  [&A](const VectorXd &x, const VectorXd &t){return A * t;},
			  x0, tol, minRD, maxit, maxInnIt, GmresRtol,
			  GmresRestart, GmresMaxit, testT, Tindex);
    }

    /* use inverse of diagonal matrix as preconditioner */
    template<typename Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    GmresHookPre( const Mat &A, const VectorXd &b,
		  const VectorXd &x0,
		  const double tol,
		  const double minRD,
		  const int maxit,
		  const int maxInnIt,
		  const double GmresRtol,
		  const int GmresRestart,
		  const int GmresMaxit,
		  const bool testT,
		  const int Tindex){
	int n = A.cols();
	ArrayXd D(n);
	for(int i = 0; i < n; i++) D(i) = abs(A(i, i)) > 1 ? 1 / A(i,i) : 1;
	//for(int i = 0; i < n; i++) D(i) = 2;
	// cout << D << endl << endl;

	auto fx = [&A, &b](const VectorXd &x){return A * x - b;};
	auto jacv = [&A](const VectorXd &x, const VectorXd &t){return A * t;};
	auto Pre = [&D](const VectorXd &x){VectorXd t = D*x.array(); return t; };
	return Gmres0HookPre(fx, jacv, Pre,			     
			     x0, tol, minRD, maxit, maxInnIt, GmresRtol,
			     GmresRestart, GmresMaxit, testT, Tindex);
    }
    
    //////////////////////////////////////////////////////////////////////
    //                    Netwon methods related                        //
    //////////////////////////////////////////////////////////////////////

    
    /**
     * @brief Inexact Newton Backtracking method
     *
     *   try to solve problem F(x) = 0 
     *--------------------------------------------------------------------- 
     *        "Inexact Newton Methods Applied to Under–Determined Systems
     *                    by. Joseph P. Simonis. "
     *
     * Algorithm BINMU: 
     *     Let x0 and t in (0,1), eta_max in [0,1), and
     *         0 < theta_min< theta_max < 1 be given.
     *         
     *     For k = 0, 1, 2, ...; do
     *
     *         Find some bar_eta_k in [0, eta_max] and bar_s_k that satisfy
     *            || F(x_k) + F′(x_k) * bar_s_k ||  ≤ bar_eta_k * ||F(x_k)||,
     * 
     *         Evaluate F(x_k+ s_k). Set eta_k = bar_eta_k,  s_k = bar_s_k.
     *         While || F(x_k+ s_k) || > [1 − t(1 − eta_k)] * ||F(x_k)||, do
     *               Choose theta in [theta_min, theta_max].
     *               Update s_k = theta * s_k and eta_k = 1 − theta(1 − eta_k).
     *               Evaluate F(x_k+ s_k)
     *               
     *         Set x_{k+1}= x_k+ s_k.
     *----------------------------------------------------------------------
     *
     * In the above, we use GMRES to find s_k :
     *   F'(x_k) * s_k = - F(x_k) with relative tolerance bar_eta_k.
     * Also, for simplicity, we choose bar_eta_k to be constant.
     *
     * @param[in] fx             evaluate f(x)
     * @param[in] jacv           perform J(x)*dx. Symtex is jacv(x, dx)
     * @param[in] x0             initial guess
     * @param[in] btMaxIt        maximal backtracking iteration number
     *                           theta_max ^ btMaxIt => least shrink if fails
     * @param[in] tol            convergence tollerance
     * @param[in] eta0           initial value of eta
     * @param[in] theta_min      minimal value of forcing parameter
     * @param[in] theta_max      maximal value of forcing parameter
     *                           set 0.5 => at least shrink 1/2
     * @param[in] GmresRestart   gmres restart number
     * @param[in] GmresMaxit     gmres maximal iteration number 
     */
    template<class Fx, class Jacv>
    std::tuple<VectorXd, std::vector<double>, int>
    InexactNewtonBacktrack( Fx &fx, Jacv &jacv,
			    const VectorXd &x0,
			    const double tol,
			    const int btMaxIt,
			    const int maxit,
			    const double eta0,
			    const double t,
			    const double theta_min,
			    const double theta_max,
			    const int GmresRestart,
			    const int GmresMaxit){
	const int N = x0.size();
	VectorXd x(x0);
	std::vector<double> errVec; // errors
	
	for (size_t i = 0; i < maxit; i++){

	    ////////////////////////////////////////////////
	    // test convergence first
	    VectorXd F = fx(x); 
	    double Fnorm = F.norm(); 

	    if(INB_OUT_PRINT && i % INB_OUT_PRINT_FREQUENCE == 0)
		fprintf(stderr, "\n+++++++++++ INB: i = %zd, r = %g ++++++++++ \n", i, Fnorm);

	    errVec.push_back(Fnorm);
	    if( Fnorm < tol) return std::make_tuple(x, errVec, 0);

	    ////////////////////////////////////////////////
	    //solve ||F + F's|| < eta * ||F||
	    double eta = eta0; 
	    std::tuple<VectorXd, std::vector<double>, int>
		tmp = Gmres0([&x, &jacv](const VectorXd &t){ return jacv(x, t); },
			     -F, VectorXd::Zero(N), GmresRestart, GmresMaxit, eta);
	    if(std::get<2>(tmp) != 0) {
		fprintf(stderr, "GMRES not converged ! \n");
	    }
	    VectorXd &s = std::get<0>(tmp); // update vector 
	    printf("GMRES : %lu  %g | resdiual %g  %g\n",
		   std::get<1>(tmp).size(), std::get<1>(tmp).back(),
		   (fx(x) + jacv(x, s)).norm()/Fnorm, fx(x+s).norm());
	    
	    ////////////////////////////////////////////////
	    // use back tracking method to find appropariate scale
	    double initgp0 = 2 * F.dot(jacv(x, s));
	    double theta = 1;
	    for(size_t j = 0; j < btMaxIt; j++){
		VectorXd F1 = fx(x + s);
		double F1norm = F1.norm();

		if(INB_IN_PRINT && j % INB_IN_PRINT_FREQUENCE == 0)
		    fprintf(stderr, "INB(inner): j = %zd, r = %g \n", j, F1norm);

		if(F1norm < (1 - t * (1 - eta)) * Fnorm ) break;
		double gp0 = theta * initgp0;
		theta = chooseTheta(Fnorm, F1norm, gp0, theta_min, theta_max);
		s *= theta;
		eta = 1 - theta * (1 - eta);
		
		if(j == btMaxIt - 1) fprintf(stderr, "INB backtrack not converged !\n");
	    }
	    x += s; 		// update x
	}

	// if finish the outer loop => not converged
	fprintf(stderr, "INB not converged !\n");
	return std::make_tuple(x, errVec, 1); 
    }
    
    
    //////////////////////////////////////////////////////////////////////
    //            Levenberg-Marquardt related                           //
    //////////////////////////////////////////////////////////////////////

    /**
     * @brief Levenberg-Marquardt algorithm to minimize ||f(x)||
     *
     * @param[in]  Fx          VectorXd (*)(VectorXd x) to evaluate f(x)
     * @param[in] JJF          std::tuple<Mat, Mat, VectorXd> (*)(VectorXd x) function to
     *                         evaulate J^T*J, diag(diag(J^T*J)), J^T*f(x)
     * @param[in] solver       a linear solver to solve Ax = b
     * @param[in] x0           initial guess
     * @param[in] tol          tolerance
     * @param[in] mxit         maximal number of iterations
     * @param[in] innerMaxit   maximal number of inner iteration number
     */
    template<class Fx, template<class> class CalJJ, class LinearSolver, class Mat>
    std::tuple<VectorXd, std::vector<double>, int>
    LM0( Fx & fx, CalJJ<Mat> &JJF, LinearSolver &solver, 
	 const VectorXd &x0, const double tol,
	 const int maxit, const int innerMaxit){

	double lam = 1;
	VectorXd x(x0);
	std::vector<double> res;

	for(int i = 0; i < maxit; i++){
	    if (lam > LM_MAX_LAMBDA) return std::make_tuple(x, res, 0);
	    
	    /////////////////////////////////////////////////
	    // judge stop or not 
	    VectorXd F = fx(x); 
	    double err = F.norm();
	    res.push_back(err);
	    if(err < tol){
		fprintf(stderr, "stops at error = %g\n", err);
		return std::make_tuple(x, res, 0);
	    }	    
	    if (LM_OUT_PRINT && i % LM_OUT_PRINT_FREQUENCE == 0)
		fprintf(stderr,  "+++ LM : out loop: i = %d/%d, err=%g +++ \n", i, maxit, err);
	 
	    ////////////////////////////////////////////////
	    // construct JJ and JF
	    std::tuple<Mat, Mat, VectorXd> tmp = JJF(x); 
	    Mat &jj = std::get<0>(tmp); 
	    Mat &d = std::get<1>(tmp);
	    VectorXd &jf = std::get<2>(tmp);
	    
	    for(int j = 0; j < innerMaxit; j++){
		
		Mat H = jj + lam * d; 
		int N = H.rows();

		// solve H x = -jf
		auto cg = ConjGradSSOR(H, -jf, solver, VectorXd::Zero(N), N, 1e-6);
		VectorXd &dx = cg.first;
		std::vector<double> &r = cg.second;
		if (LM_IN_PRINT)
		    printf("CG error %g, iteration number %zd\n", r.back(), r.size());
		
		// update the state or keep unchanged
		VectorXd xnew = x + dx;
		VectorXd Fnew = fx(xnew);
		double errnew = Fnew.norm();
		
		if (LM_IN_PRINT && i % LM_IN_PRINT_FREQUENCE == 0)
		    fprintf(stderr,  "LM : inner loop: j = %d/%d, err=%g\n", j, innerMaxit, errnew);
		
		if (errnew < err){
		    x = xnew;
		    lam /= LM_LAMBDA_DECREASE_FACTOR;
		    break;
		} 
		else {
		    lam *= LM_LAMBDA_INCREASE_FACTOR;
		    if (lam > LM_MAX_LAMBDA) {
			fprintf(stderr, "lambda = %g too large \n", lam);
			break;
		    }
		}
		
	    }
	    
	    
	}
	
	// run out of loop, which means it does not converge
	return std::make_tuple(x, res, 1);
    }

    /**
     * @brief JJF template function for LM() to solve Ax=b
     */
    template<class Mat>
    struct JJF {
	const Mat &J;
	const VectorXd &b;
	
	JJF(const Mat &tJ, const VectorXd &tb) : J(tJ), b(tb) {}
	
	std::tuple<MatrixXd, MatrixXd, VectorXd>
        operator()(const VectorXd &x) {
	    MatrixXd JJ = J.transpose() * J;
	    MatrixXd d = JJ.diagonal().asDiagonal();
	    return std::make_tuple(JJ, d, J.transpose()*(J*x-b));
	}
    };

}

#endif	/* ITERMETHOD_H */

