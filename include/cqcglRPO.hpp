#ifndef CQCGLRPO
#define CQCGLRPO

#include <complex>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <omp.h>
#include "cqcgl1d.hpp"
#include "iterMethod.hpp"
#include "sparseRoutines.hpp"
#include "denseRoutines.hpp"

class CqcglRPO{
    public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;

    Cqcgl1d cgl1, cgl2, cgl3;
    int nstp;			/* integration steps for each piece */
    int M;			/* pieces of multishoot */
    const int N;		/* dimension of FFT */
    int Ndim;			/* dimension of state space */

    // Non-static Data Member Initializers => new feature of C++11
    double alpha1 = 0.01;	/* strength scale for v constraint */
    double alpha2 = 0.01;	/* strength scale for t1 constraint */
    double alpha3 = 0.01;	/* strength scale for t2 constraint */

    double Eps = 1e-4;
    MatrixXd V;
    
    
    /*---------------   constructors    ------------------------- */
    CqcglRPO(int nstp, int M,
	     int N = 512, double d = 50, double h = 0.01,
	     double Mu = -0.1, double Br = 1.0, double Bi = 0.8,
	     double Dr = 0.125, double Di = 0.5, double Gr = -0.1,
	     double Gi = -0.6, int threadNum = 4);
    CqcglRPO(int nstp, int M,
	     int N, double d, double h,
	     double b, double c,
	     double dr, double di,
	     int threadNum);
    
    ~CqcglRPO();
    CqcglRPO & operator=(const CqcglRPO &x);

    /*---------------  member functions ------------------------- */
    VectorXd Fx(const VectorXd & x);
    VectorXd DFx(const VectorXd &x, const VectorXd &dx);
    VectorXd MFx(const VectorXd &x);
    VectorXd MDFx(const VectorXd &x, const VectorXd &dx);
    
    std::tuple<VectorXd, double, double, double, double>
    findRPO(const VectorXd &x0, const double T,
	    const double th0, const double phi0,
	    const double tol = 1e-12,
	    const int btMaxIt = 20,
	    const int maxit = 100,
	    const double eta0 = 1e-4,
	    const double t = 1e-4,
	    const double theta_min = 0.1,
	    const double theta_max = 0.5,
	    const int GmresRestart = 30,
	    const int GmresMaxit = 100);

    std::tuple<MatrixXd, double, double, double, double>
    findRPOM(const MatrixXd &x0, const double T,
	     const double th0, const double phi0,
	     const double tol = 1e-12,
	     const int btMaxIt = 20,
	     const int maxit = 100,
	     const double eta0 = 1e-4,
	     const double t = 1e-4,
	     const double theta_min = 0.1,
	     const double theta_max = 0.5,
	     const int GmresRestart = 30,
	     const int GmresMaxit = 100);

    std::tuple<VectorXd, double, double, double, double>
    findRPO_hook(const VectorXd &x0, const double T,
		 const double th0, const double phi0,
		 const double tol,
		 const double minRD,
		 const int maxit,
		 const int maxInnIt,
		 const double GmresRtol,
		 const int GmresRestart,
		 const int GmresMaxit);
    
    std::tuple<MatrixXd, double, double, double, double>
    findRPOM_hook(const MatrixXd &x0, const double T,
		  const double th0, const double phi0,
		  const double tol = 1e-12,
		  const double minRD = 1e-3,
		  const int maxit = 100,
		  const int maxInnIt = 20,
		  const double GmresRtol = 1e-6,
		  const int GmresRestart = 100,
		  const int GmresMaxit = 100);

    VectorXd MFx2(const VectorXd &x);
    VectorXd MDFx2(const VectorXd &x, const VectorXd &dx);
    std::tuple<MatrixXd, double>
    findRPOM_hook2(const MatrixXd &x0, 
		   const double tol,
		   const double minRD,
		   const int maxit,
		   const int maxInnIt,
		   const double GmresRtol,
		   const int GmresRestart,
		   const int GmresMaxit);

    std::tuple<SpMat, SpMat, VectorXd> 
    calJJF(const VectorXd &x);
    std::tuple<MatrixXd, double>
    findRPOM_LM(const MatrixXd &x0, 
		const double tol,
		const int maxit,
		const int innerMaxit);

#if 0				
    inline VectorXd cgSolver(ConjugateGradient<SpMat> &CG, Eigen::SparseLU<SpMat> &solver,
			     SpMat &H, VectorXd &JF, bool doesUseMyCG = true,
			     bool doesPrint =  true);
    std::tuple<ArrayXXd, double, double, double, double>
    findPO(const ArrayXXd &aa0, const double h0, const int nstp,
	   const double th0, const double phi0,
	   const int MaxN = 200,
	   const double tol = 1e-13,
	   const bool doesUseMyCG = true,
	   const bool doesPrint = false);
#endif
    
};

template<class Mat>
struct cqcglJJF {
    typedef Eigen::SparseMatrix<double> SpMat;
    
    CqcglRPO &rpo;
    
    
    cqcglJJF(CqcglRPO &rpo_) : rpo(rpo_){}
    
    std::tuple<SpMat, SpMat, VectorXd>
    operator()(const VectorXd &x) {
	return rpo.calJJF(x);
    }	
};



#endif	/* CQCGLRPO */
