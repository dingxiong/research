#ifndef CQCGLRPO_H
#define CQCGLRPO_H

#include <complex>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "CQCGL.hpp"
#include "iterMethod.hpp"
#include "sparseRoutines.hpp"
#include "denseRoutines.hpp"

class CQCGLRPO {

public:

    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;

    CQCGL cgl1, cgl2, cgl3;
    int M;			/* pieces of multishoot */
    const int N;		/* dimension of FFT */
    int Ndim;			/* dimension of state space */

    double h0Trial = 1e-3;
    int skipRateTrial = 1000000;
    double Omega = 0;
    
    /*---------------   constructors    ------------------------- */
    CQCGLRPO(int M, int N, double d, 
	     double b, double c, double dr, double di,
	     int threadNum);
    
    ~CQCGLRPO();
    CQCGLRPO & operator=(const CQCGLRPO &x);

    /*---------------  member functions ------------------------- */
    void changeOmega(double w);

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

    VectorXd calPre(const VectorXd &x, const VectorXd &dx);
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

    
};

template<class Mat>
struct cqcglJJF {
    typedef Eigen::SparseMatrix<double> SpMat;
    
    CQCGLRPO &rpo;
    
    
    cqcglJJF(CQCGLRPO &rpo_) : rpo(rpo_){}
    
    std::tuple<SpMat, SpMat, VectorXd>
    operator()(const VectorXd &x) {
	return rpo.calJJF(x);
    }	
};



#endif	/* CQCGLRPO */
