#ifndef CQCGLRPO
#define CQCGLRPO

#include <complex>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "cqcgl1d.hpp"

class CqcglRPO{
    public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;

    Cqcgl1d cgl1, cgl2;
    int nstp;
    const int Ndim;
    
    CqcglRPO(int nstp, int N = 512, double d = 50, double h = 0.01,
	    double Mu = -0.1, double Br = 1.0, double Bi = 0.8,
	    double Dr = 0.125, double Di = 0.5, double Gr = -0.1,
	    double Gi = -0.6, int threadNum = 4);
    
    ~CqcglRPO();
    CqcglRPO & operator=(const CqcglRPO &x);

    VectorXd Fx(const VectorXd & x);
    VectorXd DFx(const VectorXd &x, const VectorXd &dx);
    std::tuple<VectorXd, double, double, double, double>
    findRPO(const VectorXd &x0, const double T, const int nstp,
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
}


#endif	/* CQCGLRPO */
