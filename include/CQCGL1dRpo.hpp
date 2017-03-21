#ifndef CQCGL1dRpo_H
#define CQCGL1dRpo_H

#include <complex>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "CQCGL1d.hpp"
#include "iterMethod.hpp"
#include "sparseRoutines.hpp"
#include "denseRoutines.hpp"

using Eigen::Vector3d;

class CQCGL1dRpo : public CQCGL1d {

public:

    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;

    ////////////////////////////////////////////////////////////
    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1dRpo(int N, double d,
	       double Mu, double Dr, double Di, double Br, double Bi, 
	       double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1dRpo(int N, double d, 
	       double b, double c, double dr, double di, 
	       int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1dRpo(int N, double d,
	       double delta, double beta, double D, double epsilon,
	       double mu, double nu, int dimTan);
    ~CQCGL1dRpo();
    CQCGL1dRpo & operator=(const CQCGL1dRpo &x);
    
    
    ////////////////////////////////////////////////////////////    
    static
    std::string toStr(double x);
    
    static
    std::string toStr(double x, double y, int id);

    static
    void write(const std::string fileName, const std::string groupName,
	       const MatrixXd &x, const double T, const int nstp,
	       const double th, const double phi, double err);    

    static
    void write2(const std::string fileName, const std::string groupName, 
		const MatrixXd &x, const int nstp, double err);

    static
    std::tuple<MatrixXd, double, int, double, double, double>
    read(const std::string fileName, const std::string groupName);

    static
    void
    move(std::string infile, std::string ingroup, 
	 std::string outfile, std::string outgroup, int flag);

    static
    VectorXcd 
    readE(std::string fileName, const std::string groupName);

    static
    MatrixXd 
    readV(std::string fileName, const std::string groupName);
    
    static
    void 
    writeE(std::string fileName, const std::string groupName, 
	   const VectorXcd &e);

    static
    void 
    writeV(std::string fileName, const std::string groupName, 
	   const MatrixXd &v);
    
    ////////////////////////////////////////////////////////////  


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

    std::tuple<MatrixXd, double, int>
    findRPOM_hook2(const MatrixXd &x0, 
		   const int nstp,
		   const double tol,
		   const double minRD,
		   const int maxit,
		   const int maxInnIt,
		   const double GmresRtol,
		   const int GmresRestart,
		   const int GmresMaxit);

    void 
    findRpoParaSeq(const std::string file, int id, double step, int Ns, bool isBi,
		   int nstpFlag, int paraNstp);
    
    std::tuple<SpMat, SpMat, VectorXd> 
    calJJF(const VectorXd &x);
    std::tuple<MatrixXd, double>
    findRPOM_LM(const MatrixXd &x0, 
		const double tol,
		const int maxit,
		const int innerMaxit);
    
    ////////////////////////////////////////////////////////////
    VectorXd MFx(const VectorXd &x, int nstp);
    VectorXd MDFx(const VectorXd &x, const VectorXd &dx, int nstp);
    VectorXd MFx2(const VectorXd &x, int nstp);
    VectorXd MDFx2(const VectorXd &x, const VectorXd &dx, int nstp);
    
};

template<class Mat>
struct cqcglJJF {
    typedef Eigen::SparseMatrix<double> SpMat;
    
    CQCGL1dRpo &rpo;
    
    
    cqcglJJF(CQCGL1dRpo &rpo_) : rpo(rpo_){}
    
    std::tuple<SpMat, SpMat, VectorXd>
    operator()(const VectorXd &x) {
	return rpo.calJJF(x);
    }	
};



#endif	/* CQCGL1dRpo */
