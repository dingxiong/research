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

class CQCGL1dRpo : public CQCGL1d {

public:

    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;

    int M;			/* pieces of multishoot */

    double h0Trial = 1e-3;
    int skipRateTrial = 1000000;
    
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
    void writeRpo(const string fileName, const string groupName,
		  const MatrixXd &x, const double T, const int nstp,
		  const double th, const double phi, double err);    

    static
    void writeRpo2(const std::string fileName, const string groupName, 
			   const MatrixXd &x, const int nstp, double err);

    static
    std::tuple<MatrixXd, double, int, double, double, double>
    readRpo(const string fileName, const string groupName);

    static
    void
    moveRpo(string infile, string ingroup, 
	    string outfile, string outgroup);

    static
    std::string 
    toStr(double x);
    
    static
    std::tuple<MatrixXd, double, int, double, double, double>
    readRpo(const string fileName, double di, int index);
    
    static
    void 
    writeRpo(const string fileName, double di, int index,
	     const MatrixXd &x, const double T, const int nstp,
	     const double th, const double phi, double err);

    static
    void 
    writeRpo2(const string fileName, double di, int index,
	      const MatrixXd &x, const int nstp,
	      double err);
    
    static
    void 
    moveRpo(string infile, string ingroup,
	    string outfile, double di, int index);

    static
    void 
    moveRpo(string infile, string outfile, double di, int index);

    ////////////////////////////////////////////////////////////  

    /**
     * @brief         form g*f(x,t) - x
     * @param[in] x   [Ndim + 3, 1] dimensional vector: (x, t, theta, phi)
     * @return        vector F(x, t) =
     *                  | g*f(x, t) - x|
     *                  |       0      |
     *                  |       0      |
     *                  |       0      |
     */
    template<int nstp>
    VectorXd Fx(const VectorXd & x){
	VectorXd a0 = x.head(Ndim);
	Vector3d t = x.tail<3>();
	assert(t(0) > 0); 		/* make sure T > 0 */
	VectorXd fx = intg(a0, t(0)/nstp, nstp, nstp).rightCols<1>();
	VectorXd F(Ndim + 3);
	F << Rotate(fx, t(1), t(2)) - a0, 0, 0, 0;
	return F;
    }



    /**
     * @brief get the product J * dx
     *
     * Here J = | g*J(x, t) - I,  g*v(f(x,t)),  g*t1(f(x,t)),  g*t2(f(x,t))| 
     *          |     v(x),          0             0                  0    |
     *          |     t1(x),         0             0                  0    |
     *          |     t2(x),         0             0                  0    |
     */
    template<int nstp>
    VectorXd DFx(const VectorXd &x, const VectorXd &dx){
	VectorXd a0 = x.head(Ndim);
	Vector3d t = x.tail<3>();
	VectorXd da0 = dx.head(Ndim);
	Vector3d dt = dx.tail<3>();
	assert(t(0) > 0); 		/* make sure T > 0 */
	ArrayXXd tmp = intgv(a0, da0, t(0)/nstp, nstp); /* f(x, t) and J(x, t)*dx */
	ArrayXd gfx = Rotate(tmp.col(0), t(1), t(2)); /* g(theta, phi)*f(x, t) */
	ArrayXd gJx = Rotate(tmp.col(1), t(1), t(2)); /* g(theta, phi)*J(x,t)*dx */
	ArrayXd v1 = velocity(a0);	       /* v(x) */
	ArrayXd v2 = velocity(tmp.col(0)); /* v(f(x, t)) */
	ArrayXd t1 = transTangent(a0);
	ArrayXd t2 = phaseTangent(a0);
	VectorXd DF(Ndim + 3);
	DF << gJx.matrix() - a0
	    + cgl2.Rotate(v2, t(1), t(2)).matrix() * dt(0)
	    + cgl2.transTangent(gfx).matrix() * dt(1)
	    + cgl2.phaseTangent(gfx).matrix() * dt(2),
	
	    v1.matrix().dot(da0),
	    t1.matrix().dot(da0),
	    t2.matrix().dot(da0)
	    ;

	return DF;
    }

    /* 
     * @brief multishooting form [f(x_0, t) - x_1, ... g*f(x_{M-1},t) - x_0]
     * @param[in] x   [Ndim * M + 3, 1] dimensional vector: (x, t, theta, phi)
     * @return    vector F(x, t) =
     *               |   f(x_0, t) -x_1     |
     *               |   f(x_1, t) -x_2     |
     *               |     ......           |
     *               | g*f(x_{M-1}, t) - x_0|
     *               |       0              |
     *               |       0              |
     *               |       0              |
     *
     */
    VectorXd CQCGL1dRpo::MFx(const VectorXd &x){
	Vector3d t = x.tail<3>();	   /* T, theta, phi */
	assert(t(0) > 0);		   /* make sure T > 0 */
	VectorXd F(VectorXd::Zero(Ndim * M + 3));

	for(size_t i = 0; i < M; i++){
	    VectorXd fx = cgl1.intg(x.segment(i*Ndim, Ndim), nstp, nstp).rightCols<1>();
	    if(i != M-1){		// the first M-1 vectors
		F.segment(i*Ndim, Ndim) = fx - x.segment((i+1)*Ndim, Ndim);
	    }
	    else{			// the last vector
		F.segment(i*Ndim, Ndim) = cgl1.Rotate(fx, t(1), t(2)).matrix() - x.head(Ndim);
	    }
	}
    
	return F;
    }
    
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
    
    CQCGL1dRpo &rpo;
    
    
    cqcglJJF(CQCGL1dRpo &rpo_) : rpo(rpo_){}
    
    std::tuple<SpMat, SpMat, VectorXd>
    operator()(const VectorXd &x) {
	return rpo.calJJF(x);
    }	
};



#endif	/* CQCGL1dRpo */
