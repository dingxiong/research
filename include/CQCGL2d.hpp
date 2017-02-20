/* to compile the class
 *  
 *  compile with Mutithreads FFT support:
 *  g++ cqcgl2d.cc  -shared -fpic -march=corei7 -O3 -msse2 -msse4 -lfftw3_threads -lfftw3 -lm -fopenmp -DTFFT  -I $XDAPPS/eigen/include/eigen3 -I../include -std=c++0x
 *
 *  complile with only one thread:
 *  g++ cqcgl2d.cc -shared -fpic -lfftw3 -lm -fopenmp  -march=corei7 -O3 -msse2 -msse4 -I/usr/include/eigen3 -I../../include -std=c++0x
 *
 *  */
#ifndef CQCGL2D_H
#define CQCGL2D_H

// #include <fftw3.h>
#include <complex>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "denseRoutines.hpp"
#include "myH5.hpp"
#include "myfft.hpp"
using std::pair; using std::make_pair; 
using Eigen::MatrixXd; using Eigen::VectorXd;
using Eigen::MatrixXcd; using Eigen::VectorXcd;
using Eigen::ArrayXXcd; using Eigen::ArrayXcd;
using Eigen::ArrayXXd; using Eigen::ArrayXd;
using Eigen::ConjugateGradient;
using Eigen::PartialPivLU;
using Eigen::Map; using Eigen::Ref;

//////////////////////////////////////////////////////////////////////
//                       class CQCGLgeneral                         //
//////////////////////////////////////////////////////////////////////

/**
 * @brief two dimensional cubic quintic complex Ginzburg-Landau equation
 *
 * The dimension of the mesh is [M x N] corresponding to actual simulation
 * domain [dy x dx]. Therefore, the x direction is discretized to N points,
 * and y direction is divided to M points. Memory is contiguous in the
 * y direction.
 */
class CQCGL2d {
  
public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    const int N, M;		/* dimension of FFT */
    const double dx, dy;	/* system domain size */
    
    int Ne, Me;			/* effective number of modes */
    int Nplus, Nminus, Nalias;
    int Mplus, Mminus, Malias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;		/* used for comoving frame */
    ArrayXd Kx, Kx2, Ky, Ky2, QKx, QKy;

    ArrayXXcd L, E, E2, a21, a31, a32, a41, a43, b1, b2, b4;

    MyFFT::FFT2d F[5], JF[5];
    
    /* for time step adaptive ETDRK4 and Krogstad  */
    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-10;
    double nu = 0.9;		/* safe factor */
    double mumax = 2.5;		/* maximal time step increase factor */
    double mumin = 0.4;		/* minimal time step decrease factor */
    double mue = 1.25;		/* upper lazy threshold */
    double muc = 0.85;		/* lower lazy threshold */

    int NCalCoe = 0;	      /* times to evaluate coefficient */
    int NReject = 0;	      /* times that new state is rejected */
    int NCallF = 0;	      /* times to call velocity function f */
    int NSteps = 0;	      /* total number of integrations steps */
    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    
    int cellSize = 500;	/* size of cell when resize output container */
    int MC = 64;	/* number of sample points */
    int R = 1;		/* radius for evaluating phi(z) */

    int Method = 1;
    int constETDPrint = 0;
    ////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////
    CQCGL2d(int N, int M, double dx, double dy,
	    double Mu, double Dr, double Di,
	    double Br, double Bi, double Gr,
	    double Gi,  
	    int threadNum);
    ~CQCGL2d();
    CQCGL2d & operator=(const CQCGL2d &x);
    
    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////

    //============================================================    
    inline int calNe(const double N);
    void CGLInit();
    void changeOmega(double w);
    void calCoe(const double h);
    void oneStep(double &du, const bool onlyOrbit);
    ArrayXXcd ZR(const Ref<const ArrayXcd> &z);
    double adaptTs(bool &doChange, bool &doAccept, const double s);


    void saveState(H5::H5File &file, int id, const ArrayXXcd &a, 
		   const ArrayXXcd &v, const int flag);
    ArrayXXcd 
    constETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
	     const double h, const int Nt, 
	     const int skip_rate, const bool onlyOrbit,
	     const bool doSaveDisk, const std::string fileName);
    ArrayXXcd
    adaptETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
	     const double h0, const double tend, 
	     const int skip_rate, const bool onlyOrbit,
	     const bool doSaveDisk, const std::string fileName);
    ArrayXXcd 
    intg(const ArrayXXcd &a0, const double h, const int Nt, const int skip_rate,
	 const bool doSaveDisk, const std::string fileName);
    ArrayXXcd
    aintg(const ArrayXXcd &a0, const double h, const double tend, 
	  const int skip_rate, const bool doSaveDisk, const std::string fileName);
    ArrayXXcd
    intgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
	  const int Nt, const int skip_rate,
	  const bool doSaveDisk, const std::string fileName);
    ArrayXXcd 
    aintgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
	   const double tend, const int skip_rate,
	   const bool doSaveDisk, const std::string fileName);
    
    void dealias(const int k, const bool onlyOrbit);
    void NL(const int k, const bool onlyOrbit);
    ArrayXXcd unpad(const ArrayXXcd &v);
    ArrayXXcd pad(const ArrayXXcd &v);
    ArrayXd c2r(const ArrayXXcd &v);
    ArrayXXcd r2c(const ArrayXd &v);
    
    //============================================================  

    ArrayXXcd Fourier2Config(const Ref<const ArrayXXcd> &aa);
    ArrayXXcd Config2Fourier(const Ref<const ArrayXXcd> &AA);
    
    ArrayXXcd velocity(const ArrayXXcd &a0);
    ArrayXXcd velocityReq(const ArrayXXcd &a0, const double wthx,
			  const double wthy, const double wphi);
    ArrayXXcd stab(const ArrayXXcd &a0, const ArrayXXcd &v0);
    ArrayXXcd stabReq(const ArrayXXcd &a0, const ArrayXXcd &v0,
		      const double wthx, const double wthy,
		      const double wphi);
    
    ArrayXXcd rotate(const Ref<const ArrayXXcd> &a0, const int mode, const double th1 = 0,
		     const double th2 = 0, const double th3 = 0);
    ArrayXXcd tangent(const Ref<const ArrayXXcd> &a0, const int mode);


    std::tuple<ArrayXXcd, double, double, double, double>
    readReq(const std::string fileName, const std::string groupName);
    
    
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2sliceWrap(const Ref<const ArrayXXd> &aa);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2slice(const Ref<const ArrayXXd> &aa);
    ArrayXXd orbit2sliceSimple(const Ref<const ArrayXXd> &aa);
    MatrixXd ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x);

};



#endif  /* CQCGL2D_H */
