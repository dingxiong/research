/* to compile the class
 *  
 *  compile with Mutithreads FFT support:
 *  g++ cqcgl1d.cc  -shared -fpic -march=corei7 -O3 -msse2 -msse4 -lfftw3_threads -lfftw3 -lm -fopenmp -DTFFT  -I $XDAPPS/eigen/include/eigen3 -I../include -std=c++0x
 *
 *  complile with only one thread:
 *  g++ cqcgl1d.cc -shared -fpic -lfftw3 -lm -fopenmp  -march=corei7 -O3 -msse2 -msse4 -I/usr/include/eigen3 -I../../include -std=c++0x
 *
 *  */
#ifndef CQCGL1D_H
#define CQCGL1D_H

#include <complex>
#include <utility>
#include <algorithm>
#include <vector>
#include <unsupported/Eigen/FFT>
#include "denseRoutines.hpp"
#include "EIDc.hpp"

//////////////////////////////////////////////////////////////////////
//                       class CQCGL1d                              //
//////////////////////////////////////////////////////////////////////
class CQCGL1d {
  
public:
    typedef std::complex<double> dcp;
    
    const int N;		/* dimension of FFT */
    const double d;		/* system domain size */
    
    const int Ne;		/* effective number of modes */
    const int Ndim;		/* dimension of state space */
    const int Nplus, Nminus, Nalias;
    const int DimTan;		// the dimension of tangent space
    
    bool IsQintic = true;	//  False => cubic equation
    
    double Mu, Dr, Di, Br, Bi, Gr, Gi;
    double Omega = 0;		/* used for comoving frame */
    ArrayXd K, K2, QK;

    ArrayXcd L;

    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;	/* size of cell when resize output container */

    struct NL {
	CQCGL1d *cgl;
	int N;
	dcp B, G;
	ArrayXXcd AA;
	
	NL();
	NL(CQCGL1d *cgl, int cols);
	~NL();
	void operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t);
    };
    
    NL nl, nl2;
    
    ArrayXXcd Yv[10], Nv[10], Yv2[10], Nv2[10];
    EIDc eidc, eidc2;
    
    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////

    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1d(int N, double d,
	    double Mu, double Dr, double Di, double Br, double Bi, 
	    double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1d(int N, double d, 
	    double b, double c, double dr, double di, 
	    int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1d(int N, double d,
	    double delta, double beta, double D, double epsilon,
	    double mu, double nu, int dimTan);
    ~CQCGL1d();
    CQCGL1d & operator=(const CQCGL1d &x);

    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////

    //============================================================    
    void setScheme(std::string x);
    void changeOmega(double w);

    ArrayXXd 
    intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgjC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd
    intgvC(const ArrayXd &a0, const ArrayXXd &v, const double h, const double tend);
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd 
    intgv(const ArrayXXd &a0, const ArrayXXd &v, const double h, const double tend);
    
    ArrayXXd C2R(const ArrayXXcd &v);
    ArrayXXcd R2C(const ArrayXXd &v);
    ArrayXXd c2r(const ArrayXXcd &v);
    ArrayXXcd r2c(const ArrayXXd &v);
    
    //============================================================  

    ArrayXXcd Fourier2Config(const Ref<const ArrayXXd> &aa);
    ArrayXXd Config2Fourier(const Ref<const ArrayXXcd> &AA);
    ArrayXXd Fourier2ConfigMag(const Ref<const ArrayXXd> &aa);
    ArrayXXd calPhase(const Ref<const ArrayXXcd> &AA);
    ArrayXXd Fourier2Phase(const Ref<const ArrayXXd> &aa);
    VectorXd calQ(const Ref<const ArrayXXd> &aa);
    VectorXd calMoment(const Ref<const ArrayXXd> &aa, const int p = 1);
    
    ArrayXd velocity(const ArrayXd &a0);
    ArrayXd velocityReq(const ArrayXd &a0, const double th,
			const double phi);
    VectorXd velSlice(const Ref<const VectorXd> &aH);
    VectorXd velPhase(const Ref<const VectorXd> &aH);
    MatrixXd rk4(const VectorXd &a0, const double dt, const int nstp, const int nq);
    MatrixXd velJ(const MatrixXd &xj);
    std::pair<MatrixXd, MatrixXd>
    rk4j(const VectorXd &a0, const double dt, const int nstp, const int nq, const int nqr);
    
    ArrayXcd
    Lyap(const Ref<const ArrayXXd> &aa);
    ArrayXd
    LyapVel(const Ref<const ArrayXXd> &aa);
    MatrixXd stab(const ArrayXd &a0);
    MatrixXd stabReq(const ArrayXd &a0, double th, double phi);
    VectorXcd eReq(const ArrayXd &a0, double wth, double wphi);
    MatrixXcd vReq(const ArrayXd &a0, double wth, double wphi);
    std::pair<VectorXcd, MatrixXcd>
    evReq(const ArrayXd &a0, double wth, double wphi);

    ArrayXXd reflect(const Ref<const ArrayXXd> &aa);
    inline ArrayXd rcos2th(const ArrayXd &x, const ArrayXd &y);
    inline ArrayXd rsin2th(const ArrayXd &x, const ArrayXd &y);
    inline double rcos2thGrad(const double x, const double y);
    inline double rsin2thGrad(const double x, const double y);
    ArrayXXd reduceRef1(const Ref<const ArrayXXd> &aaHat);
    ArrayXXd reduceRef2(const Ref<const ArrayXXd> &step1);
    std::vector<int> refIndex3();
    ArrayXXd reduceRef3(const Ref<const ArrayXXd> &aa);
    ArrayXXd reduceReflection(const Ref<const ArrayXXd> &aaHat);
    MatrixXd refGrad1();
    MatrixXd refGrad2(const ArrayXd &x);
    MatrixXd refGrad3(const ArrayXd &x);
    MatrixXd refGradMat(const ArrayXd &x);
    MatrixXd reflectVe(const MatrixXd &veHat, const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
			  const int trunc = 0);
    
    ArrayXXd transRotate(const Ref<const ArrayXXd> &aa, const double th);
    ArrayXXd transTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd transGenerator();

    ArrayXXd phaseRotate(const Ref<const ArrayXXd> &aa, const double phi);
    ArrayXXd phaseTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd phaseGenerator();

    ArrayXXd Rotate(const Ref<const ArrayXXd> &aa, const double th, const double phi);
    ArrayXXd rotateOrbit(const Ref<const ArrayXXd> &aa, const ArrayXd &th,
			 const ArrayXd &phi);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2slice(const Ref<const ArrayXXd> &aa, const int method);
    MatrixXd ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    reduceAllSymmetries(const Ref<const ArrayXXd> &aa, int flag);
    MatrixXd reduceVe(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag);

    std::tuple<ArrayXd, double, double>
    planeWave(int k, bool isPositve);
    VectorXcd planeWaveStabE(int k, bool isPositve);
    std::pair<VectorXcd, MatrixXcd>
    planeWaveStabEV(int k, bool isPositve);
};



#endif  /* CQCGL1D_H */
