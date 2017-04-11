#ifndef KSINT_H
#define KSINT_H

#include <fftw3.h>
#include <complex>
#include <tuple>
#include <utility>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include "EIDr.hpp"

/*============================================================
 *                       Class : KS integrator
 *============================================================*/
class KS{
  
public:
    typedef std::complex<double> dcp;

    //////////////////////////////////////////////////////////////////////
    /* member variables */
    const int N;
    const double d;

    ArrayXd K, L;
    ArrayXcd G;
    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;	/* size of cell when resize output container */
    
    struct NL {
	KS *ks;
	const int N;
	ArrayXXd u;
	NL();
	NL(KS *ks, int cols);
	~NL();
	void operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t);
    };
    
    NL nl, nl2;
    ArrayXXcd Yv[10], Nv[10], Yv2[10], Nv2[10];
    EIDr eidc, eidc2;;
    ////////////////////////////////////////////////////////////
    

    //////////////////////////////////////////////////////////////////////
    /* constructor, destructor, copy assignment */
    KS(int N, double d);
    KS & operator=(const KS &x);
    ~KS();
  
    /* ------------------------------------------------------------ */
    void 
    setScheme(std::string x);
    ArrayXXd 
    intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgjC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd 
    C2R(const Ref<const ArrayXXcd> &v);
    ArrayXXcd 
    R2C(const Ref<const ArrayXXd> &v);
    /* ------------------------------------------------------------ */
    
    VectorXd 
    velocity(const Ref<const ArrayXd> &a0);
    VectorXd
    velReq(const Ref<const VectorXd> &a0, const double c);
    MatrixXd 
    stab(const Ref<const ArrayXd> &a0);
    MatrixXd 
    stabReq(const Ref<const VectorXd> &a0, const double theta);
    std::pair<VectorXcd, Eigen::MatrixXcd>
    evEq(const Ref<const VectorXd> &a0);
    std::pair<VectorXcd, Eigen::MatrixXcd>
    evReq(const Ref<const VectorXd> &a0, const double theta);
    
    double 
    pump(const ArrayXcd &vc);
    double 
    disspation(const ArrayXcd &vc);
    ArrayXXd 
    reflect(const Ref<const ArrayXXd> &aa);
    ArrayXXd 
    half2whole(const Ref<const ArrayXXd> &aa);
    ArrayXXd 
    rotate(const Ref<const ArrayXXd> &aa, const double th);
    ArrayXXd
    gTangent(const Ref<const ArrayXXd> &x);
    MatrixXd 
    gGenerator();
    std::pair<MatrixXd, VectorXd> 
    orbitToSlice(const Ref<const MatrixXd> &aa);
    MatrixXd 
    veToSlice(const MatrixXd &ve, const Ref<const VectorXd> &x);
    MatrixXd 
    veToSliceAll(const MatrixXd &eigVecs, const MatrixXd &aa,
		 const int trunc = 0);
    std::pair<ArrayXXd, ArrayXXd>
    orbitAndFvWhole(const ArrayXd &a0, const ArrayXXd &ve,
		    const double h,
		    const size_t nstp, const std::string ppType
		    );
    MatrixXd veTrunc(const MatrixXd ve, const int pos, const int trunc = 0);
    std::pair<ArrayXXd, ArrayXXd>
    orbitAndFvWholeSlice(const ArrayXd &a0, const ArrayXXd &ve,
			 const double h,
			 const size_t nstp, const std::string ppType,
			 const int pos
			 );
    std::vector<int> reflectIndex();
    ArrayXXd reduceReflection(const Ref<const ArrayXXd> &aaHat);
    MatrixXd GammaMat(const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVe(const MatrixXd &ve, const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
			  const int trunc = 0);

    MatrixXd calMag(const Ref<const MatrixXd> &aa);
    std::pair<MatrixXd, MatrixXd>
    toPole(const Ref<const MatrixXd> &aa);
    MatrixXcd
    a2f(const Ref<const MatrixXd> &aa);
    MatrixXd
    f2a(const Ref<const MatrixXcd> &f);

    std::pair<MatrixXd, VectorXd>
    redSO2(const Ref<const MatrixXd> &aa, const int p, const bool toY);
    std::pair<MatrixXd, VectorXi>
    fundDomain(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund);
    std::tuple<MatrixXd, VectorXi, VectorXd>
    redO2f(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund);
    MatrixXd redR1(const Ref<const MatrixXd> &aa);
    MatrixXd redR2(const Ref<const MatrixXd> &cc);
    MatrixXd redRef(const Ref<const MatrixXd> &aa);
    std::pair<MatrixXd, VectorXd>
    redO2(const Ref<const MatrixXd> &aa, const int p, const bool toY);
    MatrixXd Gmat1(const Ref<const VectorXd> &x);
    MatrixXd Gmat2(const Ref<const VectorXd> &x);
    MatrixXd
    redV(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a,
	 const int p, const bool toY);
    MatrixXd redV2(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a);

};


#endif	/* KSINT_H */
