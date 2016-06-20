#include <iostream>
#include <functional>
#include "CQCGL2dReq.hpp"
#include "iterMethod.hpp"

#define cee(x) (cout << (x) << endl << endl)

namespace ph = std::placeholders;

using namespace std;
using namespace Eigen;
using namespace iterMethod;
using namespace denseRoutines;
using namespace MyH5;

//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////
CQCGL2dReq::CQCGL2dReq(int N, int M, double dx, double dy,
		       double b, double c, double dr, double di,
		       int threadNum) 
    : CQCGL2d(N, M, dx, dy, b, c, dr, di, threadNum) 
{
    calPre();
}

CQCGL2dReq::CQCGL2dReq(int N, double dx,
		       double b, double c, double dr, double di,
		       int threadNum)
    : CQCGL2d(N, dx, b, c, dr, di, threadNum)
{
    calPre();
}

CQCGL2dReq & CQCGL2dReq::operator=(const CQCGL2dReq &x){
    return *this;
}

CQCGL2dReq::~CQCGL2dReq(){}

//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////

void 
CQCGL2dReq::calPre(){    
    Lu = unpad(L);
    Tx = (dcp(0, 1) * Kx2).replicate(1, Me).transpose();
    Ty = (dcp(0, 1) * Ky2).replicate(1, Ne);
    Tp = dcp(0, 1) * ArrayXXd::Ones(Me, Ne);
}

VectorXd
CQCGL2dReq::Fx(const VectorXd &x){
    assert(x.size() == 2*Ne*Me + 3);

    Vector3d th = x.tail<3>();	// wthx, wthy, wphi
    VectorXd ra = x.head(2*Me*Ne); // use matrix type for resizing
    ArrayXXcd a = r2c(ra);

    ArrayXXcd v = velocityReq(a, th(0), th(1), th(2));

    VectorXd F(2*Ne*Me+3);
    F << c2r(v), 0, 0, 0;
    
    return F;
}

VectorXd
CQCGL2dReq::DFx(const VectorXd &x, const VectorXd &dx){
    assert(x.size() == 2*Ne*Me + 3 && dx.size() == 2*Ne*Me + 3);
    
    Vector3d th = x.tail<3>();	// wthx, wthy, wphi
    Vector3d dth = dx.tail<3>();
    VectorXd ra = x.head(2*Me*Ne);
    VectorXd rda = dx.head(2*Me*Ne); 
    ArrayXXcd a = r2c(ra);
    ArrayXXcd da = r2c(rda); 
    
    ArrayXXcd t1 = tangent(a, 1);
    ArrayXXcd t2 = tangent(a, 2);
    ArrayXXcd t3 = tangent(a, 3);

    ArrayXXcd Ax = stabReq(a, da, th(0), th(1), th(2)) +
	dth(0) * t1 + dth(1) * t2 + dth(2) * t3;

    double t1x = (t1 * da.conjugate()).sum().real();
    double t2x = (t2 * da.conjugate()).sum().real();
    double t3x = (t3 * da.conjugate()).sum().real();

    VectorXd DF(2*Ne*Me+3);
    DF << c2r(Ax), t1x, t2x, t3x; 

    return DF;
}

std::tuple<ArrayXXcd, double, double, double, double>
CQCGL2dReq::findReq_hook(const ArrayXXcd &x0, const double wthx0, 
			 const double wthy0, const double wphi0){
    HOOK_PRINT_FREQUENCE = hookPrint;
    
    assert(x0.rows() == Me && x0.cols() == Ne);
    
    auto fx = std::bind(&CQCGL2dReq::Fx, this, ph::_1);
    auto dfx = std::bind(&CQCGL2dReq::DFx, this, ph::_1, ph::_2);
    
    VectorXd x(2*Me*Ne+3);
    x << c2r(x0), wthx0, wthy0, wphi0;

    VectorXd xnew;
    std::vector<double> errs;
    int flag;
    auto Pre = [this](const VectorXd x, const VectorXd dx){
	Vector3d th = x.tail<3>();
	Vector3d dth = dx.tail<3>();
	VectorXd dra = dx.head(2*Me*Ne);
	ArrayXXcd da = r2c(dra);

	ArrayXXcd px = 1/(Lu + Tx*th(0) + Ty*th(1) + Tp*th(2)) * da;
	VectorXd p(2*Ne*Me+3);
	p << c2r(px), dth;

	return p; 
    };
    std::tie(xnew, errs, flag) = Gmres0HookPre(fx, dfx, Pre, x, tol, minRD, maxit, maxInnIt, 
					       GmresRtol, GmresRestart, GmresMaxit,
					       false, 1); 
    if(flag != 0) fprintf(stderr, "Req not converged ! \n");
    
    Vector3d th = xnew.tail<3>();
    VectorXd y = xnew.head(2*Me*Ne); 
    
    return std::make_tuple(r2c(y), th(0), th(1), th(2), errs.back());
}


/** @brief find the optimal guess of wthx, wthy and wphi for a candidate req
 * 
 *  Let $V = [t1, t2, t3]$ the matrix of group tangent, and $V = QR$.
 *  Then expansion coefficient of vector $x$ is $a = Q^T x$.
 *  So The subtraction should be $Q a$, which is $V R^{-1} a$. Therefore, the
 *  optimal expansion coefficients in V basis is $R^{-1}a$.
 *  
 *  @return    [wthx, wthy, wphi] such that velocityReq(a0, wthx, wthy, wphi) minimal
 */
Vector3d
CQCGL2dReq::optReqTh(const ArrayXXcd &a0){
    ArrayXXcd t1 = tangent(a0, 1);
    ArrayXXcd t2 = tangent(a0, 2);
    ArrayXXcd t3 = tangent(a0, 3);
    ArrayXXcd x = velocity(a0);

    MatrixXd V(2*Me*Ne, 3);
    V << c2r(t1), c2r(t2), c2r(t3);
    MatrixXd Q, R;
    std::tie(Q, R) = QR(V);
    
    Vector3d a = Q.transpose() * c2r(x).matrix();
    Vector3d b =  R.inverse() * a;
    
    return b;
}

