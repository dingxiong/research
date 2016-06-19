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
    invL = unpad(L); 
    invL = 1 / invL;

    invL.resize(Me*Ne, 1);
    GmresPre.resize(2*Me*Ne+3);
    GmresPre << c2r(invL), 1, 1, 1; 
    invL.resize(Me, Ne);
}

VectorXd
CQCGL2dReq::Fx(const VectorXd &x){
    assert(x.size() == 2*Ne*Me + 3);

    Vector3d th = x.tail<3>();	// wthx, wthy, wphi
    MatrixXd ra = x.head(2*Me*Ne); // use matrix type for resizing
    ra.resize(2*Me, Ne);
    ArrayXXcd  a = r2c(ra);

    ArrayXXcd v = velocityReq(a, th(0), th(1), th(2));
    v.resize(Me*Ne, 1);

    VectorXd F(2*Ne*Me+3);
    F << c2r(v), 0, 0, 0;
    
    return F;
}

VectorXd
CQCGL2dReq::DFx(const VectorXd &x, const VectorXd &dx){
    assert(x.size() == 2*Ne*Me + 3 && dx.size() == 2*Ne*Me + 3);
    
    Vector3d th = x.tail<3>();	// wthx, wthy, wphi
    Vector3d dth = dx.tail<3>();
    MatrixXd ra = x.head(2*Me*Ne);
    MatrixXd rda = dx.head(2*Me*Ne); 
    ra.resize(2*Me, Ne);
    rda.resize(2*Me, Ne);
    ArrayXXcd a = r2c(ra);
    ArrayXXcd da = r2c(rda); 
    
    ArrayXXcd t1 = stabReq(a, da, th(0), th(1), th(2));
    ArrayXXcd t2 = dth(0) * tangent(a, 1);
    ArrayXXcd Ax = stabReq(a, da, th(0), th(1), th(2)) +
	dth(0) * tangent(a, 1) + 
	dth(1) * tangent(a, 2) +
	dth(2) * tangent(a, 3)
	;
    Ax.resize(Me*Ne, 1);

    VectorXd DF(2*Ne*Me+3);
    DF << c2r(Ax), 0, 0, 0; 

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
    ArrayXXcd xp(x0);
    xp.resize(Me*Ne, 1);
    x << c2r(xp), wthx0, wthy0, wphi0;

    VectorXd xnew;
    std::vector<double> errs;
    int flag;
    auto Pre = [this](const VectorXd x){ VectorXd px = GmresPre * x.array(); return px; };
    std::tie(xnew, errs, flag) = Gmres0HookPre(fx, dfx, Pre, x, tol, minRD, maxit, maxInnIt, 
					       GmresRtol, GmresRestart, GmresMaxit,
					       false, 3);
    if(flag != 0) fprintf(stderr, "Req not converged ! \n");
    
    Vector3d th = xnew.tail<3>();
    VectorXd y = xnew.head(2*Me*Ne);
    y.resize(2*Me, Ne);
    
    return std::make_tuple(r2c(y), th(0), th(1), th(2), errs.back());
}
