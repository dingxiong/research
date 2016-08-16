#include <iostream>
#include <functional>
#include "CQCGL1dReq.hpp"
#include "iterMethod.hpp"
#include "myH5.hpp"

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

// A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
CQCGL1dReq::CQCGL1dReq(int N, double d,
		       double Mu, double Dr, double Di, double Br, double Bi, 
		       double Gr, double Gi, int dimTan):
    CQCGL1d(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}

// A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
CQCGL1dReq::CQCGL1dReq(int N, double d, 
		       double b, double c, double dr, double di, 
		       int dimTan):
    CQCGL1d(N, d, b, c, dr, di, dimTan){}
    
// iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
CQCGL1dReq::CQCGL1dReq(int N, double d,
		       double delta, double beta, double D, double epsilon,
		       double mu, double nu, int dimTan) :
    CQCGL1d(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

CQCGL1dReq::~CQCGL1dReq(){}

CQCGL1dReq & CQCGL1dReq::operator=(const CQCGL1dReq &x){
    return *this;
}


//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////


/**
 * @brief read req (relative equibrium) info from hdf5 file
 *
 */
std::tuple<VectorXd, double, double ,double>
CQCGL1dReq::readReq(const std::string fileName, const std::string groupName){
    H5File file(fileName, H5F_ACC_RDONLY);
    string DS = "/" + groupName + "/";

    return make_tuple(readMatrixXd(file, DS + "a").col(0),
		      readScalar<double>(file, DS + "wth"),
		      readScalar<double>(file, DS + "wphi"),
		      readScalar<double>(file, DS + "err")
		      );
}

std::tuple<VectorXd, double, double ,double>
CQCGL1dReq::readReq(const std::string fileName, 
		    const double Bi, const double Gi, int id){
    char g1[20], g2[20];
    sprintf(g1, "%013.6f", Bi);
    sprintf(g2, "%013.6f", Gi);
    
    string s1(g1);
    string s2(g2);
    string s = s1 + '/' + s2 + '/' + to_string(id);
    return readReq(fileName, s);
}

/**
 * @brief write [a, wth, wphi, err] of Req of cqcgl into a group
 * 
 * @note group should be a new group
 */
void 
CQCGL1dReq::writeReq(const std::string fileName, const std::string groupName,
		     const ArrayXd &a, const double wth, 
		     const double wphi, const double err){
    
    H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName);
    string DS = "/" + groupName + "/";
    
	
    writeMatrixXd(file, DS + "a", a);
    writeScalar<double>(file, DS + "wth", wth);
    writeScalar<double>(file, DS + "wphi", wphi);
    writeScalar<double>(file, DS + "err", err);
}

void 
CQCGL1dReq::writeReq(const std::string fileName, 
		     const double Bi, const double Gi, int id,
		     const ArrayXd &a, const double wth, const double wphi,
		     const double err){
    char g1[20], g2[20];
    sprintf(g1, "%013.6f", Bi);
    sprintf(g2, "%013.6f", Gi);
    
    string s1(g1);
    string s2(g2);
    string s = s1 + '/' + s2 + '/' + to_string(id);
    writeReq(fileName, s, a, wth, wphi, err);
}


//====================================================================================================

VectorXd
CQCGL1dReq::Fx(const VectorXd &x){
    assert(x.size() == 2*Ne + 2);

    Vector2d th = x.tail<2>();	// wth, wphi
    VectorXd a = x.head(2*Ne); // use matrix type for resizing

    ArrayXd v = velocityReq(a, th(0), th(1));

    VectorXd F(2*Ne+2);
    F << v, 0, 0;
    
    return F;
}

std::tuple<MatrixXd, MatrixXd, VectorXd>
CQCGL1dReq::calJJF(const VectorXd &x){
    ArrayXd a0 = x.head(2*Ne);
    double wth = x(2*Ne);
    double wphi = x(2*Ne+1);

    int n = a0.rows();
    assert(Ndim == n);
  
    MatrixXd DF(n, n+2); 
    ArrayXd tx_trans = transTangent(a0);
    ArrayXd tx_phase = phaseTangent(a0);
    DF.topLeftCorner(2*Ne, 2*Ne) = stabReq(a0, wth, wphi); 
    DF.col(2*Ne).head(2*Ne)= tx_trans;
    DF.col(2*Ne+1).head(2*Ne) = tx_phase;
    VectorXd F(n);
    F.head(n) = velocity(a0) + wth*tx_trans + wphi*tx_phase;


    MatrixXd JJ = DF.transpose() * DF;
    MatrixXd D  = JJ.diagonal().asDiagonal();
    VectorXd df = DF.transpose() * F;

    return std::make_tuple(JJ, D, df);
}


std::tuple<VectorXd, double, double, double, int>
CQCGL1dReq::findReq_LM(const ArrayXd &a0, const double wth0, const double wphi0, 
		       const double tol,
		       const int maxit,
		       const int innerMaxit){
    
    VectorXd x(2*Ne+2);
    x << a0, wth0, wphi0;
    
    auto fx = std::bind(&CQCGL1dReq::Fx, this, ph::_1);    
    CQCGL1dReqJJF<MatrixXd> jj(this);
    PartialPivLU<MatrixXd> solver;
    
    VectorXd xe;
    std::vector<double> res;
    int flag;
    std::tie(xe, res, flag) = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    if(flag != 0) fprintf(stderr, "Req not converged ! \n");
    
    VectorXd a = xe.head(2*Ne);
    double wth = xe(2*Ne);
    double wphi = xe(2*Ne+1);
    return std::make_tuple( a, wth, wphi, res.back(), flag );
}


/** @brief find the optimal guess of wth and wphi for a candidate req
 * 
 *  When we find a the inital state of a candidate of req, we also need
 *  to know the appropriate th and phi to start the Newton search.
 *  According to the definition of velocityReq(), this is a residual
 *  minimization problem.
 *
 *  @return    [wth, wphi, err] such that velocityReq(a0, wth, wphi) minimal
 */
std::vector<double>
CQCGL1dReq::optThPhi(const ArrayXd &a0){ 
    VectorXd t1 = transTangent(a0);
    VectorXd t2 = phaseTangent(a0);
    double c = t2.dot(t1) / t1.dot(t1);
    VectorXd t3 = t2 - c * t1;

    VectorXd v = velocity(a0);
    double a1 = t1.dot(v) / t1.dot(t1);
    double a2 = t3.dot(v) / t3.dot(t3);
    
    double err = (v - a1 * t1 - a2 * t3).norm();
    
    std::vector<double> x = {-(a1-a2*c), -a2, err};
    return x;
}

/**
 * @brief find req with a sequence of Bi or Gi
 */ 
void 
CQCGL1dReq::findReqParaSeq(const std::string file, int id, double step, int Ns, bool isBi){
    double Bi0 = Bi;
    double Gi0 = Gi;
    
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = CQCGL1dReq::readReq(file, Bi, Gi, id);
    
    ArrayXd a;
    double wth, wphi, err;
    int flag;
    
    for (int i = 0; i < Ns; i++){
	if (isBi) Bi += step;
	else Gi += step;

	std::tie(a, wth, wphi, err, flag) = findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
	if (flag == 0){
	    writeReq(file, Bi, Gi, id, a, wth, wphi, err);
	    a0 = a;
	    wth0 = wth;
	    wphi0 = wphi;
	}
	else exit(1);
    }
    
    Bi = Bi0; 			// restore Bi, Gi
    Gi = Gi0;
}

#if 0
//====================================================================================================
VectorXd
CQCGL1dReq::DFx(const VectorXd &x, const VectorXd &dx){
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
CQCGL1dReq::findReq_hook(const ArrayXXcd &x0, const double wthx0, 
			 const double wthy0, const double wphi0){
    HOOK_PRINT_FREQUENCE = hookPrint;
    
    assert(x0.rows() == Me && x0.cols() == Ne);
    
    auto fx = std::bind(&CQCGL1dReq::Fx, this, ph::_1);
    auto dfx = std::bind(&CQCGL1dReq::DFx, this, ph::_1, ph::_2);
    
    VectorXd x(2*Me*Ne+3);
    x << c2r(x0), wthx0, wthy0, wphi0;

    VectorXd xnew;
    std::vector<double> errs;
    int flag;
    auto Pre = [this](const VectorXd &x, const VectorXd &dx){
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
#endif
