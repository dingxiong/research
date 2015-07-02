#include "findRPO.hpp"
#include <functional>   // std::bind
namespace ph = std::placeholders;

//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

CqcglRPO::CqcglRPO(int nstp, int N, double d, double h, 
		   double Mu, double Br, double Bi,
		   double Dr, double Di, double Gr,
		   double Gi,  int threadNum)
    : nstp(nstp),
      cgl(N, d, h, false, 1, Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum),
      cgl2(N, d, h, true, 1, Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum);
{
    Ndim = cgl.Ndim;
}

Cqcgl1d::~Cqcgl1d(){}

Cqcgl1d & Cqcgl1d::operator=(const Cqcgl1d &x){
    return *this;
}

//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////

/**
 * @brief form g*f(x,t) - x
 * @param[in] x   [Ndim + 1, 1] dimensional vector: (x, t, theta, phi)
 * @return    vector F(x, t) =
 *               | g*f(x, t) - x|
 *               |       0      |
 *               |       0      |
 *               |       0      |
 */
VectorXd CqcglRPO::Fx(const VectorXd & x){
    Vector3d t = x.tail<3>();
    cgl.changeh(t(0)/nstp);
    VectorXXd fx = cgl.intg(x.head(Ndim), nstp, nstp);
    VectorXd F(Ndim + 3);
    F << cgl.Rotate(fx.rightCols<1>(), t(1), t(2)), t;
    return F - x;
}

/**
 * @brief get the product J * dx
 *
 * Here J = | g*J(x, t) - I,  g*v(f(x,t)),  g*t1(f(x,t)),  g*t2(f(x,t))| 
 *          |     v(x),          0             0                  0    |
 *          |     t1(x),         0             0                  0    |
 *          |     t2(x),         0             0                  0    |
 */
VectorXd CqcglRPO::DFx(const VectorXd &x, const VectorXd &dx){
    Vector3d t = x.tail<3>();
    cgl.changeh(t(0)/nstp);
    ArrayXXd tmp = cgl.intgv(x.head(Ndim), dx.head(Ndim), nstp); /* f(x, t) and J(x, t)*dx */
    ArrayXd gfx = cgl.Rotate(tmp.col(0), t(1), t(2)); /* g(theta, phi)*f(x, t) */
    ArrayXd gJx = cgl.Rotate(tmp.col(1), t(1), t(2)); /* g(theta, phi)*J(x,t)*dx */
    ArrayXd v1 = cgl.velocity(x.head(Ndim));	      /* v(x) */
    ArrayXd v2 = cgl.velocity(tmp.col(0));	      /* v(f(x, t)) */
    ArrayXd t1 = cgl.transTangent(x.head(Ndim));
    ArrayXd t2 = cgl.phaseTangent(x.head(Ndim));
    VectorXd DF(Ndim + 3);
    DF << gJx.matrix() - dx.head(Ndim)
	+ cgl.Rotate(v2, t(1), t(2)).matrix() * t(0)
	+ cgl.transTangent(gfx) * t(1)
	+ cgl.phaseTangent(gfx) * t(2),
	
	v1.matrix().dot(dx.head(Ndim)),
	t1.matrix().dot(dx.head(Ndim)),
	t2.matrix().dot(dx.head(Ndim)),
	;

    return DF;
}

/**
 * @brief find rpo in cqcgl 1d system
 *
 * @return [x, T, theta, phi, err]
 */
std::tuple<VectorXd, double, double, double, double>
CqcglRPO::findRPO(const VectorXd &x0, const double T, const int nstp,
		  const double th0, const double phi0,
		  const double tol,
		  const int btMaxIt,
		  const int maxit,
		  const double eta0,
		  const double t,
		  const double theta_min,
		  const double theta_max,
		  const int GmresRestart,
		  const int GmresMaxit){
    auto fx = std::bind(CqcglRPO::Fx, this, ph::_1);
    auto dfx = std::bind(CqcglRPO::DFx, this, ph::_1, ph::_2);
    VectorXd x(Ndim+3);
    x << x0, T, th0, phi0;
    auto result = InexactNewtonBacktrack(fx, dfx, x, tol, btMaxIt, maxit, eta0,
					 t, theta_min, theta_max, GmresRestart, GmresMaxit);
    if(std::get<2>(result != 0)){
	fprintf(stderr, "RPO not converged ! \n");
    }
    return std::make_tuple(std::get<0>(result).head(Ndim), /* x */
			   std::get<0>(result)(Ndim),	   /* T */
			   std::get<0>(result)(Ndim+1),	   /* theta */
			   std::get<0>(result)(Ndim+2),	   /* phi */
			   std::get<1>(result).back()	   /* err */
			   );
}
