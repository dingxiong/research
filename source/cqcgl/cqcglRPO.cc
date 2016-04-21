#include "cqcglRPO.hpp"
#include <functional>   // std::bind
namespace ph = std::placeholders;

using namespace denseRoutines;
using namespace sparseRoutines;
using namespace iterMethod;
using namespace Eigen;
using namespace std;

//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

CqcglRPO::CqcglRPO(int nstp, int M,
		   int N, double d, double h,
		   double b, double c,
		   double dr, double di,
		   int threadNum)
    : N(N),
      nstp(nstp),
      M(M),
      cgl1(N, d, h, false, 0, b, c, dr, di, threadNum),
      cgl1(N, d, h, true, 1, b, c, dr, di, threadNum),
      cgl1(N, d, h, true, 0, b, c, dr, di, threadNum),
{
    Ndim = cgl1.Ndim;
}


CqcglRPO::~CqcglRPO(){}

CqcglRPO & CqcglRPO::operator=(const CqcglRPO &x){
    return *this;
}

//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////

/**
 * @brief         form g*f(x,t) - x
 * @param[in] x   [Ndim + 3, 1] dimensional vector: (x, t, theta, phi)
 * @return        vector F(x, t) =
 *                  | g*f(x, t) - x|
 *                  |       0      |
 *                  |       0      |
 *                  |       0      |
 */
VectorXd CqcglRPO::Fx(const VectorXd & x){
    Vector3d t = x.tail<3>();
    assert(t(0) > 0); 		/* make sure T > 0 */
    cgl1.changeh(t(0)/nstp);
    VectorXd fx = cgl1.intg(x.head(Ndim), nstp, nstp).rightCols<1>();
    VectorXd F(Ndim + 3);
    F << cgl1.Rotate(fx, t(1), t(2)), t;
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
    Vector3d dt = dx.tail<3>();
    assert(t(0) > 0); 		/* make sure T > 0 */
    cgl2.changeh(t(0)/nstp);
    ArrayXXd tmp = cgl2.intgv(x.head(Ndim), dx.head(Ndim), nstp); /* f(x, t) and J(x, t)*dx */
    ArrayXd gfx = cgl2.Rotate(tmp.col(0), t(1), t(2)); /* g(theta, phi)*f(x, t) */
    ArrayXd gJx = cgl2.Rotate(tmp.col(1), t(1), t(2)); /* g(theta, phi)*J(x,t)*dx */
    ArrayXd v1 = cgl2.velocity(x.head(Ndim));	       /* v(x) */
    ArrayXd v2 = cgl2.velocity(tmp.col(0)); /* v(f(x, t)) */
    ArrayXd t1 = cgl2.transTangent(x.head(Ndim));
    ArrayXd t2 = cgl2.phaseTangent(x.head(Ndim));
    VectorXd DF(Ndim + 3);
    DF << gJx.matrix() - dx.head(Ndim)
	+ cgl2.Rotate(v2, t(1), t(2)).matrix() * dt(0)
	+ cgl2.transTangent(gfx).matrix() * dt(1)
	+ cgl2.phaseTangent(gfx).matrix() * dt(2),
	
	alpha1 * v1.matrix().dot(dx.head(Ndim)), /* strength scale */
	alpha2 * t1.matrix().dot(dx.head(Ndim)),
	alpha3 * t2.matrix().dot(dx.head(Ndim))
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
VectorXd CqcglRPO::MFx(const VectorXd &x){
    Vector3d t = x.tail<3>();	   /* T, theta, phi */
    assert(t(0) > 0);		   /* make sure T > 0 */
    cgl1.changeh(t(0) / nstp / M); /* period T = h*nstp*M */
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

/* 
 * @brief get the multishooting product J * dx. Dimension [nstp*M+3, 1]
 * @see MFx()
 */
VectorXd CqcglRPO::MDFx(const VectorXd &x, const VectorXd &dx){
    Vector3d t = x.tail<3>();
    Vector3d dt = dx.tail<3>();
    assert(t(0) > 0); 		/* make sure T > 0 */
    cgl2.changeh(t(0) / nstp / M);
    VectorXd DF(VectorXd::Zero(Ndim * M + 3));
    
    for(size_t i = 0; i < M; i++){
	ArrayXd xt = x.segment(i*Ndim, Ndim);
	ArrayXd dxt = dx.segment(i*Ndim, Ndim);
	ArrayXXd tmp = cgl2.intgv(xt, dxt, nstp); /* f(x, t) and J(x, t)*dx */
	
	VectorXd v1 = cgl2.velocity(xt); /* v(x) */
	VectorXd v2 = cgl2.velocity(tmp.col(0));   /* v(f(x, t)) */
	VectorXd t1 = cgl2.transTangent(xt);
	VectorXd t2 = cgl2.phaseTangent(xt);

	// update the last 3 elements
	DF(M * Ndim) += v1.dot(dxt.matrix());
	DF(M * Ndim + 1) += t1.dot(dxt.matrix());
	DF(M * Ndim + 2) += t2.dot(dxt.matrix());
	
	if(i != M-1){
	    DF.segment(i*Ndim, Ndim) = tmp.col(1).matrix()
		- dx.segment((i+1)*Ndim , Ndim)
		+ 1.0 / M * v2 * dt(0);
	}
	else{
	    ArrayXd gfx = cgl2.Rotate(tmp.col(0), t(1), t(2)); /* g(theta, phi)*f(x, t) */
	    VectorXd gJx = cgl2.Rotate(tmp.col(1), t(1), t(2)); /* g(theta, phi)*J(x,t)*dx */
	    DF.segment(i*Ndim, Ndim) = gJx
		- dx.segment(0, Ndim)
		+ 1.0/M * cgl2.Rotate(v2, t(1), t(2)).matrix() * dt(0)
		+ cgl2.transTangent(gfx).matrix() * dt(1)
		+ cgl2.phaseTangent(gfx).matrix() * dt(2)
		;
	}
    }

    // scale the strength of constraints
    DF(M * Ndim) *= alpha1;
    DF(M * Ndim + 1) *= alpha2;
    DF(M * Ndim + 2) *= alpha3;
    
    return DF;
    
}


/**
 * @brief find rpo in cqcgl 1d system
 *
 * @return [x, T, theta, phi, err]
 */
std::tuple<VectorXd, double, double, double, double>
CqcglRPO::findRPO(const VectorXd &x0, const double T,
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
    assert(x0.size() == Ndim);
    auto fx = std::bind(&CqcglRPO::Fx, this, ph::_1);
    auto dfx = std::bind(&CqcglRPO::DFx, this, ph::_1, ph::_2);
    VectorXd x(Ndim+3);
    x << x0, T, th0, phi0;
    auto result = InexactNewtonBacktrack(fx, dfx, x, tol, btMaxIt, maxit, eta0,
					 t, theta_min, theta_max, GmresRestart, GmresMaxit);
    if(std::get<2>(result) != 0){
	fprintf(stderr, "RPO not converged ! \n");
    }
    return std::make_tuple(std::get<0>(result).head(Ndim), /* x */
			   std::get<0>(result)(Ndim),	   /* T */
			   std::get<0>(result)(Ndim+1),	   /* theta */
			   std::get<0>(result)(Ndim+2),	   /* phi */
			   std::get<1>(result).back()	   /* err */
			   );
}

/**
 * @brief find rpo in cqcgl 1d system
 *
 * @return [x, T, theta, phi, err]
 */
std::tuple<MatrixXd, double, double, double, double>
CqcglRPO::findRPOM(const MatrixXd &x0, const double T,
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
    assert(x0.cols() == M && x0.rows() == Ndim);
    auto fx = std::bind(&CqcglRPO::MFx, this, ph::_1);
    auto dfx = std::bind(&CqcglRPO::MDFx, this, ph::_1, ph::_2);
    
    // initialize input 
    VectorXd x(M * Ndim + 3);
    MatrixXd tmp(x0);
    tmp.resize(M * Ndim, 1);
    x << tmp, T, th0, phi0;
    
    auto result = InexactNewtonBacktrack(fx, dfx, x, tol, btMaxIt, maxit, eta0,
					 t, theta_min, theta_max, GmresRestart, GmresMaxit);
    if(std::get<2>(result) != 0){
	fprintf(stderr, "RPO not converged ! \n");
    }

    MatrixXd tmp2(std::get<0>(result).head(M*Ndim));
    tmp2.resize(Ndim, M);
    return std::make_tuple(tmp2, /* x */
			   std::get<0>(result)(M*Ndim),	  /* T */
			   std::get<0>(result)(M*Ndim+1), /* theta */
			   std::get<0>(result)(M*Ndim+2), /* phi */
			   std::get<1>(result).back()	  /* err */
			   );
}

/**
 * @brief single shooting to find rpo with GMRES HOOK algorithm 
 *
 * @param[in]  x0            initial guess state
 * @param[in]  T             initial guess period
 * @param[in]  th0           initial guess of translation angle
 * @param[in]  phi0          initial guess of phase angle
 * @param[in]  tol           tolerance of the orbit ||x(0)-X(T)||
 * @param[in]  minRD         mininal relative descrease at each step
 * @param[in]  maxit         maximal number of iterations for Newton steps
 * @param[in]  maxInnIt      maximal number of iterations for Hook steps
 * @param[in]  GmresRtol     relative tolerence of GMRES
 * @param[in]  GmresRestart  maximal Krylov subspace dimension => inner loop size
 * @param[in]  GmresMaxit    maximal outer iteration number
 * @return     [x, T, theta, phi, err]
 *
 * @see  findRPOM_hook() for multishooting method
 */
std::tuple<VectorXd, double, double, double, double>
CqcglRPO::findRPO_hook(const VectorXd &x0, const double T,
		       const double th0, const double phi0,
		       const double tol,
		       const double minRD,
		       const int maxit,
		       const int maxInnIt,
		       const double GmresRtol,
		       const int GmresRestart,
		       const int GmresMaxit){
    assert(x0.size() == Ndim);
    auto fx = std::bind(&CqcglRPO::Fx, this, ph::_1);
    auto dfx = std::bind(&CqcglRPO::DFx, this, ph::_1, ph::_2);
    VectorXd x(Ndim + 3);
    x << x0, T, th0, phi0;
    
    auto result = Gmres0Hook(fx, dfx, x, tol, minRD, maxit, maxInnIt,
			     GmresRtol, GmresRestart, GmresMaxit,
			     true, 3);
    if(std::get<2>(result) != 0){
	fprintf(stderr, "RPO not converged ! \n");
    }

    return std::make_tuple(std::get<0>(result).head(Ndim), /* x */
			   std::get<0>(result)(Ndim),	   /* T */
			   std::get<0>(result)(Ndim+1),	   /* theta */
			   std::get<0>(result)(Ndim+2),	   /* phi */
			   std::get<1>(result).back()	   /* err */
			   );
}



/**
 * @brief find rpo in cqcgl 1d system
 *
 * @return [x, T, theta, phi, err]
 * @see findRPO_hook() for single shooting method
 */
std::tuple<MatrixXd, double, double, double, double>
CqcglRPO::findRPOM_hook(const MatrixXd &x0, const double T,
			const double th0, const double phi0,
			const double tol,
			const double minRD,
			const int maxit,
			const int maxInnIt,
			const double GmresRtol,
			const int GmresRestart,
			const int GmresMaxit){
    assert(x0.cols() == M && x0.rows() == Ndim);
    auto fx = std::bind(&CqcglRPO::MFx, this, ph::_1);
    auto dfx = std::bind(&CqcglRPO::MDFx, this, ph::_1, ph::_2);
    
    // initialize input 
    VectorXd x(M * Ndim + 3);
    MatrixXd tmp(x0);
    tmp.resize(M * Ndim, 1);
    x << tmp, T, th0, phi0;
    
    auto result = Gmres0Hook(fx, dfx, x, tol, minRD, maxit, maxInnIt,
			     GmresRtol, GmresRestart, GmresMaxit,
			     true, 3);
    if(std::get<2>(result) != 0){
	fprintf(stderr, "RPO not converged ! \n");
    }

    MatrixXd tmp2(std::get<0>(result).head(M*Ndim));
    tmp2.resize(Ndim, M);
    return std::make_tuple(tmp2, /* x */
			   std::get<0>(result)(M*Ndim),	  /* T */
			   std::get<0>(result)(M*Ndim+1), /* theta */
			   std::get<0>(result)(M*Ndim+2), /* phi */
			   std::get<1>(result).back()	  /* err */
			   );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// new version of F and DF, This version use a full multishooting method

VectorXd CqcglRPO::MFx2(const VectorXd &x){
    int N = Ndim + 3;
    VectorXd F(VectorXd::Zero(N*M));
    
    for(int i = 0; i < M; i++){
	VectorXd xi = x.segment(N*i, N);
	int j = (i+1) % M;
	VectorXd xn = x.segment(N*j, N);
	
	double t = xi(Ndim);
	double th = xi(Ndim+1);
	double phi = xi(Ndim+2);	
	assert(t > 0);
	
	cgl1.changeh( t / nstp );
	VectorXd fx = cgl1.intg(xi.head(Ndim), nstp, nstp).rightCols<1>();
	
	F.segment(i*N, Ndim) = cgl1.Rotate(fx, th, phi).matrix() - xn.head(Ndim);
    }

    return F;
}

VectorXd CqcglRPO::MDFx2(const VectorXd &x, const VectorXd &dx){
    int N = Ndim + 3;
    VectorXd DF(VectorXd::Zero(N*M));

    for (int i = 0; i < M; i++) {
	VectorXd xi = x.segment(i*N, N);
	VectorXd dxi = dx.segment(i*N, N);
	int j = (i+1) % M;
	VectorXd dxn = dx.segment(j*N, N);
	
	double t = xi(Ndim);
	double th = xi(Ndim+1);
	double phi = xi(Ndim+2);
	assert( t > 0 );
	double dt = dxi(Ndim);
	double dth = dxi(Ndim+1);
	double dphi = dxi(Ndim+2);
	
	cgl2.changeh(t / nstp);
	MatrixXd tmp = cgl2.intgv(xi.head(Ndim), dxi.head(Ndim), nstp);

	VectorXd gfx = cgl2.Rotate(tmp.col(0), th, phi);
	VectorXd gJx = cgl2.Rotate(tmp.col(1), th, phi);
	
	VectorXd v = cgl2.velocity(xi.head(Ndim));
	VectorXd vgf = cgl2.velocity(gfx); 

	VectorXd t1 = cgl2.transTangent(xi.head(Ndim));
	VectorXd tgf1 = cgl2.transTangent(gfx);

	VectorXd t2 = cgl2.phaseTangent(xi.head(Ndim));
	VectorXd tgf2 = cgl2.phaseTangent(gfx);
	
	DF.segment(i*N, Ndim) = gJx + vgf*dt + tgf1*dth + tgf2*dphi - dxn.head(Ndim);
	DF(i*N+Ndim) = v.dot(dxi.head(Ndim));
	DF(i*N+Ndim+1) = t1.dot(dxi.head(Ndim));
	DF(i*N+Ndim+2) = t2.dot(dxi.head(Ndim));
	
    }
    
    return DF;

}

std::tuple<MatrixXd, double>

CqcglRPO::findRPOM_hook2(const MatrixXd &x0, 
			 const double tol,
			 const double minRD,
			 const int maxit,
			 const int maxInnIt,
			 const double GmresRtol,
			 const int GmresRestart,
			 const int GmresMaxit){
    int N = Ndim + 3;
    assert(x0.cols() == M && x0.rows() == N);
    auto fx = std::bind(&CqcglRPO::MFx2, this, ph::_1);
    auto dfx = std::bind(&CqcglRPO::MDFx2, this, ph::_1, ph::_2);
    
    // initialize input 
    MatrixXd x(x0);
    x.resize(M * N, 1);
    
    auto result = Gmres0Hook(fx, dfx, x, tol, minRD, maxit, maxInnIt,
			     GmresRtol, GmresRestart, GmresMaxit,
			     true, 3);
    if(std::get<2>(result) != 0){
	fprintf(stderr, "RPO not converged ! \n");
    }

    MatrixXd tmp2(std::get<0>(result));
    tmp2.resize(N, M);
    return std::make_tuple(tmp2, /* x, th, phi */
			   std::get<1>(result).back()	  /* err */
			   );
}

std::tuple<SpMat, SpMat, VectorXd> 
CqcglRPO::calJJF(const VectorXd &x){
    int N = Ndim + 3;
    SpMat JJ(N*M, N*M);
    SpMat D(N*M, N*M);
    VectorXd df(VectorXd::Zero(N*M));
    
    MatrixXd F(MatrixXd::Zero(N, N));
    MatrixXd FF(N, N);
    MatrixXd FK(MatrixXd::Zero(N, N));
    MatrixXd KF(MatrixXd::Zero(N, N));

    std::vector<Tri> nzjj, nzd;
    nzjj.reserve(3*M*N*N);
    nzd.reserve(M*N);
    
    /////////////////////////////////////////
    // construct the JJ, Diag(JJ), JF
    for(int i = 0; i < M; i++){
	fprintf(stderr, "%d ", i);
	
	VectorXd xi = x.segment(i*N, N);
	int j = (i+1) % M;
	VectorXd xn = x.segment(j*N, N);
	
	double t = xi(Ndim);
	double th = xi(Ndim + 1);
	double phi = xi(Ndim + 2);
	assert( t > 0);

	cgl3.changeh( t/nstp ); 
	auto tmp = cgl3.intgj(xi.head(Ndim), nstp, nstp, nstp);
	ArrayXd fx = tmp.first.col(1);
	ArrayXXd &J = tmp.second;

	VectorXd gfx = cgl3.Rotate(fx, th, phi);
	
	df.segment(i*N, Ndim) = gfx - xn.head(Ndim);
	
	F.topLeftCorner(Ndim, Ndim) = cgl3.Rotate(J, th, phi);

	F.col(Ndim).head(Ndim) = cgl3.velocity(gfx);
	F.col(Ndim+1).head(Ndim) = cgl3.transTangent(gfx);
	F.col(Ndim+2).head(Ndim) = cgl3.phaseTangent(gfx); 
	
	F.row(Ndim).head(Ndim) = cgl3.velocity(xi.head(Ndim));
	F.row(Ndim+1).head(Ndim) = cgl3.transTangent(xi.head(Ndim)).transpose();
	F.row(Ndim+2).head(Ndim) = cgl3.phaseTangent(xi.head(Ndim)).transpose();
	
	FF = F.transpose() * F; 
	for(int i = 0; i < Ndim; i++) FF(i, i) += 1;
	
	FK.leftCols(Ndim) = -F.transpose().leftCols(Ndim);
	KF.topRows(Ndim) = -F.topRows(Ndim);
	
	std::vector<Tri> triFF = triMat(FF, i*N, i*N);
	std::vector<Tri> triFK = triMat(FK, i*N, j*N);
	std::vector<Tri> triKF = triMat(KF, j*N, i*N);
	std::vector<Tri> triD = triDiag(FF.diagonal(), i*N, i*N);
	    
	nzjj.insert(nzjj.end(), triFF.begin(), triFF.end());
	nzjj.insert(nzjj.end(), triFK.begin(), triFK.end());
	nzjj.insert(nzjj.end(), triKF.begin(), triKF.end());

	nzd.insert(nzd.end(), triD.begin(), triD.end());
    }

    JJ.setFromTriplets(nzjj.begin(), nzjj.end());
    D.setFromTriplets(nzd.begin(), nzd.end());

    return std::make_tuple(JJ, D, df);
    
}

std::tuple<MatrixXd, double>
CqcglRPO::findRPOM_LM(const MatrixXd &x0, 
		      const double tol,
		      const int maxit,
		      const int innerMaxit){
    int N = Ndim + 3;
    assert(x0.cols() == M && x0.rows() == N);
    auto fx = std::bind(&CqcglRPO::MFx2, this, ph::_1);
    
    MatrixXd x(x0);
    x.resize(M * N, 1); 
    
    // SparseLU<SpMat> solver;
    SimplicialLDLT<SpMat> solver; 
    cqcglJJF<SpMat> jj(*this);
    auto result = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    
    if(std::get<2>(result) != 0) fprintf(stderr, "RPO not converged ! \n");
    
    MatrixXd tmp2(std::get<0>(result));
    tmp2.resize(N, M);
    return std::make_pair( tmp2, std::get<1>(result).back() );
}


