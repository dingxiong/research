#include "cqcglRPO.hpp"
#include <functional>   // std::bind
namespace ph = std::placeholders;

using std::cout;
using std::endl;
using namespace sparseRoutines;
using namespace iterMethod;
using namespace Eigen;


//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

CqcglRPO::CqcglRPO(int nstp, int M,
		   int N, double d, double h, 
		   double Mu, double Br, double Bi,
		   double Dr, double Di, double Gr,
		   double Gi,  int threadNum)
    : N(N),
      nstp(nstp),
      M(M),
      cgl1(N, d, h, false, 1, Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum),
      cgl2(N, d, h, true, 1, Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum)
{
    Ndim = cgl1.Ndim;
}

CqcglRPO::CqcglRPO(int nstp, int M,
		   int N, double d, double h,
		   double b, double c,
		   double dr, double di,
		   int threadNum)
    : CqcglRPO(nstp, M, N, d, h, -1, 1, c, 1, b, -dr, -di, threadNum)
{ 
    // delegating constructor forbids other initialization in the list
    cgl1.b = b;
    cgl1.c = c;
    cgl1.dr = dr;
    cgl1.di = di;

    cgl2.b = b;
    cgl2.c = c;
    cgl2.dr = dr;
    cgl2.di = di;
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

#ifdef RPO_OMP
    omp_set_num_threads(RPO_OMP);
#pragma omp parallel for shared (x, t, F)
#endif
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
    
#ifdef RPO_OMP
    omp_set_num_threads(RPO_OMP);
#pragma omp parallel for shared (x, dx, t, dt, DF)
#endif
    for(size_t i = 0; i < M; i++){
	ArrayXd xt = x.segment(i*Ndim, Ndim);
	ArrayXd dxt = dx.segment(i*Ndim, Ndim);
	ArrayXXd tmp = cgl2.intgv(xt, dxt, nstp); /* f(x, t) and J(x, t)*dx */
	
	VectorXd v1 = cgl2.velocity(xt); /* v(x) */
	VectorXd v2 = cgl2.velocity(tmp.col(0));   /* v(f(x, t)) */
	VectorXd t1 = cgl2.transTangent(xt);
	VectorXd t2 = cgl2.phaseTangent(xt);

#ifdef RPO_OMP
#pragma omp critical
#endif
	{
	    // update the last 3 elements
	    DF(M * Ndim) += v1.dot(dxt.matrix());
	    DF(M * Ndim + 1) += 0.01 * t1.dot(dxt.matrix());
	    DF(M * Ndim + 2) += 0.01 * t2.dot(dxt.matrix());
	}
	
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
    
    auto result = Gmres0Hook(fx, dfx, x, tol, maxit, maxInnIt,
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
    
    auto result = Gmres0Hook(fx, dfx, x, tol, maxit, maxInnIt,
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



//////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#if 0				
//////////////////////////////////////////////////////////////////////
//                       Levenberg-Marquardt                        //
//                       (abanded approach)                         //
//////////////////////////////////////////////////////////////////////


/*********************************************************************** 
 *                      member functions                               *
 ***********************************************************************/

/**
 * @brief solve the Ax = b problem
 */
inline VectorXd CqcglRPO::cgSolver(ConjugateGradient<SpMat> &CG, SparseLU<SpMat> &solver,
				   SpMat &H, VectorXd &JF, bool doesUseMyCG /* = true */,
				   bool doesPrint /* = false */){
    VectorXd dF;
    if(doesUseMyCG){
	std::pair<VectorXd, std::vector<double>> cg = iterMethod::ConjGradSSOR<SpMat>
	    (H, -JF, solver, VectorXd::Zero(H.rows()), H.rows(), 1e-6);
	dF = cg.first;
	if(doesPrint){
	    printf("CG error %g, iteration number %zd\n", cg.second.back(), cg.second.size());
	}
    } else {
	CG.compute(H);
	dF = CG.solve(-JF);
	if(doesPrint){
	    printf("CG error %f, iteration number %d\n", CG.error(), CG.iterations());
	}
    }
    return dF;
}


/**
 * @brief find rpo in cqcgl 1d system
 */
std::tuple<ArrayXXd, double, double, double, double>
CqcglRPO::findPO(const ArrayXXd &aa0, const double h0, const int nstp,
		 const double th0, const double phi0,
		 const int MaxN, const double tol,
		 const bool doesUseMyCG /* = true */,
		 const bool doesPrint /* = false */){

    assert(Ndim == aa0.rows()); 
    const int M = aa0.cols();
    ArrayXXd x(aa0);
    
    double h = h0;
    double th = th0;
    double phi = phi0;
    double lam = 1; 

    ConjugateGradient<SpMat> CG;
    SparseLU<SpMat> solver; // used in the pre-CG method
    
    for(size_t i = 0; i < MaxN; i++){
	if (lam > 1e10) break;
	if(doesPrint){
	    printf("********  i = %zd/%d   ******** \n", i, MaxN);
	}
	Cqcgl1d cgl(N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi);
	VectorXd F = cgl.multiF(x, nstp, th, phi);
	double err = F.norm(); 
	if(err < tol){
	    if(doesPrint) printf("stop at norm(F)=%g\n", err);
	    break;
	}
   
	std::pair<SpMat, VectorXd> p = cgl.multishoot(x, nstp, th, phi, doesPrint); 
	SpMat JJ = p.first.transpose() * p.first;
	VectorXd JF = p.first.transpose() * p.second;
	SpMat Dia = JJ;
	Dia.prune(KeepDiag());
	
	for(size_t j = 0; j < 20; j++){
	    SpMat H = JJ + lam * Dia;
	    VectorXd dF = cgSolver(CG, solver, H, JF, doesUseMyCG, doesPrint);
	    ArrayXXd xnew = x + Map<ArrayXXd>(&dF(0), Ndim, M);
	    double hnew = h + dF(Ndim*M)/nstp; // be careful here.
	    double thnew = th + dF(Ndim*M+1);
	    double phinew = phi + dF(Ndim*M+2);
	    // printf("\nhnew = %g, thnew = %g, phinew = %f\n", hnew, thnew, phinew);
      
	    if( hnew <= 0 ){
		fprintf(stderr, "new time step is negative\n");
		break;
	    }
	    Cqcgl1d tcgl(N, d, hnew, Mu, Br, Bi, Dr, Di, Gr, Gi);
	    VectorXd newF = tcgl.multiF(xnew, nstp, thnew, phinew);
	    fprintf(stdout, "new err = %g\n", newF.norm());
	    if (newF.norm() < err){
		x = xnew;
		h = hnew;
		th = thnew;
		phi = phinew;
		lam = lam/10;
		break;
	    }
	    else{
		lam *= 10;
		// if(doesPrint) printf(" lam = %g\n", lam);
		if( lam > 1e10) {
		    fprintf(stderr, "lam = %g too large \n", lam);
		    break;
		}
	    }
	    
	}
    }

    Cqcgl1d finalCgl(N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi);
    VectorXd F = finalCgl.multiF(x, nstp, th, phi);
    return std::make_tuple(x, h, th, phi, F.norm());
}

#endif
