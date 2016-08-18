#include "CQCGL1dRpo.hpp"
#include <functional>   // std::bind
#include "myH5.hpp"

#define cee(x) (cout << (x) << endl << endl)

namespace ph = std::placeholders;

using namespace denseRoutines;
using namespace sparseRoutines;
using namespace iterMethod;
using namespace Eigen;
using namespace std;
using namespace MyH5;


//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

// A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
CQCGL1dRpo::CQCGL1dRpo(int N, double d,
		       double Mu, double Dr, double Di, double Br, double Bi, 
		       double Gr, double Gi, int dimTan):
    CQCGL1d(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}

// A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
CQCGL1dRpo::CQCGL1dRpo(int N, double d, 
		       double b, double c, double dr, double di, 
		       int dimTan):
    CQCGL1d(N, d, b, c, dr, di, dimTan){}
    
// iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
CQCGL1dRpo::CQCGL1dRpo(int N, double d,
		       double delta, double beta, double D, double epsilon,
		       double mu, double nu, int dimTan) :
    CQCGL1d(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

CQCGL1dRpo::~CQCGL1dRpo(){}

CQCGL1dRpo & CQCGL1dRpo::operator=(const CQCGL1dRpo &x){
    return *this;
}

//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////
std::string 
CQCGL1dRpo::toStr(double x){
    char buffer [20];
    sprintf (buffer, "%013.6f", x);
    return std::string(buffer);
}

std::string
CQCGL1dRpo::toStr(double x, double y, int id){
    return toStr(x) + '/' + toStr(y) + '/' + to_string(id);
}

/**
 * @note group should be a new group
 * [x, T,  nstp, theta, phi, err]
 */
void CQCGL1dRpo::writeRpo(const string fileName, const string groupName,
			  const MatrixXd &x, const double T, const int nstp,
			  const double th, const double phi, double err){

    H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";

    writeMatrixXd(file, DS + "x", x);
    writeScalar<double>(file, DS + "T", T);
    writeScalar<int>(file, DS + "nstp", nstp);
    writeScalar<double>(file, DS + "th", th);
    writeScalar<double>(file, DS + "phi", phi);
    writeScalar<double>(file, DS + "err", err);
}

void CQCGL1dRpo::writeRpo2(const std::string fileName, const string groupName, 
			   const MatrixXd &x, const int nstp, double err){	
    MatrixXd tmp = x.bottomRows(3).rowwise().sum();
    double T = tmp(0);
    double th = tmp(1);
    double phi = tmp(2);
    
    writeRpo(fileName, groupName, x, T, nstp, th, phi, err);
}

std::tuple<MatrixXd, double, int, double, double, double>
CQCGL1dRpo::readRpo(const string fileName, const string groupName){
    H5File file(fileName, H5F_ACC_RDONLY);
    string DS = "/" + groupName + "/";
    
    return make_tuple(readMatrixXd(file, DS + "x"),
		      readScalar<double>(file, DS + "T"),
		      readScalar<int>(file, DS + "nstp"),
		      readScalar<double>(file, DS + "th"),
		      readScalar<double>(file, DS + "phi"),
		      readScalar<double>(file, DS + "err")
		      );
    
}

/**
 * @brief move rpo from one file, group to another file, group
 */
void
CQCGL1dRpo::moveRpo(string infile, string ingroup, 
		    string outfile, string outgroup){
    MatrixXd x;
    double T, th, phi, err;
    int nstp;
    
    std::tie(x, T, nstp, th, phi, err) = readRpo(infile, ingroup);
    writeRpo(outfile, outgroup, x, T, nstp, th, phi, err);
}


//======================================================================

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
VectorXd CQCGL1dRpo::MFx(const VectorXd &x, int nstp){
    assert( (x.size() - 3) % Ndim == 0 );
    int m = (x.size() - 3) / Ndim;
	
    Vector3d t = x.tail<3>();	   /* T, theta, phi */
    assert(t(0) > 0);		   /* make sure T > 0 */
    VectorXd F(VectorXd::Zero(Ndim * m + 3));

    for(size_t i = 0; i < m; i++){
	VectorXd fx = intg(x.segment(i*Ndim, Ndim), t(0)/nstp/m, nstp, nstp).rightCols<1>();
	if(i != m-1){		// the first m-1 vectors
	    F.segment(i*Ndim, Ndim) = fx - x.segment((i+1)*Ndim, Ndim);
	}
	else{			// the last vector
	    F.segment(i*Ndim, Ndim) = Rotate(fx, t(1), t(2)).matrix() - x.head(Ndim);
	}
    }
	
    return F;
}
    
/* 
 * @brief get the multishooting product J * dx. Dimension [nstp*m+3, 1]
 * @see MFx()
 */
VectorXd CQCGL1dRpo::MDFx(const VectorXd &x, const VectorXd &dx, int nstp){
    assert( (x.size() - 3) % Ndim == 0 );
    int m = (x.size() - 3) / Ndim;

    Vector3d t = x.tail<3>();
    Vector3d dt = dx.tail<3>();
    assert(t(0) > 0); 		/* make sure T > 0 */
    VectorXd DF(VectorXd::Zero(Ndim * m + 3));
    
    for(size_t i = 0; i < m; i++){
	ArrayXd xt = x.segment(i*Ndim, Ndim);
	ArrayXd dxt = dx.segment(i*Ndim, Ndim);
	ArrayXXd tmp = intgv(xt, dxt, t(0)/nstp/m, nstp); /* f(x, t) and J(x, t)*dx */
	
	VectorXd v1 = velocity(xt); /* v(x) */
	VectorXd v2 = velocity(tmp.col(0));   /* v(f(x, t)) */
	VectorXd t1 = transTangent(xt);
	VectorXd t2 = phaseTangent(xt);

	// update the last 3 elements
	DF(m * Ndim) += v1.dot(dxt.matrix());
	DF(m * Ndim + 1) += t1.dot(dxt.matrix());
	DF(m * Ndim + 2) += t2.dot(dxt.matrix());
	
	if(i != m-1){
	    DF.segment(i*Ndim, Ndim) = tmp.col(1).matrix()
		- dx.segment((i+1)*Ndim , Ndim)
		+ 1.0 / m * v2 * dt(0);
	}
	else{
	    ArrayXd gfx = Rotate(tmp.col(0), t(1), t(2)); /* g(theta, phi)*f(x, t) */
	    VectorXd gJx = Rotate(tmp.col(1), t(1), t(2)); /* g(theta, phi)*J(x,t)*dx */
	    DF.segment(i*Ndim, Ndim) = gJx
		- dx.segment(0, Ndim)
		+ 1.0/m * Rotate(v2, t(1), t(2)).matrix() * dt(0)
		+ transTangent(gfx).matrix() * dt(1)
		+ phaseTangent(gfx).matrix() * dt(2)
		;
	}
    }
    
    return DF;
    
}

/**
 * @brief         form [g*f(x,t) - x, ...]
 *
 * Form the difference vector, which consists of m pieces, each piece
 * correspond to (x, t, theta, phi). If m = 0, then it reduces to single
 * shooting.
 * 
 * @param[in] x   [(Ndim + 3)*m, 1] dimensional vector. 
 * @return        vector F_i(x, t) =
 *                  | g*f(x, t) - x|
 *                  |       0      |
 *                  |       0      |
 *                  |       0      |
 *                for i = 1, 2, ..., m
 */
VectorXd CQCGL1dRpo::MFx2(const VectorXd &x, int nstp){
    int n = Ndim + 3;
    assert( x.size() % n == 0 );
    int m = x.size() / n;

    VectorXd F(VectorXd::Zero(n*m));
	
    for(int i = 0; i < m; i++){
	VectorXd xi = x.segment(n*i, n);
	int j = (i+1) % m;
	VectorXd xn = x.segment(n*j, n);
	
	double t = xi(Ndim);
	double th = xi(Ndim+1);
	double phi = xi(Ndim+2);	
	assert(t > 0);
	
	VectorXd fx = intg(xi.head(Ndim), t/nstp/m, nstp, nstp).rightCols<1>();
	
	F.segment(i*n, Ndim) = Rotate(fx, th, phi).matrix() - xn.head(Ndim);
    }
	
    return F;
}

/**
 * @brief get the product J * dx
 *
 * If m = 1, then it reduces to single shooting
 *
 * Here J_i  = | g*J(x, t) - I,  g*v(f(x,t)),  g*t1(f(x,t)),  g*t2(f(x,t))| 
 *             |     v(x),          0             0                  0    |
 *             |     t1(x),         0             0                  0    |
 *             |     t2(x),         0             0                  0    |
 */
VectorXd CQCGL1dRpo::MDFx2(const VectorXd &x, const VectorXd &dx, int nstp){
    int n = Ndim + 3;
    assert( x.size() % n == 0 );
    int m = x.size() / n;
	
    VectorXd DF(VectorXd::Zero(n*m));

    for (int i = 0; i < m; i++) {
	VectorXd xi = x.segment(i*n, n);
	VectorXd dxi = dx.segment(i*n, n);
	int j = (i+1) % m;
	VectorXd dxn = dx.segment(j*n, n);
	
	double t = xi(Ndim);
	double th = xi(Ndim+1);
	double phi = xi(Ndim+2);
	assert( t > 0 );
	double dt = dxi(Ndim);
	double dth = dxi(Ndim+1);
	double dphi = dxi(Ndim+2);
	
	MatrixXd tmp = intgv(xi.head(Ndim), dxi.head(Ndim), t/nstp/m, nstp);

	VectorXd gfx = Rotate(tmp.col(0), th, phi);
	VectorXd gJx = Rotate(tmp.col(1), th, phi);
	
	VectorXd v = velocity(xi.head(Ndim));
	VectorXd vgf = velocity(gfx); 

	VectorXd t1 = transTangent(xi.head(Ndim));
	VectorXd tgf1 = transTangent(gfx);

	VectorXd t2 = phaseTangent(xi.head(Ndim));
	VectorXd tgf2 = phaseTangent(gfx);
	
	DF.segment(i*n, Ndim) = gJx + vgf*dt + tgf1*dth + tgf2*dphi - dxn.head(Ndim);
	DF(i*n+Ndim) = v.dot(dxi.head(Ndim));
	DF(i*n+Ndim+1) = t1.dot(dxi.head(Ndim));
	DF(i*n+Ndim+2) = t2.dot(dxi.head(Ndim));
	
    }
    
    return DF;

}

//======================================================================
#if 0

/**
 * @brief find rpo in cqcgl 1d system
 *
 * @return [x, T, theta, phi, err]
 */
std::tuple<VectorXd, double, double, double, double>
CQCGL1dRpo::findRPO(const VectorXd &x0, const double T,
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
    auto fx = std::bind(&CQCGL1dRpo::Fx, this, ph::_1);
    auto dfx = std::bind(&CQCGL1dRpo::DFx, this, ph::_1, ph::_2);
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
CQCGL1dRpo::findRPOM(const MatrixXd &x0, const double T,
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
    auto fx = std::bind(&CQCGL1dRpo::MFx, this, ph::_1);
    auto dfx = std::bind(&CQCGL1dRpo::MDFx, this, ph::_1, ph::_2);
    
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
CQCGL1dRpo::findRPO_hook(const VectorXd &x0, const double T,
			 const double th0, const double phi0,
			 const double tol,
			 const double minRD,
			 const int maxit,
			 const int maxInnIt,
			 const double GmresRtol,
			 const int GmresRestart,
			 const int GmresMaxit){
    assert(x0.size() == Ndim);
    auto fx = std::bind(&CQCGL1dRpo::Fx, this, ph::_1);
    auto dfx = std::bind(&CQCGL1dRpo::DFx, this, ph::_1, ph::_2);
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
CQCGL1dRpo::findRPOM_hook(const MatrixXd &x0, const double T,
			  const double th0, const double phi0,
			  const double tol,
			  const double minRD,
			  const int maxit,
			  const int maxInnIt,
			  const double GmresRtol,
			  const int GmresRestart,
			  const int GmresMaxit){
    assert(x0.cols() == M && x0.rows() == Ndim);
    auto fx = std::bind(&CQCGL1dRpo::MFx, this, ph::_1);
    auto dfx = std::bind(&CQCGL1dRpo::MDFx, this, ph::_1, ph::_2);
    
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

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// new version of F and DF, This version use a full multishooting method

VectorXd CQCGL1dRpo::calPre(const VectorXd &x, const VectorXd &dx){
    int N = Ndim + 3;
    VectorXd DF(VectorXd::Zero(N*M));
    
    for (int i = 0; i < M; i++) {
	VectorXd xi = x.segment(i*N, N);
	VectorXd dxi = dx.segment(i*N, N);

	double th = xi(Ndim+1);
	double phi = xi(Ndim+2);

	DF.segment(i*N, N) << 
	    Rotate(dxi.head(Ndim), -th, -phi),
	    dxi.tail(3);
    }

    return DF;
}

std::tuple<MatrixXd, double, int>
CQCGL1dRpo::findRPOM_hook2(const MatrixXd &x0, 
			   const int nstp,
			   const double tol,
			   const double minRD,
			   const int maxit,
			   const int maxInnIt,
			   const double GmresRtol,
			   const int GmresRestart,
			   const int GmresMaxit){
    int n = Ndim + 3;
    int m = x0.cols();
    assert( x0.rows() == n);

    auto fx = [this, nstp](const VectorXd &x){ return MFx2(x, nstp); };
    auto dfx = [this, nstp](const VectorXd &x, const VectorXd &dx){ return MDFx2(x, dx, nstp); };
    
    // initialize input 
    MatrixXd x(x0);
    x.resize(m * n, 1);
    

    // auto Pre = [this](const VectorXd &x, const VectorXd &dx){VectorXd p = calPre(x, dx); return p; };
    auto Pre = [this](const VectorXd &x, const VectorXd &dx){ return dx; };
    VectorXd xnew;
    std::vector<double> errs;
    int flag;
    // std::tie(xnew, errs, flag) = Gmres0HookPre(fx, dfx, Pre, x, tol, minRD, maxit, maxInnIt,
    // 					       GmresRtol, GmresRestart, GmresMaxit,
    // 					       true, 3);
    std::tie(xnew, errs, flag) = Gmres0Hook(fx, dfx, x, tol, minRD, maxit, maxInnIt,
					    GmresRtol, GmresRestart, GmresMaxit,
					    true, 3);
    if(flag != 0) fprintf(stderr, "RPO not converged ! \n");

    MatrixXd tmp2(xnew);
    tmp2.resize(N, M);
    return std::make_tuple(tmp2,	/* x, th, phi */
			   errs.back(), /* err */
			   flag
			   );
}

/**
 * @brief find req with a sequence of Bi or Gi
 */ 
void 
CQCGL1dRpo::findRpoParaSeq(const std::string file, int id, double step, int Ns, bool isBi){
    double Bi0 = Bi;
    double Gi0 = Gi;
    
    MatrixXd x0;
    double T0, th0, ph0, err0;
    int nstp0;
    std::tie(x0, T0, nstp0, th0, phi0, err0) = readRpo(file, toStr(Bi, Gi, id));
    
    MatrixXd x;
    double T, th, phi, err;
    int nstp, flag;
    
    for (int i = 0; i < Ns; i++){
	if (isBi) Bi += step;
	else Gi += step;
	
	// if exist, use it as seed, otherwise find req
	if ( checkGroup(file, toStr(Bi, Gi, id), false) ){ 
	    std::tie(x0, T0, nstp0, th0, phi0, err0) = readRpo(file, toStr(Bi, Gi, id));
	}
	else {
	    fprintf(stderr, "%g, %g \n", Bi, Gi);
	    std::tie(x, err, flag) = findRPOM_hook2(x0, nstp, wth0, wphi0, 1e-10, 100, 1000);
	    if (flag == 0){
		writeReq(file, toStr(Bi, Gi, id), a, wth, wphi, err);
		a0 = a;
		wth0 = wth;
		wphi0 = wphi;
	    }
	    // else exit(1);
	}
    }
    
    Bi = Bi0; 			// restore Bi, Gi
    Gi = Gi0;
}

#if 0

////////////////////////////////////////////////////////////////////////////////////////////////////
// use LM algorithm to find RPO

std::tuple<SpMat, SpMat, VectorXd> 
CQCGL1dRpo::calJJF(const VectorXd &x){
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
CQCGL1dRpo::findRPOM_LM(const MatrixXd &x0, 
			const double tol,
			const int maxit,
			const int innerMaxit){
    int N = Ndim + 3;
    assert(x0.cols() == M && x0.rows() == N);
    auto fx = std::bind(&CQCGL1dRpo::MFx2, this, ph::_1);
    
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


#endif
