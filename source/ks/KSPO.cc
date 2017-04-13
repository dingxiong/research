#include "KSPO.hpp"
#include "iterMethod.hpp"
#include <iostream>
#include <fstream>

using namespace std; 
using namespace Eigen;

////////////////////////////////////////////////////////////
//                     class KSPO                     //
////////////////////////////////////////////////////////////

/*------------          constructor        ---------------*/
KSPO::KSPO(int N, double d) : KS(N, d) {
}
KSPO & KSPO::operator=(const KSPO &x){ 
    return *this; 
}
KSPO::~KSPO(){}

/**
 * @brief         form [g*f(x,t) - x, ...]
 *
 * Form the difference vector, which consists of m pieces, each piece
 * correspond to (x, t, theta) for RPO and (x, t) for PPO.
 * If m = 1, then it reduces to single
 * shooting.
 * 
 * @param[in] x   [N*m, 1] dimensional vector for RPO and [(N-1)*m] for PPO
 * @return        vector F_i(x, t) =
 *                  | g*f(x, t) - x|
 *                  |       0      |
 *                  |       0      |
 *                for i = 1, 2, ..., m
 */
VectorXd
KSPO::MFxRPO(const VectorXd &x, const int nstp, const bool isRPO){
    int n = isRPO ? N : N - 1;
    assert( x.size() % n == 0 );
    int m = x.size() / n;
    VectorXd F(n*m); F.setZero();
    
    for(int i = 0; i < m; i++){
	VectorXd xi = x.segment(n*i, n);
	int j = (i+1) % m;
	VectorXd xn = x.segment(n*j, n);
	
	double t = xi(N-2);
	double th = isRPO ? xi(N-1) : 0;
	assert(t > 0);
	
	VectorXd fx = intgC(xi.head(n-2), t/nstp, t, nstp); // single state
	VectorXd gfx = isRPO ? rotate(fx, th) : (i == m-1 ? reflect(fx) : fx);
	F.segment(i*n, N-2) = gfx - xn.head(N-2);
    }
    return F;
}

/**
 * @brief get  J 
 *
 * If m = 1, then it reduces to single shooting
 *
 * For RPO 
 * Here J_i  = | g*J(x, t),      g*v(f(x,t)),  g*t(f(x,t))  | 
 *             |     v(x),          0             0         |
 *             |     t(x),          0             0         |
 *
 * For PPO
 * i = 1, ..., m-1
 * Here J_i  = | J(x, t),      v(f(x,t))  | 
 *             |    v(x),          0      |
 * i = m
 * Here J_i  = | g*J(x, t),      g*v(f(x,t)) | 
 *             |     v(x),          0        |
 *
 * @note I do not enforce the constraints              
 */
std::tuple<SpMat, SpMat, VectorXd>
KSPO::calJJF(const VectorXd &x, int nstp, const bool isRPO){
    int n = isRPO ? N : N - 1;
    assert( x.size() % n == 0 );
    int m = x.size() / n;
    
    SpMat DF(m*n, m*n);    
    vector<Tri> nz;
    VectorXd F(m*n);
    
    for (int i = 0; i < m; i++) {
	VectorXd xi = x.segment(i*n, n);
	int j = (i+1) % m;
	VectorXd xn = x.segment(n*j, n);
	
	double t = xi(N-2);
	double th = isRPO ? xi(N-1) : 0;
	assert( t > 0 );
	
	auto tmp = intgjC(xi.head(N-2), t/nstp, t, nstp);
	ArrayXd &fx = tmp.first;
	ArrayXXd &J = tmp.second;

	VectorXd gfx = isRPO ? rotate(fx, th) : (i == m-1 ? reflect(fx) : fx);
	F.segment(i*n, N-2) = gfx - xn.head(N-2);	

	ArrayXXd gJ = isRPO ? rotate(J, th) : (i == m-1 ? reflect(J) : J);
	VectorXd v = velocity(fx);
	VectorXd gvfx = isRPO ? rotate(v, th) : (i == m-1 ? reflect(v) : v); 
	
	vector<Tri> triJ = triMat(gJ, i*n, i*n);
	nz.insert(nz.end(), triJ.begin(), triJ.end());
	vector<Tri> triv = triMat(gvfx, i*n, i*n+N-2);
	nz.insert(nz.end(), triv.begin(), triv.end());
	
	if(isRPO){
	    VectorXd tgfx = gTangent(gfx); 
	    vector<Tri> tritx = triMat(tgfx, i*n, i*n+N-1);
	    nz.insert(nz.end(), tritx.begin(), tritx.end());
	}
	
	// -I on the subdiagonal
	vector<Tri> triI = triDiag(N-2, -1, i*n, j*n);
	nz.insert(nz.end(), triI.begin(), triI.end());
    }
    
    DF.setFromTriplets(nz.begin(), nz.end());
    
    SpMat JJ = DF.transpose() * DF;
    SpMat D  = JJ;
    auto keepDiag = [](const int& row, const int& col, const double&){ return row == col; };
    D.prune(keepDiag);
    VectorXd df = DF.transpose() * F; 

    return std::make_tuple(JJ, D, df);
}

std::tuple<VectorXd, double, double, double, int>
CQCGL1dReq::findPO_LM(const ArrayXd &a0, const double wth0, const double wphi0, 
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

/** @brief find ppo/rpo in KS system
 *
 * @param[in] a0 guess of initial condition
 * @param[in] T guess of period
 * @param[in] h0 guess of time step
 * @param[in] Norbit number of pieces of the orbit, so each segment has Norbit/M.
 * @param[in] M number of pieces for multishooting method
 * @param[in] ppType ppo/rpo
 * @param[in] th0 guess of initial group angle. For ppo, it is zero.
 * @param[in] hinit: the initial time step to get a good starting guess.
 * @param[in] MaxN maximal number of iteration
 * @param[in] tol tolerance of convergence
 * @return initial conditions along the orbit, time step, shift, error
 *
 */
tuple<MatrixXd, double, double, double>
KSPO::findPOmulti(const ArrayXd &a0, const double T, const int Norbit, 
		      const int M, const string ppType,
		      const double hinit /* = 0.1*/,
		      const double th0 /* = 0 */,
		      const int MaxN /* = 100 */, 
		      const double tol /* = 1e-14*/, 
		      const bool Print /* = false */,
		      const bool isSingle /* = false */){ 
  bool Terminate = false;
  assert(a0.rows() == N - 2 && Norbit % M == 0);
  const int nstp = Norbit/M;
  double h = T/Norbit;
  double th = th0;
  double lam = 1;
  SparseLU<SpMat> solver; // used in the pre-CG method

  // prepare the initial state sequence
  KS ks0(N, hinit, L);
  ArrayXXd tmpx = ks0.intg(a0, (int)ceil(T/hinit), 
			(int)floor(T/hinit/M));
  ArrayXXd x = tmpx.leftCols(M); 
  double err = 0; // error 
  
  for(size_t i = 0; i < MaxN; i++){
    if(Print && i%10 == 0) printf("******  i = %zd/%d  ****** \n", i, MaxN);
    KS ks(N, h, L);
    VectorXd F;
    if(!isSingle) F = multiF(ks, x, nstp, ppType, th);
    else F = multiF(ks, x.col(0), Norbit, ppType, th);
    err = F.norm(); 
    if(err < tol){
      fprintf(stderr, "stop at norm(F)=%g for iteration %zd\n", err, i);
      break;
    }
   
    pair<SpMat, VectorXd> p = multishoot(ks, x, nstp, ppType, th, false); 
    SpMat JJ = p.first.transpose() * p.first; 
    VectorXd JF = p.first.transpose() * p.second;
    SpMat Dia = JJ; 
    Dia.prune(KeepDiag());

    for(size_t j = 0; j < 20; j++){
      ////////////////////////////////////////
      // solve the update
      SpMat H = JJ + lam * Dia; 
      pair<VectorXd, vector<double> > cg = iterMethod::ConjGradSSOR<SpMat>
	(H, -JF, solver, VectorXd::Zero(H.rows()), H.rows(), 1e-6);
      VectorXd &dF = cg.first;
      
      ////////////////////////////////////////
      // print the CG infomation
      if(Print)
	printf("CG error %g after %lu iterations.\n", 
	       cg.second.back(), cg.second.size());

      ////////////////////////////////////////
      // update the state
      ArrayXXd xnew = x + Map<ArrayXXd>(&dF(0), N-2, M);
      double hnew = h + dF((N-2)*M)/nstp; // be careful here.
      double thnew = th;
      if(ppType.compare("rpo") == 0) thnew += dF((N-2)*M+1);
      
      // check the new time step positve
      if( hnew <= 0 ){ 
	fprintf(stderr, "new time step is negative\n");
	Terminate = true;
	break;
      }
      
      KS ks1(N, hnew, L);
      VectorXd newF;
      if(!isSingle) newF = multiF(ks1, xnew, nstp, ppType, thnew); 
      else newF = multiF(ks1, xnew.col(0), Norbit, ppType, thnew); 
      if(Print) printf("err = %g \n", newF.norm());
      if (newF.norm() < err){
	x = xnew; h = hnew; th = thnew; 
	lam = lam/10; 
	break;
      }
      else{
	lam *= 10; 
	if(Print) printf("lam = %g \n", lam);
	if( lam > 1e10) {
	  fprintf(stderr, "lam = %g too large.\n", lam); 
	  Terminate = true;
	  break; 
	}
      }
      
    }
    
    if( Terminate )  break;
  }
  
  return make_tuple(x, h, th, err);
}

std::tuple<VectorXd, double, double>
KSPO::findPO(const Eigen::ArrayXd &a0, const double T, const int Norbit, 
		 const int M, const std::string ppType,
		 const double hinit /* = 0.1 */,
		 const double th0 /* = 0 */, 
		 const int MaxN /* = 100 */, 
		 const double tol /* = 1e-14 */, 
		 const bool Print /* = false */,
		 const bool isSingle /* = false */){
  tuple<MatrixXd, double, double, double> 
    tmp = findPOmulti(a0, T, Norbit, M, ppType, hinit, th0, MaxN, tol, Print, isSingle);
  return make_tuple(std::get<0>(tmp).col(0), std::get<1>(tmp), std::get<2>(tmp));
}
