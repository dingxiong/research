#include <iostream>
#include <fstream>
#include "KSPO.hpp"
#include "iterMethod.hpp"
#include "sparseRoutines.hpp"

using namespace std; 
using namespace Eigen;
using namespace sparseRoutines;
using namespace iterMethod;

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
KSPO::MFx(const VectorXd &x, const int nstp, const bool isRPO){
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
	VectorXd gfx = isRPO ? (VectorXd)rotate(fx, th) : (i == m-1 ? (VectorXd)reflect(fx) : fx);
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
	ArrayXXd &fx = tmp.first;
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

/// @return [ (x, T, th, phi), err, flag ]
std::tuple<ArrayXXd, double, int>
KSPO::findPO_LM(const ArrayXXd &a0, const bool isRPO, const int nstp,		      
		const double tol, const int maxit, const int innerMaxit){
    int cols = a0.cols(), rows = a0.rows();
    assert (rows == isRPO ? N : N - 1);
    
    Map<VectorXd> x(a0.data(), cols * rows);
    auto fx = [this, &isRPO, &nstp](const VectorXd &x){ return MFx(x, nstp, isRPO); };
    KSPOJJF<SpMat> jj(this);
    SparseLU<SpMat> solver; 
    
    VectorXd xe;
    std::vector<double> res;
    int flag;
    std::tie(xe, res, flag) = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    if(flag != 0) fprintf(stderr, "PO not converged ! \n");
    
    ArrayXXd states(xe);
    states.resize(rows, cols);
    return std::make_tuple( states, res.back(), flag );
}
