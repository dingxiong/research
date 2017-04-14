#include <iostream>
#include <fstream>
#include "KSPO.hpp"
#include "iterMethod.hpp"
#include "sparseRoutines.hpp"

using namespace std; 
using namespace Eigen;
using namespace sparseRoutines;
using namespace iterMethod;
using namespace MyH5;

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

////////////////////////////////////////////////////////////////////////////////
std::string 
KSPO::toStr(string ppType, int id){
    char g[20];
    sprintf(g, "%06d", id);
    return ppType + '/' + string(g);    
}

std::string
KSPO::toStr(double domainSize, string ppType, int id){
    char g1[20], g2[20];
    sprintf(g1, "%010.6f", domainSize);
    sprintf(g2, "%06d", id);
    return string(g1) + '/' + ppType + '/' + string(g2);
}

// [a, T, nstp, theta, err]
std::tuple<VectorXd, double, int, double, double>
KSPO::read(H5File &file, const std::string groupName, const bool isRPO){
    // H5File file(fileName, H5F_ACC_RDONLY);
    string DS = "/" + groupName + "/";
    
    // Ruslan's orignal data does not have nstp
    int nstp = checkGroup(file, groupName + "/nstp", false) ? 
	readScalar<int>(file, DS + "nstp") : 0;
    
    double theta = isRPO ? readScalar<double>(file, DS + "theta") : 0;
    
    return make_tuple(readMatrixXd(file, DS + "a").col(0),
		      readScalar<double>(file, DS + "T"),
		      nstp, 
		      theta,
		      readScalar<double>(file, DS + "err")
		      );
}

/** [a, T, nstp, theta, err]
 * @note group should be a new group
 */
void 
KSPO::write(H5File &file, const std::string groupName, const bool isRPO,
	    const ArrayXd &a, const double T, const int nstp,
	    const double theta, const double err){
    // H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
    
    writeMatrixXd(file, DS + "a", a);
    writeScalar<double>(file, DS + "T", T);
    if(nstp > 0)	   // Ruslan's orignal data does not have nstp
	writeScalar<int>(file, DS + "nstp", nstp);
    if(isRPO)
	writeScalar<double>(file, DS + "theta", theta);
    writeScalar<double>(file, DS + "err", err);
}

MatrixXd 
KSPO::readE(H5File &file, const std::string groupName){
    string DS = "/" + groupName + "/";
    MatrixXd e = readMatrixXd(file, DS + "e");
    return e;
}

MatrixXd 
KSPO::readV(H5File &file, const std::string groupName){
    string DS = "/" + groupName + "/";
    MatrixXd v = readMatrixXd(file, DS + "ve");
    return v;
}

void 
KSPO::writeE(H5File &file, const std::string groupName, 
	     const MatrixXd &e){
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
    writeMatrixXd(file, DS + "e", e);
}

void 
KSPO::writeV(H5File &file, const std::string groupName, 
	     const MatrixXd &v){
    // H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
    writeMatrixXd(file, DS + "v", v);
}

void 
KSPO::move(H5File &fin, std::string gin, H5File &fout, std::string gout,
	   int flag){
    // VectorXd a;
    // VectorXcd e;
    // MatrixXcd v;
    // double wth, wphi, err;

    // std::tie(a, wth, wphi, err) = read(fin, gin);
    // if (flag == 1 || flag == 2) e = readE(fin, gin);
    // if (flag == 2) v = readV(fin, gin);
    
    // write(fout, gout, a, wth, wphi, err);
    // if (flag == 1 || flag == 2) writeE(fout, gout, e);
    // if (flag == 2) writeV(fout, gout, v);
}

////////////////////////////////////////////////////////////////////////////////

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
	
	VectorXd fx = intgC(xi.head(N-2), t/nstp, t, nstp); // single state
	VectorXd gfx = isRPO ? (VectorXd)rotate(fx, th) : (i == m-1 ? (VectorXd)reflect(fx) : fx);
	F.segment(i*n, N-2) = gfx - xn.head(N-2);
    }
    return F;
}

/**
 * @brief multishooting
 *
 * For RPO 
 * Here J_i  = | g*J(x, t),      g*v(f(x,t)),  g*t(f(x,t))  | 
 *             |     v(x),          0             0         |
 *             |     t(x),          0             0         |
 * For PPO
 * i = 1, ..., m-1
 * Here J_i  = | J(x, t),      v(f(x,t))  | 
 *             |    v(x),          0      |
 * i = m
 * Here J_i  = | g*J(x, t),      g*v(f(x,t)) | 
 *             |     v(x),          0        |
 *
 * If m = 1, then it reduces to single shooting
 *
 * @note I do not enforce the constraints              
 */
std::tuple<SpMat, SpMat, VectorXd>
KSPO::multiCalJJF(const VectorXd &x, int nstp, const bool isRPO){
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
	VectorXd vgfx = velocity(gfx);
	
	vector<Tri> triJ = triMat(gJ, i*n, i*n);
	nz.insert(nz.end(), triJ.begin(), triJ.end());
	vector<Tri> triv = triMat(vgfx, i*n, i*n+N-2);
	nz.insert(nz.end(), triv.begin(), triv.end());
	
	if(isRPO){
	    VectorXd tgfx = gTangent(gfx); 
	    vector<Tri> tritx = triMat(tgfx, i*n, i*n+N-1);
	    nz.insert(nz.end(), tritx.begin(), tritx.end());
	}
	
	// // constraint
	// ArrayXXd v = velocity(xi.head(N-2)).transpose();
	// triv = triMat(v, i*n+N-2, i*n);
	// nz.insert(nz.end(), triv.begin(), triv.end());
	// if(isRPO){
	//     ArrayXXd t = gTangent(xi.head(N-2)).transpose();
	//     vector<Tri> tritx = triMat(t, i*n+N-1, i*n);
	//     nz.insert(nz.end(), tritx.begin(), tritx.end());
	// }

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

// single shooting
// Here   J  = | g*J(x, t) - I,  g*v(f(x,t)),  g*t(f(x,t))  | 
//             |     v(x),          0             0         |
//             |     t(x),          0             0         |
std::tuple<MatrixXd, MatrixXd, VectorXd>
KSPO::calJJF(const VectorXd &x, int nstp, const bool isRPO){
    int n = isRPO ? N : N - 1;
    assert(x.size() == n);
    
    MatrixXd DF(n, n); DF.setZero();
    VectorXd F(n); F.setZero();
    
    double t = x(N-2);
    double th = isRPO ? x(N-1) : 0;
    assert( t > 0 );
    
    auto tmp = intgjC(x.head(N-2), t/nstp, t, nstp);
    ArrayXXd &fx = tmp.first;
    ArrayXXd &J = tmp.second;

    VectorXd gfx = isRPO ? rotate(fx, th) :  reflect(fx);
    F.head(N-2) = gfx - x.head(N-2);	
    
    DF.topLeftCorner(N-2, N-2) = (isRPO ? rotate(J, th) : reflect(J)).matrix() - MatrixXd::Identity(N-2, N-2) ;
    DF.col(N-2).head(N-2) = velocity(gfx);    
    if(isRPO) DF.col(N-1).head(N-2) = gTangent(gfx); 
    
    // DF.row(N-2).head(N-2) = velocity(x.head(N-2)).transpose();
    // if(isRPO) DF.row(N-1).head(N-2) = gTangent(x.head(N-2)).transpose();
    
    
    MatrixXd JJ = DF.transpose() * DF;
    MatrixXd D  = JJ.diagonal().asDiagonal();
    VectorXd df = DF.transpose() * F; 

    return std::make_tuple(JJ, D, df);
}


/// @return [ (x, T, th), err, flag ]
std::tuple<ArrayXXd, double, int>
KSPO::findPO_LM(const ArrayXXd &a0, const bool isRPO, const int nstp,		      
		const double tol, const int maxit, const int innerMaxit){
    int cols = a0.cols(), rows = a0.rows();
    assert (rows == isRPO ? N : N - 1);
    
    Map<const VectorXd> x(a0.data(), cols * rows);
    auto fx = [this, &isRPO, &nstp](const VectorXd &x){ return MFx(x, nstp, isRPO); };
    
    VectorXd xe;
    std::vector<double> res;
    int flag;

    if(cols == 1){
	KSPO_JJF<MatrixXd> jj(this, nstp, isRPO);
	PartialPivLU<MatrixXd> solver;
	std::tie(xe, res, flag) = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    }
    else {
	KSPO_MULTIJJF<SpMat> jj(this, nstp, isRPO);
	SparseLU<SpMat> solver; 	
	std::tie(xe, res, flag) = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    }
    if(flag != 0) fprintf(stderr, "PO not converged ! \n");
    
    ArrayXXd states(xe);
    states.resize(rows, cols);
    return std::make_tuple( states, res.back(), flag );
}
