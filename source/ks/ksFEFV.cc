#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include <H5Cpp.h>
#include "ksFEFV.hpp"
#include "myH5.hpp"
#include "ksint.hpp"
#include "ped.hpp"

using namespace std;
using namespace Eigen;

/** @brief calculate Floquet exponents and Floquet vectors of KS system.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @param[in] MaxN maximal number of PED iteration
 *  @param[in] tol tolerance of PED
 *  @param[in] nqr spacing
 *  @return FE and FV
 */
pair<MatrixXd, MatrixXd>
KScalFEFV(const string fileName,
	  const string ppType,
	  const int ppId,
	  const int L /* = 22 */,
	  const int MaxN /* = 80000 */,
	  const double tol /* = 1e-15 */,
	  const int nqr /* = 1 */,
	  const int trunc /* = 0 */){
    // get the initla condition of the orbit
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);

    const int N = a.rows();
    const int Nks = N + 2;
    
    // solve the Jacobian of this po.
    KS ks(Nks, L);
    std::pair<ArrayXXd, ArrayXXd> tmp2 = ks.intgj(a.col(0), T/nstp, nstp, nqr);
    MatrixXd daa = tmp2.second; 
    daa = daa.rightCols(daa.cols() - daa.rows()).eval();
  
    // calculate the Flqouet exponents and vectors.
    PED ped;
    ped.reverseOrder(daa); // reverse order.
    if(ppType.compare("ppo") == 0)
	daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
    else // R*J for rpo
	daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
    pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, MaxN, tol, false, trunc);
    MatrixXd &eigvals = eigv.first; 
    eigvals.col(0) = eigvals.col(0).array()/T;
    MatrixXd &eigvecs = eigv.second;

    return make_pair(eigvals, eigvecs);
}


/** @brief calculate Floquet exponents of KS system.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @param[in] MaxN maximal number of PED iteration
 *  @param[in] tol tolerance of PED
 *  @param[in] nqr spacing
 *  @return FE and FV
 */
MatrixXd
KScalFE(const string fileName,
	const string ppType,
	const int ppId,
	const int L /* = 22 */,
	const int MaxN /* = 80000 */,
	const double tol /* = 1e-15 */,
	const int nqr /* = 1 */){
    // get the initla condition of the orbit
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);

    const int N = a.rows();
    const int Nks = N + 2;
    
    // solve the Jacobian of this po.
    KS ks(Nks, L);
    pair<ArrayXXd, ArrayXXd> tmp2 = ks.intgj(a.col(0), T/nstp, nstp, nqr);
    MatrixXd daa = tmp2.second;
    daa = daa.rightCols(daa.cols() - daa.rows()).eval();
  
    // calculate the Flqouet exponents and vectors.
    PED ped;
    ped.reverseOrder(daa); // reverse order.
    if(ppType.compare("ppo") == 0)
	daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
    else // R*J for rpo
	daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
    MatrixXd eigvals = ped.EigVals(daa, MaxN, tol, false);
    eigvals.col(0) = eigvals.col(0).array()/T;

    return eigvals;
}

MatrixXd
KSgetR(const string fileName,
       const string ppType,
       const int ppId,
       const int L /* = 22 */,
       const int MaxN /* = 80000 */,
       const double tol /* = 1e-15 */,
       const int nqr /* = 1 */){
    
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);

    const int N = a.rows();
    const int Nks = N + 2;
    
    // solve the Jacobian of this po.
    KS ks(Nks, L);
    pair<ArrayXXd, ArrayXXd> tmp2 = ks.intgj(a.col(0), T/nstp, nstp, nqr);
    MatrixXd daa = tmp2.second;
    daa = daa.rightCols(daa.cols() - daa.rows()).eval();
  
    // calculate the Flqouet exponents and vectors.
    PED ped;
    ped.reverseOrder(daa); // reverse order.
    if(ppType.compare("ppo") == 0)
	daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
    else // R*J for rpo
	daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
    MatrixXd eigvals = ped.EigVals(daa, MaxN, tol, false);
    eigvals.col(0) = eigvals.col(0).array()/T;

    return daa;
}

void 
KScalWriteFE(const string inputfileName,
	     const string outputfileName,
	     const string ppType,
	     const int ppId,
	     const int L /* = 22 */,
	     const int MaxN /* = 80000 */,
	     const double tol /* = 1e-15 */,
	     const int nqr /* = 1 */){
    MatrixXd eigvals = KScalFE(inputfileName, ppType, ppId, L, MaxN, tol, nqr);  
    MyH5::KSwriteFE(outputfileName, ppType, ppId, eigvals);
}


void 
KScalWriteFEInit(const string inputfileName,
		 const string outputfileName,
		 const string ppType,
		 const int ppId,
		 const int L /* = 22 */,
		 const int MaxN /* = 80000 */,
		 const double tol /* = 1e-15 */,
		 const int nqr /* = 1 */){
    MatrixXd eigvals = KScalFE(inputfileName, ppType, ppId, L, MaxN, tol, nqr);  
    MyH5::KSwriteFE(outputfileName, ppType, ppId, eigvals);

    auto tmp = MyH5::KSreadRPO(inputfileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);
    MyH5::KSwriteRPO(outputfileName, ppType, ppId, a, T, nstp, r, s);
}


void 
KScalWriteFEFV(const string inputfileName,
	       const string outputfileName,
	       const string ppType,
	       const int ppId,
	       const int L /* = 22 */,
	       const int MaxN /* = 80000 */,
	       const double tol /* = 1e-15 */,
	       const int nqr /* = 1 */,
	       const int trunc /* = 0 */){
    pair<MatrixXd, MatrixXd> eigv = KScalFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
    MatrixXd &eigvals = eigv.first;
    MatrixXd &eigvecs = eigv.second; 
  
    MyH5::KSwriteFEFV(outputfileName, ppType, ppId, eigvals, eigvecs);
    
}


/**
 * @brief calculate FE and FV and save inital conditions and FE, FV to outputfile
 */
void 
KScalWriteFEFVInit(const string inputfileName,
		   const string outputfileName,
		   const string ppType,
		   const int ppId,
		   const int L /* = 22 */,
		   const int MaxN /* = 80000 */,
		   const double tol /* = 1e-15 */,
		   const int nqr /* = 1 */,
		   const int trunc /* = 0 */){
    pair<MatrixXd, MatrixXd> eigv = KScalFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
    MatrixXd &eigvals = eigv.first;
    MatrixXd &eigvecs = eigv.second; 
  
    MyH5::KSwriteFEFV(outputfileName, ppType, ppId, eigvals, eigvecs);
    
    auto tmp = MyH5::KSreadRPO(inputfileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);
    MyH5::KSwriteRPO(outputfileName, ppType, ppId, a, T, nstp, r, s);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//  Left eigenvector related 

std::pair<MatrixXd, MatrixXd>
KScalLeftFEFV(const std::string fileName, const std::string ppType, const int ppId,
	      const int L, const int MaxN, const double tol,
	      const int nqr, const int trunc){
    
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);

    const int N = a.rows();
    const int Nks = N + 2;
    
    // solve the Jacobian of this po.
    KS ks(Nks, L);
    std::pair<ArrayXXd, ArrayXXd> tmp2 = ks.intgj(a.col(0), T/nstp, nstp, nqr);
    MatrixXd daa = tmp2.second;
    
    // get the order of J_1^T, J_2^T, ..., (R*J_M)^T 
    int M = daa.cols();
    daa.resize(N, N*M);
    if(ppType.compare("ppo") == 0)
	daa.rightCols(N) = ks.Reflection(daa.rightCols(N)); // R*J for ppo
    else // R*J for rpo
	daa.rightCols(N) = ks.Rotation(daa.rightCols(N), -s*2*M_PI/L);
    for(int i = 0; i < M; i++) {
	MatrixXd tmp = daa.middleCols(i*N, N).transpose();
	daa.middleCols(i*N, N) = tmp;
    }
    
    // calculate the Flqouet exponents and vectors.
    PED ped;
    pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, MaxN, tol, false, trunc);
    MatrixXd &eigvals = eigv.first; 
    eigvals.col(0) = eigvals.col(0).array()/T;
    MatrixXd &eigvecs = eigv.second;

    return make_pair(eigvals, eigvecs);
}

void 
KScalWriteLeftFEFV(const string inputfileName,
		   const string outputfileName,
		   const string ppType,
		   const int ppId,
		   const int L,
		   const int MaxN ,
		   const double tol ,
		   const int nqr ,
		   const int trunc ){
    std::pair<MatrixXd, MatrixXd> eigv = KScalLeftFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
    MatrixXd &eigvals = eigv.first;
    MatrixXd &eigvecs = eigv.second; 
    
    MyH5::KSwriteFEFV(outputfileName, ppType, ppId, eigvals, eigvecs);
}
