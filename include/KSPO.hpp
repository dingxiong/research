#ifndef KSPO_H
#define KSPO_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ksint.hpp"

class KSPO : public KS{

public:
    //////////////////////////////////////////////////
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    //////////////////////////////////////////////////
    KSPO(int N, double d);
    KSPO & operator=(const KSPO &x);
    ~KSPO();
    
    //////////////////////////////////////////////////
    VectorXd
    MFx(const VectorXd &x, const int nstp, const bool isRPO);
    std::tuple<SpMat, SpMat, VectorXd>
    calJJF(const VectorXd &x, int nstp, const bool isRPO);
    std::tuple<ArrayXXd, double, int>
    findPO_LM(const ArrayXXd &a0, const bool isRPO, const int nstp,		      
	      const double tol, const int maxit, const int innerMaxit);
    
};

template<class Mat>
struct KSPOJJF {    
    KSPO *ks;
    bool isRPO;
    int nstp;
    KSPOJJF(KSPO *ks, int nstp, bool isRPO) : ks(ks), nstp(nstp), isRPO(isRPO){}
    
    std::tuple<KSPO::SpMat, KSPO::SpMat, VectorXd>
    operator()(const VectorXd &x) {
	return ks->calJJF(x, nstp, isRPO);
    }	
};

#endif	/* KSPO_H */
