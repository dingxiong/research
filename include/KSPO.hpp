#ifndef KSPO_H
#define KSPO_H

#include "ksint.hpp"
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>

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
    std::tuple<VectorXd, double, double>
    findPO(const Eigen::ArrayXd &a0, const double T, const int Norbit, 
	   const int M, const std::string ppType,
	   const double hinit = 0.1,
	   const double th0 = 0, 
	   const int MaxN = 100, 
	   const double tol = 1e-14, 
	   const bool Print = false,
	   const bool isSingle = false);
    std::tuple<MatrixXd, double, double, double>
    findPOmulti(const Eigen::ArrayXd &a0, const double T, const int Norbit, 
		const int M, const std::string ppType,
		const double hinit = 0.1,
		const double th0 = 0, 
		const int MaxN = 100, 
		const double tol = 1e-14, 
		const bool Print = false,
		const bool isSingle = false);
  
};
#endif	/* KSPO_H */
