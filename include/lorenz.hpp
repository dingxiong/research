#ifndef LORENZ_H
#define LORENZ_H

#include <Eigen/Dense>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "denseRoutines.hpp"


/**
 *        | sigma * (y -x)     |
 * v(x) = | rho * x - y - x*z  |
 *        | x*y - b*z          |
 */
class Lorenz {
    
public:
    
    double Sigma = 10;
    double B = 8.0 / 3;		/* caution: use float  */
    double Rho = 28;
    
    /* ============================================================ */
    Lorenz();
    ~Lorenz();
    Lorenz & operator=(const Lorenz &x);
    
    /* ============================================================ */
    
    Eigen::Vector3d vel(const Eigen::Ref<const Eigen::Vector3d> &x);
    Eigen::Matrix3d stab(const Eigen::Ref<const Eigen::Vector3d> &x);
    Eigen::MatrixXd velJ(const Eigen::Ref<const Eigen::MatrixXd> &x);
    Eigen::MatrixXd intg(const Eigen::Ref<const Eigen::Vector3d> &x0, const double h, 
			 const int nstp, const int nq);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    intgj(const Eigen::Ref<const Eigen::Vector3d> &x0, const double h, 
	  const int hstp, const int xs, const int js);
    
    Eigen::Matrix3d equilibria();
    Eigen::Matrix3d equilibriaStab(const int i);
    std::pair<Eigen::VectorXcd, Eigen::MatrixXcd>
    equilibriaEV(const int i);
    Eigen::MatrixXd
    equilibriaIntg(const int i, const int j, const double eps,
		   const double h, const int nstp, 
		   const int nq);
};


#endif	/* LORENZ_H */
