/** \mainpage some frequently used dense matrix related routines 
 *
 *  \section sec_intro Introduction
 *  \section sec_use usage
 *  Example:
 *  \code
 *  g++ yourfile.cc /path/to/denseRoutines.cc -I/path/to/sparseRoutines.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 */

#ifndef DENSEROUTINES_H
#define DENSEROUTINES_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace denseRoutines {

    using namespace std;
    using namespace Eigen;

    double angleSubspace(const Ref<const MatrixXd> &A,
			 const Ref<const MatrixXd> &B);
    double angleSpaceVector(const Ref<const MatrixXd> &Q,
			    const Ref<const VectorXd> &V);
   
    std::vector< std::pair<double, int> > 
    findMarginal(const Ref<const VectorXd> &Exponent,
		 const int k = 2 );
    MatrixXi 
    indexSubspace(const Ref<const VectorXd> &RCP, 
		  const Ref<const VectorXd> &Exponent);
    std::pair<MatrixXi, MatrixXi> 
    subspBound(const MatrixXi subspDim, const MatrixXi ixSp);

    void normc(MatrixXd &A);
    std::vector<int> csort(const VectorXcd &e);
    VectorXcd eEig(const MatrixXd &A);
    MatrixXcd vEig(const MatrixXd &A);
    std::pair<VectorXcd, MatrixXcd> evEig(const MatrixXd &A);
    VectorXd centerRand(const int N, const double frac);
    MatrixXd realv(const MatrixXcd &v);
    MatrixXd orthAxes(const MatrixXd &v);
    MatrixXd orthAxes(const VectorXd &e1, const VectorXd &e2);
    MatrixXd orthAxes(const VectorXd &e1, const VectorXd &e2, 
		      const VectorXd &e3);
    VectorXd spacing(const Ref<const MatrixXd> &v);
    int minDisIndex(const Ref<const VectorXd> &a, 
		    const Ref<const MatrixXd> &v, double &minD);
    int minDisIndex(const Ref<const VectorXd> &a, 
		    const Ref<const MatrixXd> &v);
}

#endif	// DENSEROUTINES_H

