/** \mainpage some frequently used spase matrix related routines 
 *
 *  \section sec_intro Introduction
 *  \section sec_use usage
 *  Example:
 *  \code
 *  g++ yourfile.cc /path/to/sparseRoutines.cc -I/path/to/sparseRoutines.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 */

#ifndef SPARSEROUTINES_H
#define SPARSEROUTINES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>

namespace sparseRoutines {

    struct KeepDiag{
	inline bool operator() (const int& row, const int& col,
				const double&) const{
	    return row == col;
	}
    };
    
    std::vector<Eigen::Triplet<double> >
    triMat(const Eigen::MatrixXd &A, const size_t M = 0 , 
	   const size_t N = 0);
    
    std::vector<Eigen::Triplet<double> >
    triDiag(const size_t n, const double x,
	    const size_t M = 0, const size_t N = 0 );
    
}

#endif	// SPARSEROUTINES_H

