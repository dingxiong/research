/* to comiple:
 * g++ ksrefine.cc std=c++0x -O3 -march=corei7 -msse2 -msse4
 * -L ../../lib -I ../../include -I $XDAPPS/eigen/include/eigen3
 * -lksint -lfftw3 -lm
 */

#ifndef KSREFINE_H
#define KSREFINE_H

#include "ksint.hpp"
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class KSrefine {

public:
  //////////////////////////////////////////////////
  // variables
  const int N;			/* N = 32 */
  const double L;		/* L = 22 */

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
  Eigen::VectorXd 
  multiF(KS &ks, const Eigen::ArrayXXd &x, const int nstp, 
	 const std::string ppType, const double th = 0.0);
  std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> 
  multishoot(KS &ks, const Eigen::ArrayXXd &x, const int nstp, 
	     const std::string ppType, const double th = 0.0, 
	     const bool Print = false);

  //////////////////////////////////////////////////
  KSrefine(int N = 32, double L = 22);
  explicit KSrefine(const KSrefine &x);
  KSrefine & operator=(const KSrefine &x);
  ~KSrefine();

protected:
  
  std::vector<Eigen::Triplet<double> >
  triMat(const MatrixXd &A, const size_t M = 0, const size_t N = 0);
  std::vector<Eigen::Triplet<double> >
  triDiag(const size_t n, const double x, const size_t M = 0, 
	  const size_t N = 0 );
  int
  calColSizeDF(const int &rows, const int &cols, const std::string ppType);
  
};
#endif	/* KSREFINE_H */
