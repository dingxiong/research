/* How to compile this program:
 * h5c++ readks.cc ../ped/ped.cc ../ksint/ksint.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../../include -lfftw3
 * -march=corei7 -msse4 -O3
 */
#ifndef READKS_H
#define READKS_H

#include "ksint.hpp"
#include "ped.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <string>

class ReadKS {

public:
  //////////////////////////////////////////////////
  const std::string fileName; // intial condition H5 file
  const std::string fileNameE; // exponents H5 file
  const std::string fileNameEV; // vectors H5 file
  const int N;	// dimension of eigenvectors. default N = 30
  const int Nks; // truncation number of KS. default Nks = 32
  const double L; // size of KS. default L = 22
  //////////////////////////////////////////////////

  //////////////////////////////////////////////////
  ReadKS(std::string s1, std::string s2, std::string s3, int N = 30,
	 int Nks = 32, double L = 22);
  explicit ReadKS(const ReadKS &x);
  ReadKS & operator=(const ReadKS &x);
  ~ReadKS();
  //////////////////////////////////////////////////

  //////////////////////////////////////////////////
  Eigen::MatrixXi 
  checkExistEV(const std::string &ppType, const int NN);  
  std::tuple<Eigen::ArrayXd, double, double>
  readKSorigin(const std::string &ppType, const int ppId);
  std::tuple<Eigen::ArrayXd, double, double, double, double>
  readKSinit(const std::string &ppType, const int ppId);
  std::tuple<MatrixXd, double, double, double, double>
  readKSinitMulti(const std::string fileName, const std::string &ppType, const int ppId);
  void 
  writeKSinit(const std::string fileName, const std::string ppType, 
	      const int ppId,
	      const std::tuple<ArrayXd, double, double, double, double> ksinit
	      );
  void 
  writeKSinitMulti(const std::string fileName, const std::string ppType, 
		   const int ppId,
		   const std::tuple<MatrixXd, double, double, double, double> ksinit
		   );
  Eigen::MatrixXd
  readKSe(const std::string &ppType, const int ppId);
  Eigen::MatrixXd
  readKSve(const std::string &ppType, const int ppId);
  void 
  writeKSe(const std::string &ppType, const int ppId, 
	   const Eigen::MatrixXd &eigvals, const bool rewrite = false);
  void 
  writeKSev(const std::string &ppType, const int ppId,
	    const Eigen::MatrixXd &eigvals, const Eigen::MatrixXd &eigvecs, 
	    const bool rewrite = false);
  std::pair<MatrixXd, MatrixXd>
  calKSFloquet(const std::string ppType, const int ppId, 
	       const int MaxN = 80000, const double tol = 1e-15,
	       const int nqr = 1, const int trunc = 0);
  std::pair<MatrixXd, MatrixXd>
  calKSFloquetMulti(const std::string fileName, 
		    const std::string ppType, const int ppId, 
		    const int MaxN = 80000, const double tol = 1e-15,
		    const int nqr = 1, const int trunc = 0);
  MatrixXd
  calKSFloquetOnlyE(const std::string ppType, const int ppId, 
		    const int MaxN = 80000, const double tol = 1e-15,
		    const int nqr = 1);
  void 
  calKSOneOrbit( const std::string ppType, const int ppId,
		 const int MaxN  = 80000, const double tol = 1e-15,
		 const bool rewrite = false, const int nqr = 1,
		 const int trunc = 0);
  void 
  calKSOneOrbitOnlyE( const std::string ppType, const int ppId,
		      const int MaxN = 80000, const double tol = 1e-15,
		      const bool rewrite = false, const int nqr = 1);
  std::vector< std::pair<double, int> > 
  findMarginal(const Eigen::Ref<const Eigen::VectorXd> &Exponent);
  Eigen::MatrixXi 
  indexSubspace(const Eigen::Ref<const Eigen::VectorXd> &RCP, 
		const Eigen::Ref<const Eigen::VectorXd> &Exponent);

  //////////////////////////////////////////////////
  
};

#endif	/* READKS_H */
