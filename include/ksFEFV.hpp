#ifndef KSFEFV_H
#define KSFEFV_H

#include <Eigen/Dense>


std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
KScalFEFV(const std::string fileName,
	  const std::string ppType,
	  const int ppId,
	  const int L = 22 ,
	  const int MaxN = 80000,
	  const double tol = 1e-15,
	  const int nqr = 1,
	  const int trunc = 0);

Eigen::MatrixXd
KScalFE(const std::string fileName,
	const std::string ppType,
	const int ppId,
	const int L = 22,
	const int MaxN = 80000,
	const double tol = 1e-15,
	const int nqr = 1);

void 
KScalWriteFE(const std::string inputfileName,
	     const std::string outputfileName,
	     const std::string ppType,
	     const int ppId,
	     const int L = 22,
	     const int MaxN = 80000,
	     const double tol = 1e-15,
	     const int nqr = 1);
void
KScalWriteFEInit(const std::string inputfileName,
		 const std::string outputfileName,
		 const std::string ppType,
		 const int ppId,
		 const int L = 22,
		 const int MaxN = 80000,
		 const double tol = 1e-15,
		 const int nqr = 1);

void 
KScalWriteFEFV(const std::string inputfileName,
	       const std::string outputfileName,
	       const std::string ppType,
	       const int ppId,
	       const int L = 22,
	       const int MaxN = 80000,
	       const double tol = 1e-15,
	       const int nqr = 1,
	       const int trunc = 0);

void 
KScalWriteFEFVInit(const std::string inputfileName,
		   const std::string outputfileName,
		   const std::string ppType,
		   const int ppId,
		   const int L = 22,
		   const int MaxN = 80000,
		   const double tol = 1e-15,
		   const int nqr = 1,
		   const int trunc = 0);

#endif	/* KSFEFV_H */

