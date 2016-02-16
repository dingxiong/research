#ifndef KSDIM_H
#define KSDIM_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h> 		/* for create folder */

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> 
anglePO(const std::string fileName, const std::string ppType,
	const int ppId, const Eigen::MatrixXi subspDim);

void
anglePOs(const std::string fileName, const std::string ppType,
	const int N, const int NN,
	const std::string saveFolder,
	const std::string spType, const int M = 29);
void
anglePOs(const std::string fileName, const std::string ppType,
	 const int N, const std::vector<int> ppIds,
	 const std::string saveFolder,
	 const std::string spType, const int M);

Eigen::MatrixXd partialHyperb(const std::string fileName,
			      const std::string ppType,
			      const int ppId);
void partialHyperbOneType(const std::string fileName,
			  const std::string ppType,
			  const int NN, const std::string saveFolder);
void partialHyperbAll(const std::string fileName,
		      const int NNppo, const int NNrpo,
		      const std::string saveFolder);
Eigen::MatrixXd
localFE(const std::string fileName, const std::string ppType, const int ppId);
void localFEOneType(const std::string fileName, const std::string ppType,
		    const int NN, const std::string saveFolder);
void localFEAll(const std::string fileName, const int NNppo, const int NNrpo,
		const std::string saveFolder);

#endif	/* KSDIM_H */

