#ifndef KSDIM_H
#define KSDIM_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> 
anglePO(const std::string fileName, const std::string ppType,
	const int ppId, const Eigen::MatrixXi subspDim);

void
anglePOs(const std::string fileName, const std::string ppType,
	const int N, const int NN,
	const std::string saveFolder,
	const std::string spType, const int M = 29);


#endif	/* KSDIM_H */

