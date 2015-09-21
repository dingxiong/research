#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>
#include <H5Cpp.h>
#include "myH5.hpp"
#include "denseRoutines.hpp"
#include "ksdim.hpp"

using namespace std;
using namespace Eigen;

/** @brief calculate angle between two subspaces along an upo
 *
 *  @param[in] fileName file that stores the ppo/rpo information
 *  @param[in] ppType ppo or rpo
 *  @param[in] ppId id of the periodic orbit
 *  @param[in] subspDim 4xM2  matrix storing the bounds of subspaces
 *  @return first matrix : each row stores the angles at one point
 *          second matrix: matrix indicating whether bounds are the
 *          same as specified
 *  @see subspBound
 */
std::pair<MatrixXd, MatrixXi> 
anglePO(const string fileName, const string ppType,
	const int ppId, const MatrixXi subspDim){
    assert(subspDim.rows() == 4);
    MatrixXd eigVals = MyH5::KSreadFE(fileName, ppType, ppId); // Floquet exponents
    MatrixXd eigVecs = MyH5::KSreadFV(fileName, ppType, ppId); // Floquet vectors
    // left and right bounds
    MatrixXi ixSp = denseRoutines::indexSubspace(eigVals.col(2), eigVals.col(0));
  
    const int N = eigVals.rows(); // be careful about N when FVs are not complete
    assert ( eigVecs.rows()  % N == 0);
    const int NFV = eigVecs.rows() / N;
    const int M = eigVecs.cols();
    const int M2 = subspDim.cols();
    MatrixXd ang_po(M, M2);
   
    // calculate the exact bound of this indices.
    std::pair<MatrixXi, MatrixXi> tmp = denseRoutines::subspBound(subspDim, ixSp);
    MatrixXi &bound = tmp.first;
    MatrixXi &boundStrict = tmp.second; 

    for(size_t i = 0; i < M; i++){
	MatrixXd ve = eigVecs.col(i);
	ve.resize(N, NFV);
	for(size_t j = 0; j < M2; j++){      
	    double ang = denseRoutines::angleSubspace(ve.middleCols(bound(0, j), bound(1,j)-bound(0,j)+1), 
						      ve.middleCols(bound(2, j), bound(3,j)-bound(2,j)+1) );
	    ang_po(i, j) = ang;
	}
    }
  
    return std::make_pair(ang_po, boundStrict);
}

/**
 * Test the tangency of tangent bundle.
 *
 * @param[in] N         number of FVs at each point 
 * @param[in] NN        number of periodic orbits
 * @param[in] spType    "vector" or "space"
 */
void
anglePOs(const string fileName, const string ppType,
	 const int N, const int NN,
	 const string saveFolder,
	 const string spType, const int M)
{
    ////////////////////////////////////////////////////////////
    // judge subspace or vector
    MatrixXi subspDim(4, M);
    if(spType.compare("vector") == 0)
	for (size_t i = 0; i < M; i++) subspDim.col(i) << i, i, i+1, i+1;
    else if(spType.compare("space") == 0)
	for (size_t i = 0; i < M; i++) subspDim.col(i) << 0, i, i+1, N-1;
    else {
	printf("invalid spType.\n");
    }
    ////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////
    // specify the save folder
    ofstream file[M];
    string angName("ang");
    for(size_t i = 0; i < M; i++){
	file[i].open(saveFolder + angName + to_string(i) + ".dat", ios::trunc);
	file[i].precision(16);
    }
    ////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////	
    // get the index of POs which converge.
    MatrixXi status = MyH5::checkExistEV(fileName, ppType, NN); 
    for(size_t i = 0; i < NN; i++)
	{
	    if( 1 == status(i,0) ){
		int ppId = i + 1;	      
		printf("========= i = %zd ========\n", i);
		std::pair<MatrixXd, MatrixXi> tmp =
		    anglePO(fileName, ppType, ppId, subspDim);
		MatrixXd &ang = tmp.first;
		MatrixXi &boundStrict = tmp.second;
		cout << boundStrict << endl;
		// check whether there are degeneracy
		MatrixXi pro = boundStrict.row(0).array() *
		    boundStrict.row(1).array() *
		    boundStrict.row(2).array() *
		    boundStrict.row(3).array();
		for(size_t i = 0; i < M; i++){
		    // only keep the orbits whose 4 bounds are not degenerate
		    if(pro(0, i) == 1) file[i] << ang.col(i) << endl;
		}
	    }
	}
    for(size_t i = 0; i < M; i++) file[i].close();
}

