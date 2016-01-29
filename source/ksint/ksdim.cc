#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>
#include <H5Cpp.h>
#include "myH5.hpp"
#include "denseRoutines.hpp"
#include "ksdim.hpp"
#include "ksint.hpp"

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
 * @param[in] M         number of angles
 * @param[in] saveFolder folder name to save angles
 * @see anglePOs()
 */
void
anglePOs(const string fileName, const string ppType,
	 const int N, const int NN,
	 const string saveFolder,
	 const string spType, const int M)
{
    std::vector<int> ppIds(NN);
    for(int i = 0; i < NN; i++) ppIds[i] = i+1;
    anglePOs(fileName, ppType, N, ppIds, saveFolder, spType, M);
}


void
anglePOs(const string fileName, const string ppType,
	 const int N, const std::vector<int> ppIds,
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
    // create subfolders for each ppId
    for(size_t i = 0; i < ppIds.size(); i++){
	std::string tmp = saveFolder + '/' + to_string(ppIds[i]);
	int s = mkdir( tmp.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if(s != 0){
	    fprintf(stderr, "\n creating ppId folder fails. \n");
	    exit(-1);
	}
    }
    ////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////	
    // get the index of POs which converge.
    MatrixXi status = MyH5::checkExistEV(fileName, ppType, ppIds); 
    for(size_t i = 0; i < ppIds.size(); i++){

	if( 1 == status(i,0) ){

	    int ppId = ppIds[i];	      
	    printf("========= i = %zd, ppId = %d ========\n", i, ppId);

	    // create files 
	    ofstream file[M];
	    string angName("ang");
	    for(size_t j = 0; j < M; j++){
		std::string tmp = saveFolder + '/' + to_string(ppId) + '/' + angName + to_string(j) + ".dat";
		file[j].open(tmp, ios::trunc);
		file[j].precision(16);
	    }
		
	    // calculation
	    
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
		
	    // close files
	    for(size_t i = 0; i < M; i++) file[i].close();
	}
    }
}


/**
 * Calucate local expansion rate of FVs along a periodic orbit
 *
 * @param[in] fileName   h5 file that stores the ppo/rpo
 * @param[in] ppType     'rpo' or 'ppo'
 * @param[in] ppId       index of po
 * @return               expansion rate. each column is the expansion rate
 *                       at a specific point along the orbit
 */
MatrixXd partialHyperb(const string fileName, const string ppType,
		       const int ppId)
{
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);
    MatrixXd eigVals = MyH5::KSreadFE(fileName, ppType, ppId); // Floquet exponents
    MatrixXd eigVecs = MyH5::KSreadFV(fileName, ppType, ppId); // Floquet vectors
    
    const int N = eigVals.rows(); // be careful about N when FVs are not complete
    assert ( eigVecs.rows()  % N == 0);
    const int NFV = eigVecs.rows() / N;
    const int M = eigVecs.cols();

    const int Nks = a.rows() + 2;
    const double L = 22;
    assert(nstp % M == 0);
    const int space = nstp / M;
    KS ks(Nks, T/nstp, L);
    auto tmp2 = ks.intgj(a.col(0), nstp, nstp, space);
    ArrayXXd &daa = tmp2.second;
    
    MatrixXd Jv(N, NFV*M); 
    for(size_t i = 0; i < M; i++){
	MatrixXd ve = eigVecs.col(i);
	ve.resize(N, NFV);
	MatrixXd J = daa.col(i); 
	J.resize(N, N);
	Jv.middleCols(i*NFV, NFV) = J * ve; 
    }

    MatrixXd expand(NFV, M);
    for(size_t i = 0; i < NFV; i++){
	if (eigVals(i, 2) == 0) { // real case 
	    for(int j = 0; j < M; j++){
		expand(i, j) = Jv.col(j*NFV+i).norm();
	    }
	}
	else {			// complex case
	    for(int j = 0; j < M; j++){
		double e1 = Jv.col(j*NFV+i).squaredNorm();
		double e2 = Jv.col(j*NFV+i+1).squaredNorm();
		expand(i, j) = sqrt(e1+e2);
		expand(i+1, j) = expand(i, j);
	    }
	    i++;
	}
    }

    return expand;
}

void partialHyperbOneType(const string fileName, const string ppType,
		      const int NN, const string saveFolder){
    ofstream file;
    for(size_t i = 0; i < NN; i++){
	int ppId = i + 1;
	file.open(saveFolder + "FVexpand" + to_string(ppId) + ".dat", ios::trunc);
	file.precision(16);
	printf("========= i = %zd ========\n", i);
	MatrixXd expand = partialHyperb( fileName,  ppType, ppId);
	file << expand << endl;
	file.close();
    }
}

void partialHyperbAll(const string fileName, const int NNppo, const int NNrpo,
		      const string saveFolder){
    partialHyperbOneType(fileName, "ppo", NNppo, saveFolder + "ppo/");
    partialHyperbOneType(fileName, "rpo", NNrpo, saveFolder + "rpo/");
}

/*
void expandDifvAngle(const string fileName, const string ppType,
		     const int ppId, const int gTpos,
		     const MatrixXd difv, const VectorXi indexPO){
    int num = difv.cols();
    int N = difv.rows();
    assert(num == indexPO.size());
    
    auto tmp = MyH5::KSreadRPO(fileName, ppType, ppId);
    MatrixXd &a = std::get<0>(tmp);
    double T = std::get<1>(tmp);
    int nstp = (int) std::get<2>(tmp);
    double r = std::get<3>(tmp);
    double s = std::get<4>(tmp);
    MatrixXd eigVecs = MyH5::KSreadFV(fileName, ppType, ppId); // Floquet vectors
    
    assert ( eigVecs.rows()  % N == 0);
    const int NFV = eigVecs.rows() / N;
    const int M = eigVecs.cols();    
    const int Nks = a.rows() + 2;
    const double L = 22;
    assert(nstp % M == 0);
    const int space = nstp / M;
    KS ks(Nks, T/nstp, L);
    ArrayXXd aa = ks.intg(a.col(0), nstp, space);

    if(ppType.compare("ppo") == 0) {
	aaWhole = ks.half2whole(aa.leftCols(aa.cols()-1));
	eigVecsWhole = ks.half2whole(eigVecs); // this is correct. Think carefully.
    } else {
	aaWhole = aa.leftCols(aa.cols()-1); // one less
	eigVecsWhole = eigVecs;
    }

    MatrixXd veSlice = ks.veToSliceAll(eigVecsWhole, aaWhole, NFV);
    MatrixXd ve_trunc = veTrunc(veSlice, gTpos, Trunc);
    
    
}
*/
