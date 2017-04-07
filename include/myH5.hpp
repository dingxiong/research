#ifndef MYH5_H
#define MYH5_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <unordered_set>
#include "H5Cpp.h"

/**
 * Note,
 * scalars have dimension 0,
 * vectors have dimension 1
 */
namespace MyH5 {
    using namespace H5;
    using namespace std;
    using namespace Eigen;

    template<typename Scalar>
    PredType getPredType();

    template <typename Scalar>
    Scalar readScalar(H5File &file, string DSitem);

    template <typename Scalar>
    void 
    writeScalar(H5File &file, string DSitem, Scalar value);
    
    void 
    writeMatrixXd(H5File &file, string DSitem, const MatrixXd &mat);
    void 
    writeVectorXd(H5File &file, string DSitem, const VectorXd &vec);
    MatrixXd 
    readMatrixXd(H5File &file, string DSitem);
    VectorXd 
    readVectorXd(H5File &file, string DSitem);
    bool 
    checkGroup(H5File &file, const std::string groupName, const bool doCreate);
    bool 
    checkGroup(std::string fileName, const std::string groupName, const bool doCreate);
    vector<string> 
    scanGroup(std::string fileName);
    void 
    scanGroupHelp(hid_t gid, unordered_set<string> &result, vector<string> &curt);
    
    //////////////////////////////////////////////////////////////////////
    /* KS related */
    MatrixXi
    checkExistEV(const string fileName, const string ppType, const int NN);
    MatrixXi
    checkExistEV(const string fileName, const string ppType, const std::vector<int> ppIds);
    std::tuple<MatrixXd, double, double>
    KSreadOrigin(const string fileName, const string &ppType, const int ppId);

    std::tuple<MatrixXd, double, int, double, double>
    KSreadRPO(const string fileName, const string &ppType, const int ppId);
    void 
    KSwriteRPO(const string fileName, const string ppType, const int ppId,
	       const MatrixXd &a, const double T, const int nstp,
	       const double r, const double s
	       );
    void 
    KSmoveRPO(const std::string inFile, const std::string outFile, const std::string ppType, 
	      const int ppId);

    //-----------------------------------------------------------
    MatrixXd
    KSreadFE(const string fileName, const string ppType, const int ppId); 
    void 
    KSwriteFE(const string fileName, const string ppType, const int ppId, 
	      const MatrixXd &eigvals);
    void 
    KSmoveFE(const std::string inFile, const std::string outFile, const std::string ppType,
	     const int ppId);
    
    //-----------------------------------------------------------
    MatrixXd
    KSreadFV(const string fileName, const string ppType, const int ppId);
    void
    KSwriteFV(const string fileName, const string ppType, const int ppId, 
	      const MatrixXd &eigvecs);
    void 
    KSmoveFV(const std::string inFile, const std::string outFile, const std::string ppType,
	     const int ppId);

    //-----------------------------------------------------------
    std::pair<MatrixXd, MatrixXd>
    KSreadFEFV(const string fileName, const string ppType, const int ppId);
    void 
    KSwriteFEFV(const string fileName, const string ppType, const int ppId,
		const MatrixXd &eigvals, const MatrixXd &eigvecs);
    void 
    KSmoveFEFV(const std::string inFile, const std::string outFile, const std::string ppType,
	       const int ppId);
    //-----------------------------------------------------------

    void KScheckGroups(const string fileName, const string ppType,
		       const int ppId);

    std::pair<VectorXd, double>
    KSreadEq(const std::string fileName, const int Id);
    std::tuple<VectorXd, double, double>
    KSreadReq(const std::string fileName, const int Id);
    void KScheckReqGroups(const string fileName, const int Id);
    void KScheckEqGroups(const string fileName, const int Id);
    void 
    KSwriteEq(const string fileName, const int Id, 
	      const VectorXd &a, const double err);
    void 
    KSwriteReq(const string fileName, const int Id,
	       const VectorXd &a, const double omega,
	       const double err);        
    void 
    KSwriteEqE(const string fileName, const int Id, 
	       const VectorXcd e);
    void 
    KSwriteReqE(const string fileName, const int Id, 
		const VectorXcd e);
}

//////////////////////////////////////////////////
//              Implementation                  //
//////////////////////////////////////////////////

namespace MyH5 {

    /**
     * @brief obtain the data type in HDF5
     */
    template<typename Scalar>
    PredType getPredType(){
	if(typeid(Scalar) == typeid(double)){
	    return PredType::NATIVE_DOUBLE;
	}
	else if (typeid(Scalar) == typeid(int)){
	    return PredType::NATIVE_INT;
	}
	else{
	    fprintf(stderr, "getPredType received undetermined type !\n");
	    exit(1);
	}
    }

    /**
     * @brief read a scaler from a HDF5 file
     *
     * @param[in] file    H5File object. shold open as H5F_ACC_RDWR
     * @param[in] DSitem  the pass to the dataset
     * @return            the scalar
     */
    template <typename Scalar>
    Scalar readScalar(H5File &file, string DSitem){
	DataSet item = file.openDataSet(DSitem);
	DataSpace dsp = item.getSpace();
	// assert(dsp.getSimpleExtentNdims() == 0);
	// hsize_t dims[1];
	// int ndims = dsp.getSimpleExtentDims(dims, NULL);
	Scalar value(0);
	PredType type = getPredType<Scalar>();
	item.read(&value, type);
	return value;
    }

    /**
     * @brief write a scaler to a HDf5 file
     *
     * @param[in] file    H5File object. shold open as H5F_ACC_RDWR
     * @param[in] value   the scalar need to be wrote
     */
    template <typename Scalar>
    void writeScalar(H5File &file, string DSitem, Scalar value){
	hsize_t dim[0];
	DataSpace dsp(0, dim);
	PredType type = getPredType<Scalar>();
	DataSet item = file.createDataSet(DSitem, type, dsp);
	item.write(&value, type);
    }

}


#endif	/* MYH5_H */

