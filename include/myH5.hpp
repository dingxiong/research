#ifndef MYH5_H
#define MYH5_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
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
    void writeScalar(H5File &file, string DSitem, Scalar value);
    
    void writeMatrixXd(H5File &file, string DSitem, const MatrixXd &mat);
    void writeVectorXd(H5File &file, string DSitem, const VectorXd &vec);
    MatrixXd readMatrixXd(H5File &file, string DSitem);
    VectorXd readVectorXd(H5File &file, string DSitem);
    
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

    VectorXd
    KSreadEq(const std::string fileName, const int Id);
    std::pair<VectorXd, double>
    KSreadReq(const std::string fileName, const int Id);
    

    //////////////////////////////////////////////////////////////////////
    /* cqcgl related */
    void CqcglWriteRPO(const string fileName, const string groupName,
		       const MatrixXd &x, const double T, const int nstp,
		       const double th, const double phi, double err);
    std::tuple<MatrixXd, double, int, double, double, double>
    CqcglReadRPO(const string fileName, const string groupName);
    void
    CqcglReadRPO(const string fileName, const string groupName,
		      MatrixXd &x, double &T, int &nstp,
		      double &th, double &phi, double &err);
    void
    CqcglMoveRPO(string infile, string ingroup, 
		 string outfile, string outgroup);
    std::string formDiGroupName(double di);
    void CqcglCheckDiExist(const string fileName, double di);
    void CqcglReadRPO(const string fileName, double di, int index,
		      MatrixXd &x, double &T, int &nstp,
		      double &th, double &phi, double &err);
    void 
    CqcglMoveRPO(string infile, string ingroup,
		 string outfile, double di, int index);
    void 
    CqcglMoveRPO(string infile, string outfile, double di, int index);
    void 
    CqcglWriteRPO(const string fileName, double di, int index,
		  const MatrixXd &x, const double T, const int nstp,
		  const double th, const double phi, double err);
    
    std::tuple<VectorXd, double, double ,double>
    CqcglReadReq(const string fileName, const string groupName);
    void 
    CqcglReadReq(const string fileName, const string groupName, 
		 VectorXd &a, double &wth, double &wphi, 
		 double &err);
    void 
    CqcglWriteReq(const string fileName, const string groupName,
		  const MatrixXd &a, const double wth, 
		  const double wphi, const double err);
    void CqcglWriteRPO2(const std::string fileName, const string groupName, 
			const MatrixXd &x, const int nstp, double err);
    void CqcglWriteRPO2(const string fileName, double di, int index,
			const MatrixXd &x, const int nstp,
			double err);
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

