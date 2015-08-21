#ifndef MYH5_H
#define MYH5_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include "H5Cpp.h"

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
    MatrixXd readMatrixXd(H5File &file, string DSitem);

    //////////////////////////////////////////////////////////////////////
    /* KS related */
    MatrixXi
    checkExistEV(const string fileName, const string ppType, const int NN);
    std::tuple<MatrixXd, double, double>
    KSreadOrigin(const string fileName, const string &ppType, const int ppId);
    std::tuple<MatrixXd, double, double, double, double>
    KSreadRPO(const string fileName, const string &ppType, const int ppId);
    void 
    KSwriteRPO(const string fileName, const string ppType, const int ppId,
	       const MatrixXd &a, const double T, const double nstp,
	       const double r, const double s
	       );

    MatrixXd
    KSreadFE(const string fileName, const string ppType, const int ppId);
    MatrixXd
    KSreadFV(const string fileName, const string ppType, const int ppId);
    void 
    KSwriteFE(const string fileName, const string ppType, const int ppId, 
	      const MatrixXd &eigvals);
    void 
    KSwriteFEFV(const string fileName, const string ppType, const int ppId,
		const MatrixXd &eigvals, const MatrixXd &eigvecs);

    //////////////////////////////////////////////////////////////////////
    /* cqcgl related */
    void CqcglWriteRPO(const string fileName, const string groupName,
		       const MatrixXd &x, const double T, const int nstp,
		       const double th, const double phi, double err);
    std::tuple<MatrixXd, double, int, double, double, double>
    CqcglReadRPO(const string fileName, const string groupName);
    
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
	// assert(dsp.getSimpleExtentNdims() == 1);
	hsize_t dims[1];
	int ndims = dsp.getSimpleExtentDims(dims, NULL);
	// assert(dims[0] == 1);
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
	hsize_t dim[] = {1};
	DataSpace dsp(1, dim);
	PredType type = getPredType<Scalar>();
	DataSet item = file.createDataSet(DSitem, type, dsp);
	item.write(&value, type);
    }

}


#endif	/* MYH5_H */

