#include "myH5.hpp"

namespace MyH5 {

    /**
     * @brief read a double matrix
     */
    MatrixXd readMatrixXd(H5File &file, string DSitem){
	DataSet item = file.openDataSet(DSitem);
	DataSpace dsp = item.getSpace();
	assert(dsp.getSimpleExtentNdims() == 2);
	hsize_t dims[2];
	int ndims = dsp.getSimpleExtentDims(dims, NULL);
	MatrixXd x(dims[1], dims[0]);	/* HDF5 uses row major by default */
	item.read(x.data(), PredType::NATIVE_DOUBLE);
	return x;
    }
    
    /**
     * @brief write a double matrix
     */
    void writeMatrixXd(H5File &file, string DSitem, const MatrixXd &mat){
	const int N = mat.rows();
	const int M = mat.cols();

	hsize_t dim[] = {M, N};	/* HDF5 uses row major by default */
	DataSpace dsp(2, dim);
	DataSet item = file.createDataSet(DSitem, PredType::NATIVE_DOUBLE, dsp);
	item.write(mat.data(), PredType::NATIVE_DOUBLE);
    }

    
    //////////////////////////////////////////////////////////////////////
    ///////////////       cqcgl related         //////////////////////////
    //////////////////////////////////////////////////////////////////////
    
    /**
     * @note group should be a new group
     * [x, T,  nstp, theta, phi, err]
     */
    void CqcglWriteRPO(const string fileName, const string groupName,
		  const MatrixXd &x, const double T, const int nstp,
		  const double th, const double phi, double err){

	H5File file(fileName, H5F_ACC_RDWR);
	Group group(file.createGroup("/"+groupName));
	string DS = "/" + groupName + "/";

	writeMatrixXd(file, DS + "x", x);
	writeScalar<double>(file, DS + "T", T);
	writeScalar<int>(file, DS + "nstp", nstp);
	writeScalar<double>(file, DS + "th", th);
	writeScalar<double>(file, DS + "phi", phi);
	writeScalar<double>(file, DS + "err", err);
    }


    /* @note dateset should exist */
    std::tuple<MatrixXd, double, int, double, double, double>
    CqcglReadRPO(const string fileName, const string groupName){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + groupName + "/";
    
	return make_tuple(readMatrixXd(file, DS + "x"),
			  readScalar<double>(file, DS + "T"),
			  readScalar<int>(file, DS + "nstp"),
			  readScalar<double>(file, DS + "th"),
			  readScalar<double>(file, DS + "phi"),
			  readScalar<double>(file, DS + "err")
			  );
    
    }

}
