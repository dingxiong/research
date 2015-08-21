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
    ///////////////         ks related          //////////////////////////
    //////////////////////////////////////////////////////////////////////

    /** @brief check the existence of Floquet vectors.
     *
     *  Some orbits with 100 < T < 120 fail to converge, so
     *  their Flqouet spectrum is not availble. The function
     *  uses exception handling, though ugly but the only method
     *  I could think of right now.
     *
     *  @param[in] ppType ppo/rpo
     *  @param[in] NN number of orbits need to be investigated
     *  @return N*2 matrix stands for exsitence of
     *          Floquet exponents and Floquet vector. '1' exist, '0' not exist.
     */
    MatrixXi checkExistEV(const string fileName, const string ppType, const int NN){
	H5File file(fileName, H5F_ACC_RDONLY);  
  
	MatrixXi status(NN, 2);
  
	for(size_t i = 0; i < NN; i++){
	    int ppId = i + 1;
	    string DS_e = "/" + ppType + "/" + to_string(ppId) + "/" + "e";
	    string DS_ve = "/" + ppType + "/" + to_string(ppId) + "/" + "ve";
	    // check the existance of eigenvalues
	    try {
		DataSet tmp = file.openDataSet(DS_e);
		status(i,0) = 1;
	    }
	    catch (FileIException not_found_error) {
		status(i, 0) = 0;
	    }
	    // check the existance of eigenvectors
	    try {
		DataSet tmp = file.openDataSet(DS_ve);
		status(i,1) = 1;
	    }
	    catch (FileIException not_found_error) {
		status(i, 1) = 0;
	    }
	}
  
	return status;
    }

    
    /** @brief read initial condition from Ruslan's file
     * 
     * @return [a, T, s]
     * @note  s = 0 for ppo
     */
    std::tuple<MatrixXd, double, double>
    KSreadOrigin(const string fileName, const string &ppType, const int ppId){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	double s = 0.0;
	if(ppType.compare("rpo") == 0) s = readScalar<double>(file, DS + "s");
	return make_tuple(readMatrixXd(file, DS + "a"),
			  readScalar<double>(file, DS + "T"),
			  s
			  );

    }
    
    /** @brief read initial conditons of KS system.
     *
     *  For ppo, s = 0.
     *
     *  @param[in] ppType periodic type: ppo/rpo
     *  @param[in] ppId  id of the orbit
     *  @return a, T, nstp, r, s 
     */
    std::tuple<MatrixXd, double, double, double, double>
    KSreadRPO(const string fileName, const string &ppType, const int ppId){
	
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";
	
	double s = 0.0;
	if(ppType.compare("rpo") == 0) s = readScalar<double>(file, DS + "s");
	
	return make_tuple(readMatrixXd(file, DS + "a"),
			  readScalar<double>(file, DS + "T"),
			  readScalar<double>(file, DS + "nstp"),
			  readScalar<double>(file, DS + "r"),
			  s
			  );
	
    }


    /** @brief rewrite the refined initial condition
     *
     *  Originally, Ruslan's file has a, T, r for ppo and a, T, r, s for rpo.
     *  Now, I refine the initial conditon, so a, T, r, (s) are updated and
     *  nstp is added.
     *
     *  @param[in] ksinit the update data in order: a, T, nstp, r, s
     */
    void 
    KSwriteRPO(const string fileName, const string ppType, const int ppId,
	       const MatrixXd &a, const double T, const double nstp,
	       const double r, const double s
	       ){
	H5File file(fileName, H5F_ACC_RDWR);
	Group group(file.createGroup("/" + ppType + to_string(ppId)));
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	writeMatrixXd(file, DS + "a", a);
	writeScalar<double>(file, DS + "T", T);
	writeScalar<double>(file, DS + "nstp", nstp);
	writeScalar<double>(file, DS + "r", r);
	writeScalar<double>(file, DS + "s", s);
    }


    /** @brief read Floquet exponents of KS system.
     *
     *  @param[in] ppType periodic type: ppo/rpo
     *  @param[in] ppId  id of the orbit
     *  @return exponents matrix
     */
    MatrixXd
    KSreadFE(const string fileName, const string ppType, const int ppId){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	return readMatrixXd(file, DS + "e"); 
    }

    /** @brief read Floquet vectors of KS system.
     *
     *  @param[in] ppType periodic type: ppo/rpo
     *  @param[in] ppId  id of the orbit
     *  @return vectors
     */
    MatrixXd
    KSreadFV(const string fileName, const string ppType, const int ppId){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	return readMatrixXd(file, DS + "ve");
    }

    /** @brief write the Floquet exponents
     *
     *  Since HDF5 uses row wise storage, so the Floquet exponents are
     *  stored in row order.
     */
    void 
    KSwriteFE(const string fileName, const string ppType, const int ppId, 
	      const MatrixXd &eigvals){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	writeMatrixXd(file, DS + "e", eigvals);
    }

    /** @brief write the Floquet exponents and Floquet vectors  */
    void 
    KSwriteFEFV(const string fileName, const string ppType, const int ppId,
		const MatrixXd &eigvals, const MatrixXd &eigvecs){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	writeMatrixXd(file, DS + "e", eigvals);
	writeMatrixXd(file, DS + "ve", eigvecs);
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
