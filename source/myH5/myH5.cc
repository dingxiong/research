#include "myH5.hpp"
#include <iostream>

namespace MyH5 {

    /**
     * @brief read a double matrix
     */
    MatrixXd readMatrixXd(H5File &file, string DSitem){
	DataSet item = file.openDataSet(DSitem);
	DataSpace dsp = item.getSpace();
	const int D = dsp.getSimpleExtentNdims();
	if (D == 2) {
	    hsize_t dims[2];
	    int ndims = dsp.getSimpleExtentDims(dims, NULL);
	    MatrixXd x(dims[1], dims[0]);	/* HDF5 uses row major by default */
	    item.read(x.data(), PredType::NATIVE_DOUBLE);
	    return x;
	}
	else if (D == 1) {
	    hsize_t dims[1];
	    int ndims = dsp.getSimpleExtentDims(dims, NULL);
	    MatrixXd x(dims[0], 1);
	    item.read(x.data(), PredType::NATIVE_DOUBLE);
	    return x;
	}
	else {
	    fprintf(stderr, "readMatrixXd() dimension wrong !\n");
	    exit(-1);
	}
    }
    
    /**
     * @brief read a double vector
     */
    VectorXd readVectorXd(H5File &file, string DSitem){
	DataSet item = file.openDataSet(DSitem);
	DataSpace dsp = item.getSpace();
	const int D = dsp.getSimpleExtentNdims();
	assert ( D == 1);
	
	hsize_t dims[1];
	int ndims = dsp.getSimpleExtentDims(dims, NULL);
	VectorXd x(dims[0]);
	item.read(x.data(), PredType::NATIVE_DOUBLE);

	return x;
    }

    /**
     * @brief write a double matrix
     */
    void writeMatrixXd(H5File &file, string DSitem, const MatrixXd &mat){
	const int N = mat.rows();
	const int M = mat.cols();
	
	if ( 1 == M){
	    hsize_t dim[] = {N};
	    DataSpace dsp(1, dim);
	    DataSet item = file.createDataSet(DSitem, PredType::NATIVE_DOUBLE, dsp);
	    item.write(mat.data(), PredType::NATIVE_DOUBLE);
	}
	else {
	    hsize_t dim[] = {M, N};	/* HDF5 uses row major by default */
	    DataSpace dsp(2, dim);
	    DataSet item = file.createDataSet(DSitem, PredType::NATIVE_DOUBLE, dsp);
	    item.write(mat.data(), PredType::NATIVE_DOUBLE);
	}
	
    }

    /**
     * @brief write a double vector
     */
    void writeVectorXd(H5File &file, string DSitem, const VectorXd &vec){
	const int N = vec.size();
	
	hsize_t dim[] = {N};
	DataSpace dsp(1, dim);
	DataSet item = file.createDataSet(DSitem, PredType::NATIVE_DOUBLE, dsp);
	item.write(vec.data(), PredType::NATIVE_DOUBLE);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////         ks related          ////////////////////////////////////////////////////////////////////////////
    
    /**
     * check the existence of groups. If not, then crate it.
     */
    void KScheckGroups(const string fileName, const string ppType,
		       const int ppId){
	
	H5File file(fileName, H5F_ACC_RDWR);
	string g1 =  "/" + ppType;
	string g2 =  "/" + ppType + "/" + to_string(ppId);
	if(H5Lexists(file.getId(), g1.c_str(), H5P_DEFAULT) == false){
	    file.createGroup(g1.c_str());
	    file.createGroup(g2.c_str());	    
	}
	else{
	    if(H5Lexists(file.getId(), g2.c_str(), H5P_DEFAULT) == false)
		file.createGroup(g2.c_str());	 
	}
    }

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
    MatrixXi checkExistEV(const string fileName, const string ppType, const std::vector<int> ppIds){
	H5File file(fileName, H5F_ACC_RDONLY);  
  
	MatrixXi status(ppIds.size(), 2);
  
	for(size_t i = 0; i < ppIds.size(); i++){
	    int ppId = ppIds[i];
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

    MatrixXi checkExistEV(const string fileName, const string ppType, const int NN){

	std::vector<int> ppIds(NN);
	for(int i = 0; i < NN; i++) ppIds[i] = i+1;
	return checkExistEV(fileName, ppType, ppIds);
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
    std::tuple<MatrixXd, double, int, double, double>
    KSreadRPO(const string fileName, const string &ppType, const int ppId){
	
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";
	
	double s = 0.0;
	if(ppType.compare("rpo") == 0) s = readScalar<double>(file, DS + "s");
	
	return make_tuple(readMatrixXd(file, DS + "a"),
			  readScalar<double>(file, DS + "T"),
			  readScalar<int>(file, DS + "nstp"),
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
	       const MatrixXd &a, const double T, const int nstp,
	       const double r, const double s
	       ){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";
	
	KScheckGroups(fileName, ppType, ppId);
	
	writeMatrixXd(file, DS + "a", a);
	writeScalar<double>(file, DS + "T", T);
	writeScalar<int>(file, DS + "nstp", nstp);
	writeScalar<double>(file, DS + "r", r);
	if(ppType.compare("rpo") == 0) 
	    writeScalar<double>(file, DS + "s", s);
    }
    
    void 
    KSmoveRPO(const std::string inFile, const std::string outFile, const std::string ppType, 
	      const int ppId){
	auto tmp = KSreadRPO(inFile, ppType, ppId);
	KSwriteRPO(outFile, ppType, ppId, std::get<0>(tmp), std::get<1>(tmp), std::get<2>(tmp),
		   std::get<3>(tmp), std::get<4>(tmp));
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
	
	KScheckGroups(fileName, ppType, ppId);
	
	writeMatrixXd(file, DS + "e", eigvals);
    }

    void 
    KSmoveFE(const std::string inFile, const std::string outFile, const std::string ppType,
	     const int ppId){
	auto tmp = KSreadFE(inFile, ppType, ppId);
	KSwriteFE(outFile, ppType, ppId, tmp);
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


    /** @brief write FV into file  */
    void
    KSwriteFV(const string fileName, const string ppType, const int ppId, 
	      const MatrixXd &eigvecs){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";
	
	KScheckGroups(fileName, ppType, ppId);
	
	writeMatrixXd(file, DS + "ve", eigvecs);
    }
    
    void 
    KSmoveFV(const std::string inFile, const std::string outFile, const std::string ppType,
	     const int ppId){
	auto tmp = KSreadFV(inFile, ppType, ppId);
	KSwriteFV(outFile, ppType, ppId, tmp);
    }

    std::pair<MatrixXd, MatrixXd>
    KSreadFEFV(const string fileName, const string ppType, const int ppId){
		H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	return std::make_pair(readMatrixXd(file, DS + "e"),
			      readMatrixXd(file, DS + "ve")
			      ); 
    }

    /** @brief write the Floquet exponents and Floquet vectors  */
    void 
    KSwriteFEFV(const string fileName, const string ppType, const int ppId,
		const MatrixXd &eigvals, const MatrixXd &eigvecs){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/" + ppType + "/" + to_string(ppId) + "/";

	KScheckGroups(fileName, ppType, ppId);
	
	writeMatrixXd(file, DS + "e", eigvals);
	writeMatrixXd(file, DS + "ve", eigvecs);
    }
    
    void 
    KSmoveFEFV(const std::string inFile, const std::string outFile, const std::string ppType,
	       const int ppId){
	auto tmp = KSreadFEFV(inFile, ppType, ppId);
	KSwriteFEFV(outFile, ppType, ppId, tmp.first, tmp.second);
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////       cqcgl related         ////////////////////////////////////////////////////////////////////////////

    
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

    void CqcglWriteRPO2(const std::string fileName, const string groupName, 
			const MatrixXd &x, const int nstp, double err){
	H5File file(fileName, H5F_ACC_RDWR);
	Group group(file.createGroup("/"+groupName));
	string DS = "/" + groupName + "/";
	
	MatrixXd tmp = x.bottomRows(3).rowwise().sum();
	double T = tmp(0);
	double th = tmp(1);
	double phi = tmp(2);
	
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

    /**
     * @brief read rpo info from hdf5 file for cqcgl
     *
     *  This is a short version
     */
    void CqcglReadRPO(const string fileName, const string groupName,
		      MatrixXd &x, double &T, int &nstp,
		      double &th, double &phi, double &err){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + groupName + "/";
	
	x = readMatrixXd(file, DS + "x");
	T = readScalar<double>(file, DS + "T");
	nstp =  readScalar<int>(file, DS + "nstp");
	th = readScalar<double>(file, DS + "th");
	phi = readScalar<double>(file, DS + "phi");
	err = readScalar<double>(file, DS + "err");
    }

    /**
     * @brief move rpo from one file, group to another file, group
     */
    void
    CqcglMoveRPO(string infile, string ingroup, 
		 string outfile, string outgroup){
	auto result = CqcglReadRPO(infile, ingroup);
	CqcglWriteRPO(outfile, outgroup,
		      std::get<0>(result), /* x */
		      std::get<1>(result), /* T */
		      std::get<2>(result), /* nstp */
		      std::get<3>(result), /* th */
		      std::get<4>(result), /* phi */
		      std::get<5>(result)  /* err */
		      );

    }

    /*------------------------------------------------------------  */
    /* The following are overloading functions specific to di groups */
    /*------------------------------------------------------------  */

    /**
     * form the group name with di
     */
    std::string formDiGroupName(double di){
	char buffer [20];
	int cx = snprintf ( buffer, 20, "%.6f", di);
	assert(cx > 6);
	string groupName = "/" + std::string(buffer);
	
	return groupName;
    }
    
    /**
     * check the existence of group. If not, then crate it.
     */
    void CqcglCheckDiExist(const string fileName, double di){
	
	H5File file(fileName, H5F_ACC_RDWR);
	string groupName = formDiGroupName(di);
	if(H5Lexists(file.getId(), groupName.c_str(), H5P_DEFAULT) == false)
	    file.createGroup(groupName.c_str());
    }
    
    /**
     * @brief read rpo info from hdf5 file for cqcgl
     *
     *  This is specific to di group.
     *  This is a short version.
     */
    void CqcglReadRPO(const string fileName, double di, int index,
		      MatrixXd &x, double &T, int &nstp,
		      double &th, double &phi, double &err){
	string groupName = formDiGroupName(di) + "/" + std::to_string(index);	
	CqcglReadRPO(fileName, groupName, x, T, nstp, th, phi, err);
    }


    /**
     * @brief move rpo from one file, group to another file, group
     *
     *  The output group is formed as '/di/index'
     *
     *  @see CqcglMoveRPO()
     */
    void CqcglMoveRPO(string infile, string ingroup,
		      string outfile, double di, int index){
	CqcglCheckDiExist(outfile, di);
	
	string outgroup = formDiGroupName(di) + "/" + std::to_string(index);
	CqcglMoveRPO(infile, ingroup, outfile, outgroup);
    }
   
    /**
     * @brief move rpo from one file, group to another file, group
     *
     *  The input and output group are all '/di/index'
     *
     *  @see CqcglMoveRPO()
     */
    void CqcglMoveRPO(string infile, string outfile, double di, int index){
	CqcglCheckDiExist(outfile, di);
	string groupName = formDiGroupName(di) + "/" + std::to_string(index);
	CqcglMoveRPO(infile, groupName, outfile, groupName); 
    }


    /**
     * @brief The output group is formed as '/di/index'
     * 
     * @see CqcglWriteRPO(), formDiGroupName()
     */
    void CqcglWriteRPO(const string fileName, double di, int index,
		       const MatrixXd &x, const double T, const int nstp,
		       const double th, const double phi, double err){
	CqcglCheckDiExist(fileName, di);
	
	string groupName = formDiGroupName(di) + "/" + std::to_string(index);
	CqcglWriteRPO(fileName, groupName, x, T, nstp, th, phi, err);
    }

    void CqcglWriteRPO2(const string fileName, double di, int index,
			const MatrixXd &x, const int nstp,
			double err){
	CqcglCheckDiExist(fileName, di);
	
	string groupName = formDiGroupName(di) + "/" + std::to_string(index);
	CqcglWriteRPO2(fileName, groupName, x, nstp, err);
    }
    

    /**** -------------------------------------------------- ****/
    
    /**
     * @brief read req (relative equibrium) info from hdf5 file for cqcgl
     *
     * Note, the return a is a vector not a matrix.
     * 
     * @see the short version
     */
    std::tuple<VectorXd, double, double ,double>
    CqcglReadReq(const string fileName, const string groupName){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + groupName + "/";
	
	return make_tuple(readMatrixXd(file, DS + "a").col(0),
			  readScalar<double>(file, DS + "wth"),
			  readScalar<double>(file, DS + "wphi"),
			  readScalar<double>(file, DS + "err")
			  );
    }


    /**
     * @brief read req (relative equibrium) info from hdf5 file for cqcgl
     *
     *  This is a short version
     */
    void 
    CqcglReadReq(const string fileName, const string groupName, 
		 VectorXd &a, double &wth, double &wphi, 
		 double &err){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/" + groupName + "/";
	
	a = readMatrixXd(file, DS + "a").col(0);
	wth = readScalar<double>(file, DS + "wth");
	wphi = readScalar<double>(file, DS + "wphi");
	err = readScalar<double>(file, DS + "err");

    }
    

    /**
     * @brief write [a, wth, wphi, err] of Req of cqcgl into a group
     * 
     * @note group should be a new group
     */
    void 
    CqcglWriteReq(const string fileName, const string groupName,
		  const MatrixXd &a, const double wth, 
		  const double wphi, const double err){
	
	H5File file(fileName, H5F_ACC_RDWR);
	Group group(file.createGroup("/"+groupName));
	string DS = "/" + groupName + "/";
	
	writeMatrixXd(file, DS + "a", a);
	writeScalar<double>(file, DS + "wth", wth);
	writeScalar<double>(file, DS + "wphi", wphi);
	writeScalar<double>(file, DS + "err", err);
    }
    

}
