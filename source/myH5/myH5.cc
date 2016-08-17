#include "myH5.hpp"
#include <iostream>
#include <sstream>

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

    /**
     * @brief check the existence of groups recursively. If not exist, then create it.
     *
     * @param[in] groupName  some string like "a/b/c". Note, no '/' at the head or tail.
     */
    void checkGroup(H5File &file, const std::string groupName){
	stringstream ss(groupName);
	string item, g;
	hid_t id = file.getId();
	
	while (getline(ss, item, '/')) {
	    g += '/' + item;
	    if(H5Lexists(id, g.c_str(), H5P_DEFAULT) == false){
		file.createGroup(g.c_str());
	    }
	}
	
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


    std::pair<VectorXd, double>
    KSreadEq(const std::string fileName, const int Id){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/E/" + to_string(Id) + "/";
	return std::make_pair(readMatrixXd(file, DS + "a"),
			      readScalar<double>(file, DS + "err")
			      );
    }
    
    std::tuple<VectorXd, double, double>
    KSreadReq(const std::string fileName, const int Id){
	H5File file(fileName, H5F_ACC_RDONLY);
	string DS = "/tw/" + to_string(Id) + "/";
	return std::make_tuple(readMatrixXd(file, DS + "a"),
			       readScalar<double>(file, DS + "w"),
			       readScalar<double>(file, DS + "err")
			       );
    }
    
    
    void KScheckReqGroups(const string fileName, const int Id){
	H5File file(fileName, H5F_ACC_RDWR);
	string g1 = "/tw";
	string g2 = "/tw/" + to_string(Id);
	if ( H5Lexists(file.getId(), g1.c_str(), H5P_DEFAULT) == false ){
	    file.createGroup(g1.c_str());
	    file.createGroup(g2.c_str());
	}
	else{
	    if ( H5Lexists(file.getId(), g2.c_str(), H5P_DEFAULT) == false ){
		file.createGroup(g2.c_str());
	    }
	}
    }

    void KScheckEqGroups(const string fileName, const int Id){
	H5File file(fileName, H5F_ACC_RDWR);
	string g1 = "/E";
	string g2 = "/E/" + to_string(Id);
	if ( H5Lexists(file.getId(), g1.c_str(), H5P_DEFAULT) == false ){
	    file.createGroup(g1.c_str());
	    file.createGroup(g2.c_str());
	}
	else{
	    if ( H5Lexists(file.getId(), g2.c_str(), H5P_DEFAULT) == false ){
		file.createGroup(g2.c_str());
	    }
	}
    }

    void 
    KSwriteEq(const string fileName, const int Id, 
	      const VectorXd &a, const double err){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/E/" + to_string(Id) + "/";

	KScheckEqGroups(fileName, Id);
	
	writeMatrixXd(file, DS + "a", a);
	writeScalar<double>(file, DS + "err", err);
    }

    void 
    KSwriteReq(const string fileName, const int Id,
	       const VectorXd &a, const double omega,
	       const double err){
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/tw/" + to_string(Id) + "/";

	KScheckReqGroups(fileName, Id);
	
	writeMatrixXd(file, DS + "a", a);
	writeScalar<double>(file, DS + "w", omega);
	writeScalar<double>(file, DS + "err", err);
    }
    
    void 
    KSwriteEqE(const string fileName, const int Id, 
	       const VectorXcd e){
	
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/E/" + to_string(Id) + "/";

	KScheckEqGroups(fileName, Id);
	
	MatrixXd er(e.size(), 2);
	er << e.real(), e.imag();
	
	writeMatrixXd(file, DS + "e", er);
    }

    void 
    KSwriteReqE(const string fileName, const int Id, 
		const VectorXcd e){
	
	H5File file(fileName, H5F_ACC_RDWR);
	string DS = "/tw/" + to_string(Id) + "/";

	KScheckReqGroups(fileName, Id);
	
	MatrixXd er(e.size(), 2);
	er << e.real(), e.imag();
	
	writeMatrixXd(file, DS + "e", er);
    }

}
