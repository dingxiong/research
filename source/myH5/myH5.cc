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
     * @brief check the existence of groups recursively. 
     *
     * If doCreate = false, it will immediately return if not exist
     * If doCreate = true. It will finish the loop 
     * 
     * @param[in] groupName  some string like "a/b/c". Note, no '/' at the head or tail.
     * @param[in] doCreate   if not exist, whether create the group
     */
    bool checkGroup(H5File &file, const std::string groupName, const bool doCreate){
	stringstream ss(groupName);
	string item, g;
	hid_t id = file.getId();
	
	bool exist = true;
	
	while (getline(ss, item, '/')) {
	    g += '/' + item;
	    if(H5Lexists(id, g.c_str(), H5P_DEFAULT) == false){
		exist = false;
		if (doCreate) file.createGroup(g.c_str());
		else return exist;
	    }
	}
    
	return exist;
    }

    bool checkGroup(std::string fileName, const std::string groupName, const bool doCreate){
	H5File file(fileName, H5F_ACC_RDWR);
	return checkGroup(file, groupName, doCreate);
    }
    
    // obtain all unique groupnames in a file
    // if file has datasets /a/b/c/d1, /a/b/c/d2, /a/b/d3
    // the it outputs a/b/c, a/b
    // from https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/h5_info.c
    vector<vector<string>> scanGroup(std::string fileName){
	H5File file(fileName, H5F_ACC_RDONLY);
	unordered_set<string> record;
	vector<vector<string>> result;
	vector<string> curt;
	scanGroupHelp(file.getId(), result, record, curt);
	
	return result;
    }
    
    // I do not know why H5Gopen does not work but H5Gopen1 works
    void scanGroupHelp(hid_t gid, vector<vector<string>> &result, unordered_set<string> &record,
		       vector<string> &curt) {
	int MAX_NAME = 100;
	char memb_name[MAX_NAME];
	hsize_t nobj;
	
	herr_t err = H5Gget_num_objs(gid, &nobj);
	for (int i = 0; i < nobj; i++) {
	    int len = H5Gget_objname_by_idx(gid, (hsize_t)i, memb_name, (size_t)MAX_NAME );
	    int otype =  H5Gget_objtype_by_idx(gid, (size_t)i );
	    switch(otype) {
	    case H5G_GROUP: {
		hid_t grpid = H5Gopen1(gid, memb_name);
		curt.push_back(string(memb_name));
		scanGroupHelp(grpid, result, record, curt);
		curt.pop_back();
		H5Gclose(grpid);
		break;
	    }
	    case H5G_DATASET: {
		string groupName;
		for(auto s : curt) groupName += s + "/";
		groupName.pop_back();
		if (record.find(groupName) == record.end()){
		    record.insert(groupName);
		    result.push_back(curt);
		}
		break;
	    }
	    default:
		fprintf(stderr, "scanGroup unknown? \n");
	    }
	}
    }


    ////////////////////////////////////////////////////////////////////////////////
    //                             KS related
    ////////////////////////////////////////////////////////////////////////////////

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

   

}
