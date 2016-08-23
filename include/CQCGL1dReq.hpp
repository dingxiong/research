#ifndef CQCGL1dREQ_H		
#define CQCGL1dREQ_H

#include "CQCGL1d.hpp"
#include "myH5.hpp"

using namespace H5;

class CQCGL1dReq : public CQCGL1d {

public :
    
    ////////////////////////////////////////////////////////////
    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1dReq(int N, double d,
	       double Mu, double Dr, double Di, double Br, double Bi, 
	       double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1dReq(int N, double d, 
	       double b, double c, double dr, double di, 
	       int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1dReq(int N, double d,
	       double delta, double beta, double D, double epsilon,
	       double mu, double nu, int dimTan);
    ~CQCGL1dReq();
    CQCGL1dReq & operator=(const CQCGL1dReq &x);

    ////////////////////////////////////////////////////////////
    static
    std::string 
    toStr(double Bi, double Gi, int index);

    static
    std::tuple<VectorXd, double, double ,double>
    readReq(H5File &file, const std::string groupName); 
    
    static
    void 
    writeReq(H5File &file, const std::string groupName,
	     const ArrayXd &a, const double wth, 
	     const double wphi, const double err);
    static
    VectorXcd 
    readE(H5File &file, const std::string groupName);

    static
    MatrixXcd 
    readV(H5File &file, const std::string groupName);
    
    static
    void 
    writeE(H5File &file, const std::string groupName, 
	   const VectorXcd e);
    
    static
    void 
    writeV(H5File &file, const std::string groupName, 
	   const MatrixXcd v);
    
    static
    void 
    moveReq(H5File &fin, std::string gin, H5File &fout, std::string gout,
	    int flag = 0);
    
    ////////////////////////////////////////////////////////////
    VectorXd Fx(const VectorXd &x);
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    calJJF(const VectorXd &x);
    std::tuple<VectorXd, double, double, double, int>
    findReq_LM(const ArrayXd &a0, const double wth0, const double wphi0, 
	       const double tol,
	       const int maxit,
	       const int innerMaxit);
    std::vector<double>
    optThPhi(const ArrayXd &a0);
    void 
    findReqParaSeq(H5File &file, int id, double step, int Ns, bool isBi);
    void 
    calEVParaSeq(H5File &file, std::vector<int> ids, std::vector<double> Bis,
		 std::vector<double> Gis, bool saveV);
    
    
};

template<class Mat>
struct CQCGL1dReqJJF {
    
    CQCGL1dReq *req;
    
    CQCGL1dReqJJF(CQCGL1dReq *req) : req(req){}
    
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    operator()(const VectorXd &x) {
	return req->calJJF(x);
    }	
};

#endif	/* CQCGL1dREQ_H */
