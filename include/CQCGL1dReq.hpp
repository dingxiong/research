#ifndef CQCGL1dREQ_H		
#define CQCGL1dREQ_H

#include "CQCGL1d.hpp"

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
    std::tuple<VectorXd, double, double ,double>
    readReq(const std::string fileName, const std::string groupName); 
    
    static
    std::tuple<VectorXd, double, double ,double>
    readReq(const std::string fileName, 
	    const double Bi, const double Gi, int id);

    static
    void 
    writeReq(const std::string fileName, const std::string groupName,
	     const ArrayXd &a, const double wth, 
	     const double wphi, const double err);
    static
    void 
    writeReq(const std::string fileName, 
	     const double Bi, const double Gi, int id,
	     const ArrayXd &a, const double wth, const double wphi,
	     const double err);
    
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
    findReqParaSeq(const std::string file, int id, double step, int Ns, bool isBi);

    
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
