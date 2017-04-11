    std::pair<ArrayXXd, ArrayXXd>    
    intgjMulti(const MatrixXd aa0, size_t nstp, size_t np, size_t nqr);
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    calReqJJF(const Ref<const VectorXd> &x);
    std::tuple<VectorXd, double, double>
    findReq(const Ref<const VectorXd> &x, const double tol, 
	    const int maxit, const int innerMaxit);
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    calEqJJF(const Ref<const VectorXd> &x);
    std::pair<VectorXd, double>
    findEq(const Ref<const VectorXd> &x, const double tol,
	   const int maxit, const int innerMaxit);



/*============================================================
 *                       Class : Calculate KS Req LM 
 *============================================================*/
template<class Mat>
struct KSReqJJF {
    KS &ks;
    KSReqJJF(KS &ks_) : ks(ks_){}
    
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    operator()(const VectorXd &x) {
	return ks.calReqJJF(x);
    }	
};

/*============================================================
 *                       Class : Calculate KS Eq LM 
 *============================================================*/
template<class Mat>
struct KSEqJJF {
    KS &ks;
    KSEqJJF(KS &ks_) : ks(ks_){}
    
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    operator()(const VectorXd &x) {
	return ks.calEqJJF(x);
    }	
};

