/* try to form the Jacobian for finding the relative equilibirum
 *
 *      | A+wT , tx|
 *  J = | tx   ,  0|
 *
 *  @param[in] x  [a0, omega]
 *  return J^T*J, diag(J^T*J), J^T * F
 */
std::tuple<MatrixXd, MatrixXd, VectorXd>
KS::calReqJJF(const Ref<const VectorXd> &x){
    assert(x.size() == N-1);
    double omega = x(N-2); 

    MatrixXd A = stabReq(x.head(N-2), omega); 
    VectorXd tx = gTangent(x.head(N-2)); 
    
    MatrixXd J(N-1, N-1);
    J << 
	A, tx, 
	tx.transpose(), 0; 

    VectorXd F(N-1);
    F << velg(x.head(N-2), omega), 0 ;

    MatrixXd JJ = J.transpose() * J; 
    MatrixXd DJJ = JJ.diagonal().asDiagonal(); 
    VectorXd JF = J.transpose() * F; 
    
    return std::make_tuple(JJ, DJJ, JF); 
}
/**
 * @see calReqJJF
 */
std::tuple<MatrixXd, MatrixXd, VectorXd>
KS::calEqJJF(const Ref<const VectorXd> &x){
    assert(x.size() == N-2);
    
    MatrixXd J = stab(x);
    VectorXd F = velocity(x);

    MatrixXd JJ  = J.transpose() * J;
    MatrixXd DJJ = JJ.diagonal().asDiagonal(); 
    VectorXd JF = J.transpose() * F; 
    
    return std::make_tuple(JJ, DJJ, JF);
}

/* find reqs in KS  */
std::tuple<VectorXd, double, double>
KS::findReq(const Ref<const VectorXd> &x, const double tol, 
	    const int maxit, const int innerMaxit){

    auto fx = [&](const VectorXd &x){
	VectorXd F(N-1);
	F << velg(x.head(N-2), x(N-2)), 0; 
	return F;
    };
    
    KSReqJJF<MatrixXd> jj(*this);    
    ColPivHouseholderQR<MatrixXd> solver; 

    auto result = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    if(std::get<2>(result) != 0) fprintf(stderr, "Req not converged ! \n");
    
    VectorXd at = std::get<0>(result);
    return std::make_tuple(at.head(N-2), at(N-2) , std::get<1>(result).back() );
}

/* find eq in KS */
std::pair<VectorXd, double>
KS::findEq(const Ref<const VectorXd> &x, const double tol,
	   const int maxit, const int innerMaxit){
    
    auto fx = [&](const VectorXd &x){
	return velocity(x);
    };
    
    KSEqJJF<MatrixXd> jj(*this);
    ColPivHouseholderQR<MatrixXd> solver;
    
    auto result = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);
    if(std::get<2>(result) != 0) fprintf(stderr, "Req not converged ! \n");
    
    return std::make_pair(std::get<0>(result), std::get<1>(result).back() );   
}
