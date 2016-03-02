#include "iterMethod.hpp"

namespace iterMethod {

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // global variables
    
    bool CG_PRINT = true;
    int CG_PRINT_FREQUENCE = 100;

    bool GMRES_OUT_PRINT = true;
    int GMRES_OUT_PRINT_FREQUENCE = 1;

    bool GMRES_IN_PRINT = true;
    int GMRES_IN_PRINT_FREQUENCE = 1;

    bool HOOK_PRINT = true;
    int HOOK_PRINT_FREQUENCE = 1;

    bool INB_OUT_PRINT = true;
    int INB_OUT_PRINT_FREQUENCE = 1;

    bool INB_IN_PRINT = true;
    int INB_IN_PRINT_FREQUENCE = 1;

    bool LM_OUT_PRINT = true;
    int LM_OUT_PRINT_FREQUENCE = 1;

    bool LM_IN_PRINT = true;
    int LM_IN_PRINT_FREQUENCE = 1;

    double LM_LAMBDA_INCREASE_FACTOR = 10;
    double LM_LAMBDA_DECREASE_FACTOR = 10;
    double LM_MAX_LAMBDA = 1e10;
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////




    /**
     * @brief obtain the cos and sin from coordinate (x, y)
     */
    void rotmat(const double &x, const double &y,
		double *c, double *s){
	if(y == 0){
	    *c = 1;
	    *s = 0;
	}
	else if ( fabs(y) > fabs(x) ){
	    double tmp = x / y;
	    *s = 1.0 / sqrt(1.0 + tmp*tmp);
	    *c = tmp * (*s);
	}
	else {
	    double tmp = y / x;
	    *c = 1.0 / sqrt(1.0 + tmp*tmp);
	    *s = tmp * (*c);
	}
    }


    /**
     * @brief choose the scaling parameter theta in the Inexact Newton Backtracking algorithm
     *
     * When choosing theta, we take a quadratic model
     *      g(theta) = ||F(x + theta * s)|| = (g(1) - g(0) - g'(0))*theta^2 + g'(0)*theta + g(0)
     * with
     *      g'(0) = 2 * F^{T}(x) * F'(x) * s
     * The derivatives are:
     *      g'(theta) = 2 * (g(1) - g(0) - g'(0))*theta + g'(0)
     *      g''(theta) = 2 * (g(1) - g(0) - g'(0))
     *
     * If g''(0) < 0, take theta = theta_max
     * If g''(0) > 0, take theta =  - g'(0) /  2 * (g(1) - g(0) - g'(0))
     */
    double chooseTheta(const double g0, const double g1, const double gp0,
		       const double theta_min, const double theta_max){
	double theta;
	double gpp0 = 2 * (g1 - g0 - gp0);
	if (gpp0 < 0) theta = theta_max;
	else theta = -gp0 / gpp0;
	
	// restrict theta in [theta_min, theta_max]
	return theta > theta_max ? theta_max : (theta < theta_min ? theta_min : theta);
    }


    //////////////////////////////////////////////////////////////////////
    ///////////////////   trust region GMRES related /////////////////////
    //////////////////////////////////////////////////////////////////////
    /**
     * @brief calculate z = d_i * p_i / (d_i^2 + mu)
     */
    ArrayXd calz(const ArrayXd &D, const ArrayXd &p, const double mu){
	return D * p / (D.square() + mu);
    }

    /**
     * @brief use newton method to find the parameter mu which minize
     *        ||D*z - p|| w.r.t. ||z||< delta
     */
    std::tuple<double, std::vector<double>, int>
    findTrustRegion(const ArrayXd &D, const ArrayXd &p, double delta,
		    const double tol, const int maxit,
		    const double mu0){
	double mu = mu0;
	std::vector<double> errVec;

	// first check whether mu = 0 is enough
	if( (p / D).matrix().norm() <= delta )
	    return std::make_tuple(0, errVec, 0);
	
	// use Newton iteration to find mu
	for(size_t i = 0; i < maxit; i++){
	    ArrayXd denorm = D.square() + mu;	  /* di^2 + mu */
	    ArrayXd z = D * p / denorm;		  /* z  */
	    double z2 = z.matrix().squaredNorm(); /* ||z||^2 */
	    double phi = z2 - delta * delta;	  /* phi(mu) */
		
	    errVec.push_back(phi);
	    if(fabs(phi) < tol) return std::make_tuple(mu, errVec, 0);
		
	    double dphi = (-2*(D * p).square() / denorm * denorm * denorm).sum();
	    mu -= (sqrt(z2) / delta) * ( phi / dphi );
	    // mu -=  phi / dphi ;  
	}

	// run out of loop => not converged
	return std::make_tuple(mu, errVec, 1);
    }

    //////////////////////////////////////////////////////////////////////
    //            Levenberg-Marquardt related                           //
    //////////////////////////////////////////////////////////////////////
    
    /*
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    JJF(const MatrixXd &J, const VectorXd &x){
	MatrixXd JJ = J.transpose() * J;
	MatrixXd d = JJ.diagonal().asDiagonal();
	return std::make_tuple(JJ, d, J*x);
    }

    */
    
    std::tuple<VectorXd, std::vector<double>, int>
    LM( const MatrixXd &A, const VectorXd &b,
	const VectorXd &x0, const double tol,
	const int maxit, const int innerMaxit){
	
	ColPivHouseholderQR<MatrixXd> solver;
	auto fx = [&A, &b](const VectorXd &x) { return A*x - b; };
	JJF<MatrixXd> jj(A, b); 
	return LM0( fx, jj, solver, x0, tol, maxit, innerMaxit );
    }

}
