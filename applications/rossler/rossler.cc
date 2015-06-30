/*
 * g++ rossler.cc -std=c++11 -I $XDAPPS/sources/boost_1_57_0 -I $XDAPPS/eigen/include/eigen3
 */
#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;
    
class Rossler {
    
protected:
    //////////////////////////////////////////////////////////////////////
    //                              Inner classes                       //
    //////////////////////////////////////////////////////////////////////

    /**
     * @brief velocity of Rossler system
     *
     *   dx/dt = - y - z
     *   dy/dt = x + a * y
     *   dz/dt = b + z * (x - c)
     */
    struct Velocity{
	const double a, b, c;
	Velocity (double a = 0.2, double b=0.2, double c=5.7) : a(a), b(b), c(c) {}
	void operator()(const std::vector<double> &x, std::vector<double> &dxdt,
			const double /* t */){
	    dxdt[0] = -x[1] - x[2];
	    dxdt[1] = x[0] + a * x[1];
	    dxdt[2] = b + x[2] * (x[0] - c);
	}
    };

    /**
     * @brief calculate J * dx for Rossler
     *
     *               | 0, -1, -1  |
     *  Jacobian  =  | 1,  a,  0  |
     *               | z,  0, x-c |
     *
     */
    struct Jacv{
	const double a, b, c;
	Jacv (double a = 0.2, double b=0.2, double c=5.7) : a(a), b(b), c(c) {}
	void operator()(const std::vector<double> &x, std::vector<double> &dxdt,
			const double /* t */){
	    dxdt[0] = -x[1] - x[2];
	    dxdt[1] = x[0] + a * x[1];
	    dxdt[2] = b + x[2] * (x[0] - c);
	    
	    dxdt[3] = -x[4] - x[5];
	    dxdt[4] = x[3] + a*x[4];
	    dxdt[5] = x[2]*x[3] + (x[0]-c)*x[5];
	}
	
    };

    struct StoreStates{
	ArrayXXd &xx;
	ArrayXd &tt;
	int blockSize, totalNum;
	StoreStates(ArrayXd &xx, ArrayXd &tt,
		    int blockSize = 100) :
	    xx(xx), tt(tt), blockSize(blockSize), totalNum(0)
	{}
	
	void operator()(const std::vector<double> &x, double t){
	    if(totalNum >= tt.size()){
		xx.conservativeResize(NoChange, xx.cols() + blockSize);
		tt.conservativeResize(tt.size() + blockSize);
	    }
	    xx.col(totalNum) = ArrayXd::Map(&x[0], x.size());
	    tt(totalNum++) = t;
	}
    };
    
    
public:
    
    const double a, b, c;
    Velocity velocity;
    Jacv jacv;
    StoreStates storeStates;
    double odeAtol, odeRtol;

    Rossler(double odeAtol = 1e-16, double odeRtol = 1e-12,
	    double a = 0.2, double b=0.2, double c=5.7) :
	a(a), b(b), c(c),
	velocity(a, b, c),
	jacv(a, b, c),
	storeStates()
	odeAtol(odeAtol), odeRtol(odeRtol) {}
    
    
    std::pair<ArrayXXd, ArrayXd>
    intg(const Array3d &x0, const double h, const int nstp){
	const int blockSize = 100;
	int totalNum = 0;
	ArrayXXd xx(3, blockSize);
	ArrayXd tt(blockSize);
	std::vector<double> x(&x0[0], &x0[0] + 3);
	integrate_adaptive(make_controlled< runge_kutta_cash_karp54<std::vector<double>> >( 1.0e-16 , 1.0e-12 ) ,
			   // integrate(
			   velocity, x, 0.0, nstp*h, h,
			   [&xx, &tt, &totalNum, &blockSize](const std::vector<double> &x, double t){		      
			       if(totalNum >= tt.size()){
				   xx.conservativeResize(NoChange, xx.cols() + blockSize);
				   tt.conservativeResize(tt.size() + blockSize);
			       }
			       xx.col(totalNum) = ArrayXd::Map(&x[0], x.size());
			       tt(totalNum++) = t;
		      
			   });
	return make_pair(xx.leftCols(totalNum), tt.head(totalNum));
    }
    

};

int main(){
    Rossler ros;
    Array3d x0;
    x0 << 1, 6.0918, 1.2997;
    auto tmp = ros.intg(x0, 0.01, 588);
    cout << tmp.first << endl;
    return 0;
}
