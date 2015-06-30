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
	int blockSize;
	static int totalNum;
	StoreStates(ArrayXXd &xx, ArrayXd &tt,
		    int blockSize = 100) :
	    xx(xx), tt(tt), blockSize(blockSize)
	{ cout << totalNum << endl;}
	
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
    double odeAtol, odeRtol;


    Rossler(double odeAtol = 1e-16, double odeRtol = 1e-12,
	    double a = 0.2, double b=0.2, double c=5.7) :
	a(a), b(b), c(c),
	velocity(a, b, c),
	jacv(a, b, c),
	odeAtol(odeAtol), odeRtol(odeRtol)
    {}
    
    template< typename Vel>
    std::pair<ArrayXXd, ArrayXd>
    intg0(const ArrayXd &x0, const double h, const int nstp, Vel vel, int blockSize = 100){
	ArrayXXd xx(x0.size(), blockSize);
	ArrayXd tt(blockSize);
	int totalNum = 0;
	std::vector<double> x(&x0[0], &x0[0] + x0.size());
	integrate_adaptive(make_controlled< runge_kutta_cash_karp54<std::vector<double>> >( odeAtol , odeRtol ) ,
			   // integrate(
			   vel, x, 0.0, nstp*h, h,
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

    std::pair<ArrayXXd, ArrayXd>
    intg(const ArrayXd &x0, const double h, const int nstp, int blockSize = 100){
	return intg0(x0, h, nstp, velocity, blockSize);
    }

    std::tuple<ArrayXXd, ArrayXXd, ArrayXd>
    intgj(const ArrayXd &x0, const ArrayXd &dx0, const double h,
	  const int nstp, int blockSize = 100){
	ArrayXd xdx(x0.size() + dx0.size());
	xdx << x0, dx0;
	auto tmp = intg0(xdx, h, nstp, jacv, blockSize);
	return std::make_tuple(tmp.first.topRows(3),
			       tmp.first.bottomRows(3),
			       tmp.second);
    }
    

};


int main(){
    Rossler ros;
    Array3d x0;
    x0 << 1, 6.0918, 1.2997;
    Array3d dx0;
    dx0 << 0.01, 0.01, 0.01;
    auto tmp = ros.intgj(x0, dx0, 0.01, 588);
    cout << std::get<1>(tmp) << endl;
    return 0;
}
