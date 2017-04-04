#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL1dEIDc.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL1dEIDc : public CQCGL1dEIDc {
  
public:

    pyCQCGL1dEIDc(int N, double d, double Mu, double Dr, double Di, 
		  double Br, double Bi, double Gr, double Gi):
	CQCGL1dEIDc(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi) {}

    pyCQCGL1dEIDc(int N, double d,
		  double b, double c, double dr, double di) :
	CQCGL1dEIDc(N, d, b, c, dr, di) {}
    
    bn::ndarray PYTs(){
	return copy2bn(Ts);
    }

    bn::ndarray PYlte(){
	return copy2bn(lte);
    }

    bn::ndarray PYhs(){
	return copy2bn(hs);
    }

   /* wrap the integrator */
    bn::ndarray PYintgC(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intgC(tmpa, h, tend, skip_rate));
    }
    
    bn::ndarray PYintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, h, tend, skip_rate));
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL1dEIDc) {
    bn::initialize();
    
    // must provide the constructor
    bp::class_<CQCGL1dEIDc>("CQCGL1dEIDc", bp::init<
			    int, double, double, double, double,
			    double, double, double, double>())
	;
    
    // the EIDc class
    bp::class_<EIDc>("EIDc", bp::init<>())
	.def_readonly("NCalCoe", &EIDc::NCalCoe)
	.def_readonly("NReject", &EIDc::NReject)
	.def_readonly("NCallF", &EIDc::NCallF)
	.def_readonly("NSteps", &EIDc::NSteps)
	.def_readonly("TotalTime", &EIDc::TotalTime)
	.def_readonly("CoefficientTime", &EIDc::CoefficientTime)
	.def_readwrite("rtol", &EIDc::rtol)
	;

    // 
    bp::class_<pyCQCGL1dEIDc, bp::bases<CQCGL1dEIDc> >("pyCQCGL1dEIDc", bp::init<
						       int, double, double, double, double,
						       double, double, double, double>())
	.def(bp::init<int, double, double, double ,double, double>())
	.def_readonly("N", &pyCQCGL1dEIDc::N)
	.def_readonly("d", &pyCQCGL1dEIDc::d)
	.def_readonly("Mu", &pyCQCGL1dEIDc::Mu)
	.def_readwrite("Br", &pyCQCGL1dEIDc::Br)
	.def_readwrite("Bi", &pyCQCGL1dEIDc::Bi)
	.def_readonly("Dr", &pyCQCGL1dEIDc::Dr)
	.def_readonly("Di", &pyCQCGL1dEIDc::Di)
	.def_readwrite("Gr", &pyCQCGL1dEIDc::Gr)
	.def_readwrite("Gi", &pyCQCGL1dEIDc::Gi)
	.def_readonly("Ndim", &pyCQCGL1dEIDc::Ndim)
	.def_readonly("Ne", &pyCQCGL1dEIDc::Ne)
	.def_readonly("Omega", &pyCQCGL1dEIDc::Omega)
	.def_readwrite("eidc", &pyCQCGL1dEIDc::eidc)
	.def("Ts", &pyCQCGL1dEIDc::PYTs)
	.def("hs", &pyCQCGL1dEIDc::PYhs)
	.def("lte", &pyCQCGL1dEIDc::PYlte)

	.def("changeOmega", &pyCQCGL1dEIDc::changeOmega)
	.def("setScheme", &pyCQCGL1dEIDc::setScheme)
	.def("intgC", &pyCQCGL1dEIDc::PYintgC)
	.def("intg", &pyCQCGL1dEIDc::PYintg)
	;

}
