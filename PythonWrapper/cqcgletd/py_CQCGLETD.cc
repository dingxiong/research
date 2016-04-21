#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CqcglETD.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class pyCqcglETD : public CqcglETD {
public :
    pyCqcglETD(int N, double d, double W, double B, double C, double DR, double DI) : 
	CqcglETD(N, d, W, B, C, DR, DI) {}

    bp::tuple PYetd(bn::ndarray a0, double tend, double h, int skip_rate, 
		    int method, bool adapt){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa0((double*)a0.get_data(), n * m);

	auto tmp = etd(tmpa0, tend, h, skip_rate, method, adapt);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

    bn::ndarray PYhs() {
	return copy2bn(etdrk4->hs);
    }
    
    bn::ndarray PYduu() {
	return copy2bn(etdrk4->duu);
    }

    bp::tuple PYetdParam(){
	return bp::make_tuple(etdrk4->NCalCoe,
			      etdrk4->NReject,
			      etdrk4->NCallF,
			      etdrk4->rtol
			      );
    }
    
    void PYsetRtol(double x){
	etdrk4->rtol = x;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_cqcgl1d_threads) {
    bn::initialize();

    bp::class_<CqcglETD>("CqcglETD", bp::init<
			 int, double, double, double, double, double , double
			 >())
	;

    
    bp::class_<pyCqcglETD, bp::bases<CqcglETD> >("pyCqcglETD", bp::init<
						 int, double, double, double, double, double , double
						 >())
	.def_readonly("N", &pyCqcglETD::N)
	.def_readonly("d", &pyCqcglETD::d)
	.def("etdParam", &pyCqcglETD::PYetdParam)
	.def("setRtol", &pyCqcglETD::PYsetRtol)
	.def("etd", &pyCqcglETD::PYetd)
	.def("hs", &pyCqcglETD::PYhs)
	.def("duu", &pyCqcglETD::PYduu)
	; 
    
}
