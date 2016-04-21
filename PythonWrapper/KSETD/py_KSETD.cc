#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "KSETD.hpp"
#include "ETDRK4.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
class pyKSETD : public KSETD {
public :
    pyKSETD(int N, double d) : KSETD(N, d) {}

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


BOOST_PYTHON_MODULE(py_ks) {
    bn::initialize();    
    bp::class_<KSETD>("KSETD", bp::init<int, double>())
	;
    
    bp::class_<pyKSETD, bp::bases<KSETD> >("pyKSETD", bp::init<int, double>())
	.def_readonly("N", &pyKSETD::N)
	.def_readonly("d", &pyKSETD::d)
	.def("etdParam", &pyKSETD::PYetdParam)
	.def("setRtol", &pyKSETD::PYsetRtol)
	.def("etd", &pyKSETD::PYetd)
	.def("hs", &pyKSETD::PYhs)
	.def("duu", &pyKSETD::PYduu)
	; 
}
