#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "ksint.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


class pyKS : public KS {
  
public:
    pyKS(int N, double d) : KS(N, d) {}
    
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
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);

	return copy2bn(intgC(tmpa, h, tend, skip_rate));
    }

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgjC(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	auto tmpav = intgjC(tmpa, h, tend, skip_rate);
	
	return bp::make_tuple(copy2bn(tmpav.first), copy2bn(tmpav.second));
    }

    bn::ndarray PYintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	return copy2bn(intg(tmpa, h, tend, skip_rate));
    }

    bp::tuple PYintgj(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	auto result = intgj(tmpa, h, tend, skip_rate);
	
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }

    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);

	return copy2bn(velocity(tmpa));
    }
    
    bn::ndarray PYvelReq(bn::ndarray a0, double theta){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(velReq(tmpa, theta));
    }
    
    /* stab */
    bn::ndarray PYstab(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(stab(tmpa));
    }

    bn::ndarray PYstabReq(bn::ndarray a0, double theta){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(stabReq(tmpa, theta));
    }
    
    /* reflection */
    bn::ndarray PYreflect(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(reflect(tmpaa));
    }

    /* half2whole */
    bn::ndarray PYhalf2whole(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(half2whole(tmpaa));
    }

    /* Rotation */
    bn::ndarray PYrotate(bn::ndarray aa, const double th){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(rotate(tmpaa, th));
    }
        
    bp::tuple PYredSO2(bn::ndarray aa, int p, bool toY){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	auto tmp = redSO2(tmpaa, p, toY);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }

    bp::tuple PYfundDomain(bn::ndarray aa, int pSlice, int pFund){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	auto tmp = fundDomain(tmpaa, pSlice, pFund);
	return bp::make_tuple( copy2bn(tmp.first), copy2bn<ArrayXXi, int>(tmp.second));
    }
    
    bp::tuple PYredO2f(bn::ndarray aa, int pSlice, int pFund){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	auto tmp = redO2f(tmpaa, pSlice, pFund);
	return bp::make_tuple( copy2bn(std::get<0>(tmp)),
			       copy2bn<ArrayXXi, int>(std::get<1>(tmp)),
			       copy2bn(std::get<2>(tmp))
			       );
    }
    
    bn::ndarray PYredV(bn::ndarray v, bn::ndarray a, int p, bool toY){
	int m, n;
	getDims(v, m, n);
	Map<MatrixXd> tmpv((double*)v.get_data(), n, m);
	getDims(a, m, n);
	Map<VectorXd> tmpa((double*)a.get_data(), n * m);

	return copy2bn( redV(tmpv, tmpa, p, toY) );
    }
    
    bn::ndarray PYcalMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	return copy2bn(calMag(tmpaa));
    }

    bp::tuple PYtoPole(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	auto tmp = toPole(tmpaa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


BOOST_PYTHON_MODULE(py_ks) {
    bn::initialize();    
    bp::class_<KS>("KS", bp::init<int, double>()) ;

    // the EIDr class
    bp::class_<EIDr>("EIDr", bp::init<>())
	.def_readonly("NCalCoe", &EIDr::NCalCoe)
	.def_readonly("NReject", &EIDr::NReject)
	.def_readonly("NCallF", &EIDr::NCallF)
	.def_readonly("NSteps", &EIDr::NSteps)
	.def_readonly("TotalTime", &EIDr::TotalTime)
	.def_readonly("CoefficientTime", &EIDr::CoefficientTime)
	.def_readwrite("rtol", &EIDr::rtol)
	;
    
    bp::class_<pyKS, bp::bases<KS> >("pyKS", bp::init<int, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readwrite("eidr", &pyKS::eidr)
	.def_readwrite("eidr2", &pyKS::eidr2)
	.def("Ts", &pyKS::PYTs)
	.def("hs", &pyKS::PYhs)
	.def("lte", &pyKS::PYlte)
	.def("intgC", &pyKS::PYintgC)
	.def("intgjC", &pyKS::PYintgjC)
	.def("intg", &pyKS::PYintg)
	.def("intgj", &pyKS::PYintgj)
	.def("velocity", &pyKS::PYvelocity)
	.def("velReq", &pyKS::PYvelReq)
	.def("stab", &pyKS::PYstab)
	.def("stabReq", &pyKS::PYstabReq)
	.def("reflect", &pyKS::PYreflect)
	.def("half2whole", &pyKS::PYhalf2whole)
	.def("rotate", &pyKS::PYrotate)
	.def("redSO2", &pyKS::PYredSO2)
	.def("fundDomain", &pyKS::PYfundDomain)
	.def("redO2f", &pyKS::PYredO2f)
	.def("redV", &pyKS::PYredV)
	.def("calMag", &pyKS::PYcalMag)
	.def("toPole", &pyKS::PYtoPole)
	;

}
