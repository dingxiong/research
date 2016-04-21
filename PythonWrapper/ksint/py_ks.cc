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
  /*
   * Python interface for ksint.cc
   * Note:
   *     1) Input and out put are arrays of C form, meaning each
   *        row is a vector
   */
  
public:
    pyKS(int N, double d) : KS(N, d) {}
    
    bn::ndarray PYlte(){
	return copy2bn(lte);
    }

    bn::ndarray PYhs(){
	return copy2bn(hs);
    }

    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);

	return copy2bn(velocity(tmpa));
    }
    
    bn::ndarray PYvelg(bn::ndarray a0, double theta){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(velg(tmpa, theta));
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

    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, double h, int Nt, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);

	return copy2bn(intg(tmpa, h, Nt, skip_rate));
    }

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, double h, int Nt, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	auto tmpav = intgj(tmpa, h, Nt, skip_rate);
	
	return bp::make_tuple(copy2bn(tmpav.first), copy2bn(tmpav.second));
    }

   bp::tuple PYaintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	auto result = aintg(tmpa, h, tend, skip_rate);
	
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }

    bp::tuple PYaintgj(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	auto result = aintgj(tmpa, h, tend, skip_rate);
	
	return bp::make_tuple(copy2bn(std::get<0>(result)),
			      copy2bn(std::get<1>(result)),
			      copy2bn(std::get<2>(result))
			      );
    }
    
    /* reflection */
    bn::ndarray PYreflection(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(Reflection(tmpaa));
    }


    /* half2whole */
    bn::ndarray PYhalf2whole(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(half2whole(tmpaa));
    }


    /* Rotation */
    bn::ndarray PYrotation(bn::ndarray aa, const double th){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(Rotation(tmpaa, th));
    }


    /* orbitToSlice */
    bp::tuple PYorbitToSlice(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	auto result = orbitToSlice(tmpaa);
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }


    /* veToSlice */
    bn::ndarray PYveToSlice(bn::ndarray ve, bn::ndarray x){
	int m, n;
	getDims(ve, m, n);
	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	getDims(x, m, n);
	Map<VectorXd> tmpx((double*)x.get_data(), n*m);
	return copy2bn(veToSlice(tmpve, tmpx));
    }

    /* veToSliceAll */
    bn::ndarray PYveToSliceAll(bn::ndarray eigVecs, bn::ndarray aa, const int trunc){
	int m, n;
	getDims(eigVecs, m, n);
	Map<MatrixXd> tmpeigVecs((double*)eigVecs.get_data(), n, m);
	getDims(aa, m, n);	
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	return copy2bn( veToSliceAll(tmpeigVecs , tmpaa, trunc) );
    }

    /* orbitAndFvWholeSlice  */
	bp::tuple PYorbitAndFvWholeSlice(bn::ndarray a, bn::ndarray ve, double h,
				     const int nstp, const string ppType,
				     const int pos){
	int m, n;
	getDims(a, m, n);
	Map<ArrayXd> tmpa((double*)a.get_data(), n*m);
	getDims(ve, m, n);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	auto tmpav = orbitAndFvWholeSlice(tmpa, tmpve, h, nstp, ppType, pos);

	return bp::make_tuple(copy2bn(tmpav.first), copy2bn(tmpav.second));
    }

    /* reduceReflection */
    bn::ndarray PYreduceReflection(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(reduceReflection(tmpaa));
    }

    /* reflectVe */
    bn::ndarray PYreflectVe(bn::ndarray ve, bn::ndarray x){
	int m, n;
	getDims(ve, m, n);
	int m2, n2;
	getDims(x, m2, n2);
	
	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	Map<ArrayXd> tmpx((double*)x.get_data(), m2*n2);
	return copy2bn( reflectVe(tmpve, tmpx) );
    }

    /* reflectVeAll */
    bn::ndarray PYreflectVeAll(bn::ndarray ve, bn::ndarray x, int trunc){
	int m, n;
	getDims(ve, m, n);
	int m2, n2;
	getDims(x, m2, n2);
	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	Map<MatrixXd> tmpx((double*)x.get_data(), n2, m2);
	return copy2bn( reflectVeAll(tmpve, tmpx, trunc) );
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

    
    bp::tuple PYredSO2(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	auto tmp = redSO2(tmpaa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

    bn::ndarray PYredRef(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	return copy2bn(redRef(tmpaa));
    }

    bp::tuple PYredO2(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	auto tmp = redO2(tmpaa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

    bp::tuple PYredV(bn::ndarray v, bn::ndarray a){
	int m, n;
	getDims(v, m, n);
	Map<MatrixXd> tmpv((double*)v.get_data(), n, m);
	getDims(a, m, n);
	Map<VectorXd> tmpa((double*)a.get_data(), n * m);

	auto tmp = redV(tmpv, tmpa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

    bn::ndarray PYredV2(bn::ndarray v, bn::ndarray a){
	int m, n;
	getDims(v, m, n);
	Map<MatrixXd> tmpv((double*)v.get_data(), n, m);
	getDims(a, m, n);
	Map<VectorXd> tmpa((double*)a.get_data(), n * m);

	return copy2bn(redV2(tmpv, tmpa));
    }

};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


BOOST_PYTHON_MODULE(py_ks) {
    bn::initialize();    
    bp::class_<KS>("KS", bp::init<int, double>()) ;
    
    bp::class_<pyKS, bp::bases<KS> >("pyKS", bp::init<int, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readwrite("rtol", &pyKS::rtol)
	.def_readwrite("nu", &pyKS::nu)
	.def_readwrite("mumax", &pyKS::mumax)
	.def_readwrite("mumin", &pyKS::mumin)
	.def_readwrite("mue", &pyKS::mue)
	.def_readwrite("muc", &pyKS::muc)
	.def_readwrite("NCalCoe", &pyKS::NCalCoe)
	.def_readwrite("NReject", &pyKS::NReject)
	.def_readwrite("NCallF", &pyKS::NCallF)
	.def_readwrite("Method", &pyKS::Method)
	.def("hs", &pyKS::PYhs)
	.def("lte", &pyKS::PYlte)
	.def("velocity", &pyKS::PYvelocity)
	.def("velg", &pyKS::PYvelg)
	.def("stab", &pyKS::PYstab)
	.def("stabReq", &pyKS::PYstabReq)
	.def("intg", &pyKS::PYintg)
	.def("intgj", &pyKS::PYintgj)
	.def("aintg", &pyKS::PYaintg)
	.def("aintgj", &pyKS::PYaintgj)
	.def("Reflection", &pyKS::PYreflection)
	.def("half2whole", &pyKS::PYhalf2whole)
	.def("Rotation", &pyKS::PYrotation)
	.def("orbitToSlice", &pyKS::PYorbitToSlice)
	.def("veToSlice", &pyKS::PYveToSlice)
	.def("veToSliceAll", &pyKS::PYveToSliceAll)
	.def("orbitAndFvWholeSlice", &pyKS::PYorbitAndFvWholeSlice)
	.def("reduceReflection", &pyKS::PYreduceReflection)
	.def("reflectVe", &pyKS::PYreflectVe)
	.def("reflectVeAll", &pyKS::PYreflectVeAll)
	.def("calMag", &pyKS::PYcalMag)
	.def("toPole", &pyKS::PYtoPole)
	.def("redSO2", &pyKS::PYredSO2)
	.def("redRef", &pyKS::PYredRef)
	.def("redO2", &pyKS::PYredO2)
	.def("redV", &pyKS::PYredV)
	.def("redV2", &pyKS::PYredV2)
	;

}
