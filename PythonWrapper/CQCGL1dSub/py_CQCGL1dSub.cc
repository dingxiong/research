#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL1dSub.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL1dSub : public CQCGL1dSub {
  
public:

    pyCQCGL1dSub(int N, double d,
		 double Mu, double Dr, double Di, double Br, double Bi, 
		 double Gr, double Gi, int dimTan):
	CQCGL1dSub(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}
    
    pyCQCGL1dSub(int N, double d,
		 double b, double c, double dr, double di,
		 int dimTan) :
	CQCGL1dSub(N, d, b, c, dr, di, dimTan) {}

    pyCQCGL1dSub(int N, double d,
		 double delta, double beta, double D, double epsilon,
		 double mu, double nu, int dimTan):
	CQCGL1dSub(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

    bn::ndarray PYTs(){
	return copy2bn(Ts);
    }

    bn::ndarray PYlte(){
	return copy2bn(lte);
    }

    bn::ndarray PYhs(){
	return copy2bn(hs);
    }

    /* K */
    bn::ndarray PYK(){
	return copy2bn(K);
    }

    /* L */
    bn::ndarray PYL(){
	return copy2bnc(L);
    }

   /* wrap the integrator */
    bn::ndarray PYintgC(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intgC(tmpa, h, tend, skip_rate));
    }

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgjC(bn::ndarray a0, double h, double tend, size_t skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = intgjC(tmpa, h, tend, skip_rate);
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }
    
    bn::ndarray PYintgvC(bn::ndarray a0, bn::ndarray v, double h, double tend){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn(intgvC(tmpa, tmpv, h, tend));
    }    

    bn::ndarray PYintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, h, tend, skip_rate));
    }

    bp::tuple PYintgj(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = intgj(tmpa, h, tend, skip_rate);
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }

    bn::ndarray PYintgv(bn::ndarray a0, bn::ndarray v, double h, double tend){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn(intgv(tmpa, tmpv, h, tend));
    }

    /* wrap Fourier2Config */
    bn::ndarray PYFourier2Config(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bnc( Fourier2Config(tmpaa) );
    }

    /* wrap Config2Fourier */
    bn::ndarray PYConfig2Fourier(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXcd> tmpAA((dcp*)AA.get_data(), n, m);
	return copy2bn( Config2Fourier(tmpAA) );
    }
#if 0   
    /* wrap Fourier2ConfigMag */
    bn::ndarray PYFourier2ConfigMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2ConfigMag(tmpaa) );
    }
    
    /* wrap Fourier2Phase */
    bn::ndarray PYFourier2Phase(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2Phase(tmpaa) );
    }
    
    bn::ndarray PYcalQ(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(calQ(tmpaa));
    }
    
    bn::ndarray PYcalMoment(const bn::ndarray &aa, const int p){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn(calMoment(tmpaa, p));
    }
    
    bn::ndarray PYcalMoment2(const bn::ndarray &aa){
	return PYcalMoment(aa, 1);
    }
    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(velocity(tmpa));
    }

    /* wrap the velSlice */
    bn::ndarray PYvelSlice(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	return copy2bn(velSlice(tmpa));
    }

    /* wrap the velPhase */
    bn::ndarray PYvelPhase(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velPhase(tmpa);
	return copy2bn(tmpv);
    }  

    /* wrap velocityReq */
    bn::ndarray PYvelocityReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocityReq(tmpa, th, phi);
	return copy2bn(tmpv);
    }

    /* orbit2slice */
    bp::tuple PYorbit2slice(const bn::ndarray &aa, int method){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(tmpaa, method);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), 
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
    }

    bn::ndarray PYve2slice(const bn::ndarray &ve, const bn::ndarray &x, int flag){
	int m, n;
	getDims(ve, m, n);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	getDims(x, m, n);
	Map<ArrayXd> tmpx((double*)x.get_data(), n*m);

	return copy2bn(ve2slice(tmpve, tmpx, flag));
    }

    /* stability matrix */
    bn::ndarray PYstab(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	return copy2bn(stab(tmpa));
    }

    /* stability matrix for relative equibrium */
    bn::ndarray PYstabReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	return copy2bn(stabReq(tmpa, th, phi));
    }


    /* reflection */
    bn::ndarray PYreflect(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( reflect(tmpaa) );
    }


    /* reduceReflection */
    bn::ndarray PYreduceReflection(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( reduceReflection(tmpaa) );
    }

    /* refGradMat */
    bn::ndarray PYrefGradMat(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXd> tmpaa((double*)aa.get_data(), n * m);
	return copy2bn( refGradMat(tmpaa) );
    }

    /* reflectVe */
    bn::ndarray PYreflectVe(const bn::ndarray &veHat, const bn::ndarray &xHat){
	int m, n;
	getDims(veHat, m, n);	
	Map<MatrixXd> tmpveHat((double*)veHat.get_data(), n, m);
	int m2, n2;
	getDims(xHat, m2, n2);
	Map<ArrayXd> tmpxHat((double*)xHat.get_data(), n2*m2);
	return copy2bn( reflectVe(tmpveHat, tmpxHat) );
    }
    
    /* transRotate */
    bn::ndarray PYtransRotate(bn::ndarray aa, double th){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transRotate(tmpaa, th) );
    }

    /* transTangent */
    bn::ndarray PYtransTangent(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transTangent(tmpaa) );
    }

    /* phaseRotate */
    bn::ndarray PYphaseRotate(bn::ndarray aa, double phi){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( phaseRotate(tmpaa, phi) );
    }
    
    /* phaseTangent */
    bn::ndarray PYphaseTangent(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( phaseTangent(tmpaa) );
    }

    /* Rotate */
    bn::ndarray PYRotate(bn::ndarray aa, double th, double phi){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Rotate(tmpaa, th, phi) );
    }

    /* rotateOrbit */
    bn::ndarray PYrotateOrbit(bn::ndarray aa, bn::ndarray th, bn::ndarray phi){
	int m, n;
	getDims(aa, m, n);
	int m2, n2;
	getDims(th, m2, n2);
	int m4, n4;
	getDims(phi, m4, n4);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	Map<ArrayXd> tmpth((double*)th.get_data(), n2 * m2);
	Map<ArrayXd> tmpphi((double*)phi.get_data(), n4 * m4);
	return copy2bn( rotateOrbit(tmpaa, tmpth, tmpphi) );
    }

    bn::ndarray PYC2R(bn::ndarray v){
	int m, n;
	getDims(v, m, n);
	Map<ArrayXXcd> tmpv((dcp*)v.get_data(), n, m);
	return copy2bn( C2R(tmpv) );
    }
#endif    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL1dSub) {
    bn::initialize();

    // must provide the constructor
    bp::class_<CQCGL1dSub>("CQCGL1dSub", bp::init<
			int, double, 
			double, double, double, double, 
			int>())
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
    
    bp::class_<pyCQCGL1dSub, bp::bases<CQCGL1dSub> >("pyCQCGL1dSub", bp::init<
					       int, double, 
					       double, double, double, double,
					       int >())
	.def(bp::init<int, double, double, double, double, double, double, 
	     double, double, int>())
	.def(bp::init<int, double, double, double, double, double,
	     double, double, int>())
	.def_readonly("N", &pyCQCGL1dSub::N)
	.def_readonly("d", &pyCQCGL1dSub::d)
	.def_readwrite("IsQintic", &pyCQCGL1dSub::IsQintic)
	.def_readonly("Mu", &pyCQCGL1dSub::Mu)
	.def_readwrite("Br", &pyCQCGL1dSub::Br)
	.def_readwrite("Bi", &pyCQCGL1dSub::Bi)
	.def_readonly("Dr", &pyCQCGL1dSub::Dr)
	.def_readonly("Di", &pyCQCGL1dSub::Di)
	.def_readwrite("Gr", &pyCQCGL1dSub::Gr)
	.def_readwrite("Gi", &pyCQCGL1dSub::Gi)
	.def_readonly("Ndim", &pyCQCGL1dSub::Ndim)
	.def_readonly("Ne", &pyCQCGL1dSub::Ne)
	.def_readonly("Omega", &pyCQCGL1dSub::Omega)
	.def_readwrite("eidc", &pyCQCGL1dSub::eidc)
	.def_readwrite("eidc2", &pyCQCGL1dSub::eidc2)
	.def("Ts", &pyCQCGL1dSub::PYTs)
	.def("hs", &pyCQCGL1dSub::PYhs)
	.def("lte", &pyCQCGL1dSub::PYlte)
	.def("K", &pyCQCGL1dSub::PYK)
	.def("L", &pyCQCGL1dSub::PYL)
	
	.def("changeOmega", &pyCQCGL1dSub::changeOmega)
	.def("intgC", &pyCQCGL1dSub::PYintgC)
	.def("intgjC", &pyCQCGL1dSub::PYintgjC)
	.def("intgvC", &pyCQCGL1dSub::PYintgv)
	.def("intg", &pyCQCGL1dSub::PYintg)
	.def("intgj", &pyCQCGL1dSub::PYintgj)
	.def("intgv", &pyCQCGL1dSub::PYintgv)
	.def("Fourier2Config", &pyCQCGL1dSub::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL1dSub::PYConfig2Fourier)
	// .def("Fourier2ConfigMag", &pyCQCGL1dSub::PYFourier2ConfigMag)
	// .def("Fourier2Phase", &pyCQCGL1dSub::PYFourier2Phase)
	// .def("calQ", &pyCQCGL1dSub::PYcalQ)
	// .def("calMoment", &pyCQCGL1dSub::PYcalMoment)
	// .def("calMoment", &pyCQCGL1dSub::PYcalMoment2)
	// .def("velocity", &pyCQCGL1dSub::PYvelocity)
	// .def("velSlice", &pyCQCGL1dSub::PYvelSlice)
	// .def("velPhase", &pyCQCGL1dSub::PYvelPhase)
	// .def("velocityReq", &pyCQCGL1dSub::PYvelocityReq)
	// .def("orbit2slice", &pyCQCGL1dSub::PYorbit2slice)
	// .def("ve2slice", &pyCQCGL1dSub::PYve2slice)
	// .def("stab", &pyCQCGL1dSub::PYstab)
	// .def("stabReq", &pyCQCGL1dSub::PYstabReq)
	// .def("reflect", &pyCQCGL1dSub::PYreflect)
	// .def("reduceReflection", &pyCQCGL1dSub::PYreduceReflection)
	// .def("refGradMat", &pyCQCGL1dSub::PYrefGradMat)
	// .def("reflectVe", &pyCQCGL1dSub::PYreflectVe)
	// .def("transRotate", &pyCQCGL1dSub::PYtransRotate) 
	// .def("transTangent", &pyCQCGL1dSub::PYtransTangent)
	// .def("phaseRotate", &pyCQCGL1dSub::PYphaseRotate)
	// .def("phaseTangent", &pyCQCGL1dSub::PYphaseTangent)
	// .def("Rotate", &pyCQCGL1dSub::PYRotate)
	// .def("rotateOrbit", &pyCQCGL1dSub::PYrotateOrbit)
	// .def("C2R", &pyCQCGL1dSub::PYC2R)
	;
}
