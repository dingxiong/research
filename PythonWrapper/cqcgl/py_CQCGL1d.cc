#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL1d.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL1d : public CQCGL1d {
  
public:

    pyCQCGL1d(int N, double d,
	      double Mu, double Dr, double Di, double Br, double Bi, 
	      double Gr, double Gi, int dimTan):
	CQCGL1d(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}
    
    pyCQCGL1d(int N, double d,
	      double b, double c, double dr, double di,
	      int dimTan) :
	CQCGL1d(N, d, b, c, dr, di, dimTan) {}

    pyCQCGL1d(int N, double d,
	      double delta, double beta, double D, double epsilon,
	      double mu, double nu, int dimTan):
	CQCGL1d(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

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
    bn::ndarray PYintg(bn::ndarray a0, double h, int Nt, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, h, Nt, skip_rate));
    }

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, double h, int Nt, size_t skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = intgj(tmpa, h, Nt, skip_rate);
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }
    
    bn::ndarray PYaintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(aintg(tmpa, h, tend, skip_rate));
    }

    bp::tuple PYaintgj(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = aintgj(tmpa, h, tend, skip_rate);
	return bp::make_tuple(copy2bn(result.first), 
			      copy2bn(result.second)
			      );
    }

    bn::ndarray PYintgv(bn::ndarray a0, bn::ndarray v, double h, int Nt){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn(intgv(tmpa, tmpv, h, Nt));
    }    

    bn::ndarray PYaintgv(bn::ndarray a0, bn::ndarray v, double h, double tend){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn(aintgv(tmpa, tmpv, h, tend));
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
    
    /* orbit2slice */
    bp::tuple PYorbit2slice(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
    }

    /* orbit2sliceUnwrap */
    bp::tuple PYorbit2sliceWrap(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2sliceWrap(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
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

    /* reflectVeAll */
    bn::ndarray PYreflectVeAll(const bn::ndarray &veHat, const bn::ndarray &aaHat,
			       const int trunc){
	int m, n;
	getDims(veHat, m, n);	
	Map<MatrixXd> tmpveHat((double*)veHat.get_data(), n, m);
	int m2, n2;
	getDims(aaHat, m2, n2);
	Map<ArrayXd> tmpaaHat((double*)aaHat.get_data(), n2, m2);
	return copy2bn( reflectVeAll(tmpveHat, tmpaaHat, trunc) );
    }
    
    
    /* ve2slice */
    bn::ndarray PYve2slice(bn::ndarray ve, bn::ndarray x){
	int m, n;
	getDims(ve, m, n);
	int m2, n2;
	getDims(x, m2, n2);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	Map<ArrayXd> tmpx((double*)x.get_data(), n2*m2);
	return copy2bn( ve2slice(tmpve, tmpx) );
    }
    
    /* reduceAllSymmetries */
    bp::tuple PYreduceAllSymmetries(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = 
	    reduceAllSymmetries(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
	
    }

    /* reduceVe */
    bn::ndarray PYreduceVe(const bn::ndarray &ve, const bn::ndarray &x){
	int m, n;
	getDims(ve, m, n);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	int m2, n2;
	getDims(x, m2, n2);
	Map<ArrayXd> tmpx((double*)x.get_data(), n2*m2);
	return copy2bn(reduceVe(tmpve, tmpx));
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
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL1d) {
    bn::initialize();

    // must provide the constructor
    bp::class_<CQCGL1d>("CQCGL1d", bp::init<
			int, double, 
			double, double, double, double, 
			int>())
	;
    
    
    bp::class_<pyCQCGL1d, bp::bases<CQCGL1d> >("pyCQCGL1d", bp::init<
					       int, double, 
					       double, double, double, double,
					       int >())
	.def(bp::init<int, double, double, double, double, double, double, 
	     double, double, int>())
	.def(bp::init<int, double, double, double, double, double,
	     double, double, int>())
	.def_readonly("N", &pyCQCGL1d::N)
	.def_readonly("d", &pyCQCGL1d::d)
	.def_readonly("Mu", &pyCQCGL1d::Mu)
	.def_readonly("Br", &pyCQCGL1d::Br)
	.def_readonly("Bi", &pyCQCGL1d::Bi)
	.def_readonly("Dr", &pyCQCGL1d::Dr)
	.def_readonly("Di", &pyCQCGL1d::Di)
	.def_readonly("Gr", &pyCQCGL1d::Gr)
	.def_readonly("Gi", &pyCQCGL1d::Gi)
	.def_readonly("Ndim", &pyCQCGL1d::Ndim)
	.def_readonly("Ne", &pyCQCGL1d::Ne)
	.def_readonly("Omega", &pyCQCGL1d::Omega)
	.def_readwrite("rtol", &pyCQCGL1d::rtol)
	.def_readwrite("nu", &pyCQCGL1d::nu)
	.def_readwrite("mumax", &pyCQCGL1d::mumax)
	.def_readwrite("mumin", &pyCQCGL1d::mumin)
	.def_readwrite("mue", &pyCQCGL1d::mue)
	.def_readwrite("muc", &pyCQCGL1d::muc)
	.def_readwrite("NCalCoe", &pyCQCGL1d::NCalCoe)
	.def_readwrite("NReject", &pyCQCGL1d::NReject)
	.def_readwrite("NCallF", &pyCQCGL1d::NCallF)
	.def_readwrite("NSteps", &pyCQCGL1d::NSteps)
	.def_readwrite("Method", &pyCQCGL1d::Method)
	.def("Ts", &pyCQCGL1d::PYTs)
	.def("hs", &pyCQCGL1d::PYhs)
	.def("lte", &pyCQCGL1d::PYlte)
	.def("K", &pyCQCGL1d::PYK)
	.def("L", &pyCQCGL1d::PYL)
	
	.def("changeOmega", &pyCQCGL1d::changeOmega)
	.def("intg", &pyCQCGL1d::PYintg)
	.def("intgj", &pyCQCGL1d::PYintgj)
	.def("aintg", &pyCQCGL1d::PYaintg)
	.def("aintgj", &pyCQCGL1d::PYaintgj)
	.def("intgv", &pyCQCGL1d::PYintgv)
	.def("aintgv", &pyCQCGL1d::PYaintgv)
	.def("velocity", &pyCQCGL1d::PYvelocity)
	.def("velSlice", &pyCQCGL1d::PYvelSlice)
	.def("velPhase", &pyCQCGL1d::PYvelPhase)
	.def("velocityReq", &pyCQCGL1d::PYvelocityReq)
	.def("Fourier2Config", &pyCQCGL1d::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL1d::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCQCGL1d::PYFourier2ConfigMag)
	.def("Fourier2Phase", &pyCQCGL1d::PYFourier2Phase)
	.def("orbit2sliceWrap", &pyCQCGL1d::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCQCGL1d::PYorbit2slice)
	.def("stab", &pyCQCGL1d::PYstab)
	.def("stabReq", &pyCQCGL1d::PYstabReq)
	.def("reflect", &pyCQCGL1d::PYreflect)
	.def("reduceReflection", &pyCQCGL1d::PYreduceReflection)
	.def("refGradMat", &pyCQCGL1d::PYrefGradMat)
	.def("reflectVe", &pyCQCGL1d::PYreflectVe)
	.def("reflectVeAll", &pyCQCGL1d::PYreflectVeAll)
	.def("ve2slice", &pyCQCGL1d::PYve2slice)
	.def("reduceAllSymmetries", &pyCQCGL1d::PYreduceAllSymmetries)
	.def("reduceVe", &pyCQCGL1d::PYreduceVe)
	.def("transRotate", &pyCQCGL1d::PYtransRotate) 
	.def("transTangent", &pyCQCGL1d::PYtransTangent)
	.def("phaseRotate", &pyCQCGL1d::PYphaseRotate)
	.def("phaseTangent", &pyCQCGL1d::PYphaseTangent)
	.def("Rotate", &pyCQCGL1d::PYRotate)
	.def("rotateOrbit", &pyCQCGL1d::PYrotateOrbit)
	.def("C2R", &pyCQCGL1d::PYC2R)
	;

}
