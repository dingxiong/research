#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL : public CQCGL {
  
public:

    pyCQCGL(int N, double d,
	    double b, double c, double dr, double di,
	    int dimTan, int threadNum) :
	CQCGL(N, d, b, c, dr, di, dimTan, threadNum) {}

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
    
    bp::tuple PYaintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = aintg(tmpa, h, tend, skip_rate);
	return bp::make_tuple(copy2bn(result.first), copy2bn(result.second));
    }

    bp::tuple PYaintgj(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	auto result = aintgj(tmpa, h, tend, skip_rate);
	return bp::make_tuple(copy2bn(std::get<0>(result)), 
			      copy2bn(std::get<1>(result)), 
			      copy2bn(std::get<2>(result))
			      );
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
	return copy2bn( transRotate(tmpaa, phi) );
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
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL_threads) {
    bn::initialize();

    // must provide the constructor
    bp::class_<CQCGL>("CQCGL", bp::init<
		      int, double, 
		      double, double, double, double, 
		      int, int>())
	;
    
    
    bp::class_<pyCQCGL, bp::bases<CQCGL> >("pyCQCGL", bp::init<
					   int, double, 
					   double, double, double, double,
					   int, int >())
	.def_readonly("N", &pyCQCGL::N)
	.def_readonly("d", &pyCQCGL::d)
	.def_readonly("Mu", &pyCQCGL::Mu)
	.def_readonly("Br", &pyCQCGL::Br)
	.def_readonly("Bi", &pyCQCGL::Bi)
	.def_readonly("Dr", &pyCQCGL::Dr)
	.def_readonly("Di", &pyCQCGL::Di)
	.def_readonly("Gr", &pyCQCGL::Gr)
	.def_readonly("Gi", &pyCQCGL::Gi)
	.def_readonly("Ndim", &pyCQCGL::Ndim)
	.def_readonly("Ne", &pyCQCGL::Ne)
	.def_readonly("b", &pyCQCGL::b)
	.def_readonly("c", &pyCQCGL::c)
	.def_readonly("dr", &pyCQCGL::dr)
	.def_readonly("di", &pyCQCGL::di)
	.def_readonly("Omega", &pyCQCGL::Omega)
	.def_readwrite("rtol", &pyCQCGL::rtol)
	.def_readwrite("nu", &pyCQCGL::nu)
	.def_readwrite("mumax", &pyCQCGL::mumax)
	.def_readwrite("mumin", &pyCQCGL::mumin)
	.def_readwrite("mue", &pyCQCGL::mue)
	.def_readwrite("muc", &pyCQCGL::muc)
	.def_readwrite("NCalCoe", &pyCQCGL::NCalCoe)
	.def_readwrite("NReject", &pyCQCGL::NReject)
	.def_readwrite("NCallF", &pyCQCGL::NCallF)
	.def_readwrite("Method", &pyCQCGL::Method)
	.def("hs", &pyCQCGL::PYhs)
	.def("lte", &pyCQCGL::PYlte)
	.def("K", &pyCQCGL::PYK)
	.def("L", &pyCQCGL::PYL)
	
	.def("changeOmega", &pyCQCGL::changeOmega)
	.def("intg", &pyCQCGL::PYintg)
	.def("intgj", &pyCQCGL::PYintgj)
	.def("aintg", &pyCQCGL::PYaintg)
	.def("aintgj", &pyCQCGL::PYaintgj)
	.def("velocity", &pyCQCGL::PYvelocity)
	.def("velSlice", &pyCQCGL::PYvelSlice)
	.def("velPhase", &pyCQCGL::PYvelPhase)
	.def("velocityReq", &pyCQCGL::PYvelocityReq)
	.def("Fourier2Config", &pyCQCGL::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCQCGL::PYFourier2ConfigMag)
	.def("Fourier2Phase", &pyCQCGL::PYFourier2Phase)
	.def("orbit2sliceWrap", &pyCQCGL::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCQCGL::PYorbit2slice)
	.def("stab", &pyCQCGL::PYstab)
	.def("stabReq", &pyCQCGL::PYstabReq)
	.def("reflect", &pyCQCGL::PYreflect)
	.def("reduceReflection", &pyCQCGL::PYreduceReflection)
	.def("refGradMat", &pyCQCGL::PYrefGradMat)
	.def("reflectVe", &pyCQCGL::PYreflectVe)
	.def("reflectVeAll", &pyCQCGL::PYreflectVeAll)
	.def("ve2slice", &pyCQCGL::PYve2slice)
	.def("reduceAllSymmetries", &pyCQCGL::PYreduceAllSymmetries)
	.def("reduceVe", &pyCQCGL::PYreduceVe)
	.def("transRotate", &pyCQCGL::PYtransRotate) 
	.def("transTangent", &pyCQCGL::PYtransTangent)
	.def("phaseRotate", &pyCQCGL::PYphaseRotate)
	.def("phaseTangent", &pyCQCGL::PYphaseTangent)
	.def("Rotate", &pyCQCGL::PYRotate)
	.def("rotateOrbit", &pyCQCGL::PYrotateOrbit)
	;

}
