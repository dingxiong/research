#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "cqcgl1d.hpp"
#include "cqcglRPO.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/* get the dimension of an array */
inline void getDims(bn::ndarray x, int &m, int &n){
    if(x.get_nd() == 1){
	m = 1;
	n = x.shape(0);
    } else {
	m = x.shape(0);
	n = x.shape(1);
    }
}

/*
 * @brief used to copy content in Eigen array to boost.numpy array.
 *
 *  Only work for double array/matrix
 */
inline bn::ndarray copy2bn(const Ref<const ArrayXXd> &x){
    int m = x.cols();
    int n = x.rows();

    Py_intptr_t dims[2];
    int ndim;
    if(m == 1){
	ndim = 1;
	dims[0] = n;
    }
    else {
	ndim = 2;
	dims[0] = m;
	dims[1] = n;
    }
    bn::ndarray px = bn::empty(ndim, dims, bn::dtype::get_builtin<double>());
    memcpy((void*)px.get_data(), (void*)x.data(), sizeof(double) * m * n);
	    
    return px;
}

/**
 * @brief std::vector to bp::list
 */
template <class T>
bp::list toList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
	list.append(*iter);
    }
    return list;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCqcgl1d : public Cqcgl1d {
    /*
     * Python interface for Cqcgl1d.cc
     * Note:
     *     1) Input and out put are arrays of C form, meaning each
     *        row is a vector
     */
  
public:
    pyCqcgl1d(int N, double d, double h, 
	      bool enableJacv, int Njacv,
	      double Mu,
	      double Br, double Bi, double Dr,
	      double Di, double Gr, double Gi,
	      int threadNum) :
	Cqcgl1d(N, d, h, enableJacv, Njacv,
		Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum) {}

    pyCqcgl1d(int N, double d, double h,
	      bool enableJacv, int Njacv,
	      double b, double c,
	      double dr, double di,
	      int threadNum) :
	Cqcgl1d(N, d, h, enableJacv, Njacv,
		b, c, dr, di, threadNum) {}
    
    /* wrap changeh */
    void PYchangeh(const double hnew){
	changeh(hnew);
    }

    /* K */
    bn::ndarray PYK(){
	return copy2bn(K);
    }

    /* L */
    bp::tuple PYL(){
	ArrayXd r = L.real();
	ArrayXd i = L.imag();
	return bp::make_tuple( copy2bn(r), copy2bn(i) );
    }

    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocity(tmpa);
	return copy2bn(tmpv);
    }

    /* wrap the velSlice */
    bn::ndarray PYvelSlice(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velSlice(tmpa);
	return copy2bn(tmpv);
    }

    /* wrap the velPhase */
    bn::ndarray PYvelPhase(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velPhase(tmpa);
	return copy2bn(tmpv);
    }
    
    /* wrap the Lyap */
    bp::tuple PYLyap(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXcd lya = Lyap(tmpaa);
	return bp::make_tuple(copy2bn( lya.real() ),
			      copy2bn( lya.imag() )
			      );
    }
    
    /* wrap the LyapVel */
    bn::ndarray PYLyapVel(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXd lyavel = LyapVel(tmpaa);
	return copy2bn(lyavel);
    }
    
    /* wrap velocityReq */
    bn::ndarray PYvelocityReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocityReq(tmpa, th, phi);
	return copy2bn(tmpv);
    }


    /* pad */
    bn::ndarray PYpad(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmppaa = pad(tmpaa);
	return copy2bn(pad(tmpaa));
    }

    /* generalPadding */
    bn::ndarray PYgeneralPadding(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( generalPadding(tmpaa) );
    }

    /* unpad */
    bn::ndarray PYunpad(bn::ndarray paa){
	int m, n;
	getDims(paa, m, n);
	Map<ArrayXXd> tmppaa((double*)paa.get_data(), n, m);
	return copy2bn(unpad(tmppaa));
    }

    
    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, int nstp, int np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, nstp, np));
    }
    

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, size_t nstp, size_t np, size_t nqr){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(tmpa, nstp, np, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }

    /* wrap intgv */
    bn::ndarray PYintgv(const bn::ndarray &a0, const bn::ndarray &v,
		      size_t nstp){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn( intgv(tmpa, tmpv, nstp));
    }

    /* wrap intgvs */
   bp::tuple PYintgvs(const bn::ndarray &a0, const bn::ndarray &v,
		      int nstp, int np, int nqr){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgvs(tmpa, tmpv, nstp, np, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }
    
    /* wrap Fourier2Config */
    bn::ndarray PYFourier2Config(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2Config(tmpaa) );
    }

    /* wrap Config2Fourier */
    bn::ndarray PYConfig2Fourier(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	return copy2bn( Config2Fourier(tmpAA) );
    }
    
    /* wrap Fourier2ConfigMag */
    bn::ndarray PYFourier2ConfigMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2ConfigMag(tmpaa) );
    }

    /* wrap calPhase */
    bn::ndarray PYcalPhase(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	return copy2bn( calPhase(tmpAA) );
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

    /* reduceIntg */
    bp::tuple PYreduceIntg(const bn::ndarray &a0, const size_t nstp,
			   const size_t np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = 
	    reduceIntg(tmpa0, nstp, np);
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
    
    /* findReq */
    bp::tuple PYfindReq(bn::ndarray a0, double wth0, double wphi0,
			int MaxN, double tol, bool doesUseMyCG,
			bool doesPrint){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXd, double, double, double> tmp = 
	    findReq(tmpa0, wth0, wphi0, MaxN, tol, doesUseMyCG, doesPrint);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),  std::get<1>(tmp),
			      std::get<2>(tmp), std::get<3>(tmp));
    }
    
    /* optThPhi */
    bp::list PYoptThPhi(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	
	return toList(optThPhi(tmpa0));
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
    
    /* planeWave */
    bp::tuple PYplaneWave(int k, bool isPositve){
	auto tmp = planeWave(k, isPositve);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      std::get<1>(tmp),
			      std::get<2>(tmp));
    }
    
    /* powIt */
    bp::tuple PYpowIt(bn::ndarray a0, double th, double phi, bn::ndarray Q0, 
		      bool onlyLastQ, int nstp, int nqr, 
		      int maxit, double Qtol, bool Print, int PrintFreqency){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	getDims(Q0, m, n);	
	Map<MatrixXd> tmpQ0((double*)Q0.get_data(), n, m);
	auto tmp = powIt(tmpa0, th, phi, tmpQ0, onlyLastQ, nstp, nqr, maxit, Qtol, Print, PrintFreqency);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)),
			      toList(std::get<3>(tmp))
			      );
    }

    /* powEigE */
    bn::ndarray PYpowEigE(bn::ndarray a0, double th, double phi, bn::ndarray Q0, 
			  int nstp, int nqr, 
			  int maxit, double Qtol, bool Print, int PrintFreqency){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	getDims(Q0, m, n);	
	Map<MatrixXd> tmpQ0((double*)Q0.get_data(), n, m);
	MatrixXd tmp = powEigE(tmpa0, th, phi, tmpQ0, nstp, nqr, maxit, Qtol, Print, PrintFreqency);
	return copy2bn(tmp);
    }
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class pyCgl1d : public Cgl1d {
    /*
     * Python interface for class Cgl
     * Copied from pyCqcgl except the constructor
     */    
public :
    pyCgl1d(int N, double d, double h,
	    bool enableJacv, int Njacv,
	    double b, double c,
	    int threadNum) :
	Cgl1d(N, d, h, enableJacv, Njacv, b, c, threadNum) {}

      /* wrap changeh */
    void PYchangeh(const double hnew){
	changeh(hnew);
    }

    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocity(tmpa);
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


    /* pad */
    bn::ndarray PYpad(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmppaa = pad(tmpaa);
	return copy2bn(pad(tmpaa));
    }

    /* generalPadding */
    bn::ndarray PYgeneralPadding(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( generalPadding(tmpaa) );
    }

    /* unpad */
    bn::ndarray PYunpad(bn::ndarray paa){
	int m, n;
	getDims(paa, m, n);
	Map<ArrayXXd> tmppaa((double*)paa.get_data(), n, m);
	return copy2bn(unpad(tmppaa));
    }

    
    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, size_t nstp, size_t np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, nstp, np));
    }
    

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, size_t nstp, size_t np, size_t nqr){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(tmpa, nstp, np, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }

    /* wrap intgv */
    bn::ndarray PYintgv(const bn::ndarray &a0, const bn::ndarray &v,
		      size_t nstp){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), m, n);
	return copy2bn( intgv(tmpa, tmpv, nstp));
    }

    
    /* wrap Fourier2Config */
    bn::ndarray PYFourier2Config(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2Config(tmpaa) );
    }

    /* wrap Config2Fourier */
    bn::ndarray PYConfig2Fourier(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	return copy2bn( Config2Fourier(tmpAA) );
    }
    
    /* wrap Fourier2ConfigMag */
    bn::ndarray PYFourier2ConfigMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2ConfigMag(tmpaa) );
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

    /* reduceIntg */
    bp::tuple PYreduceIntg(const bn::ndarray &a0, const size_t nstp,
			   const size_t np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = 
	    reduceIntg(tmpa0, nstp, np);
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
    
    /* findReq */
    bp::tuple PYfindReq(bn::ndarray a0, double wth0, double wphi0,
			int MaxN, double tol, bool doesUseMyCG,
			bool doesPrint){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXd, double, double, double> tmp = 
	    findReq(tmpa0, wth0, wphi0, MaxN, tol, doesUseMyCG, doesPrint);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),  std::get<1>(tmp),
			      std::get<2>(tmp), std::get<3>(tmp));
    }
    
    /* transRotate */
    bn::ndarray PYtransRotate(bn::ndarray aa, double th){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transRotate(tmpaa, th) );
    }

    /* phaseRotate */
    bn::ndarray PYphaseRotate(bn::ndarray aa, double phi){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transRotate(tmpaa, phi) );
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


class pyCqcglRPO : public CqcglRPO {
    
public:
    pyCqcglRPO(int nstp, int M,
	       int N, double d, double h,
	       double Mu, double Br, double Bi,
	       double Dr, double Di, double Gr,
	       double Gi, int threadNum) :
	CqcglRPO(nstp, M, N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi, threadNum) {}


    /* findRPO */
    bp::tuple PYfindRPO(const bn::ndarray &x0, const double T,
			const double th0,
			const double phi0,
			const double tol,
			const int btMaxIt,
			const int maxit,
			const double eta0,
			const double t,
			const double theta_min,
			const double theta_max,
			const int GmresRestart,
			const int GmresMaxit){
	int m, n; 
	getDims(x0, m, n);
	Map<VectorXd> tmpx0((double*)x0.get_data(), n*m); 
	auto tmp = findRPO(tmpx0, T, th0, phi0, tol, btMaxIt, maxit, eta0, t,
			   theta_min, theta_max, GmresRestart, GmresMaxit);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),  std::get<1>(tmp),
			      std::get<2>(tmp), std::get<3>(tmp), std::get<4>(tmp));
    }

    /* findRPO */
    bp::tuple PYfindRPOM(const bn::ndarray &x0, const double T,
			 const double th0,
			 const double phi0,
			 const double tol,
			 const int btMaxIt,
			 const int maxit,
			 const double eta0,
			 const double t,
			 const double theta_min,
			 const double theta_max,
			 const int GmresRestart,
			 const int GmresMaxit){
	int m, n; 
	getDims(x0, m, n);
	Map<MatrixXd> tmpx0((double*)x0.get_data(), n, m); 
	auto tmp = findRPOM(tmpx0, T, th0, phi0, tol, btMaxIt, maxit, eta0, t,
			    theta_min, theta_max, GmresRestart, GmresMaxit);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),  std::get<1>(tmp),
			      std::get<2>(tmp), std::get<3>(tmp), std::get<4>(tmp));
    }
    
};


BOOST_PYTHON_MODULE(py_cqcgl1d_omp) {
    bn::initialize();

    // must provide the constructor
    bp::class_<Cqcgl1d>("Cqcgl1d", bp::init<
			int, double, double,
			bool, int,
			double, double, double, double, double,
			double, double,
			int>())
	.def(bp::init<int, double, double, bool, int, double, double, double, double, int>())
	;
    
    bp::class_<Cgl1d>("Cgl1d", bp::init<
			int, double, double,
			bool, int,
			double, double,
			int>()) ;
    
    bp::class_<CqcglRPO>("CqcglRPO", bp::init<
			 int, int, int, double, double,
			 double, double, double,
			 double, double, double,
			 double, int >());
    
    bp::class_<pyCqcgl1d, bp::bases<Cqcgl1d> >("pyCqcgl1d", bp::init<
					       int, double, double,
					       bool, int,
					       double, double, double, double, double,
					       double, double,
					       int >())
	.def_readonly("N", &pyCqcgl1d::N)
	.def_readonly("d", &pyCqcgl1d::d)
	.def_readonly("h", &pyCqcgl1d::h)
	.def_readonly("trueNjacv", &pyCqcgl1d::trueNjacv)
	.def_readonly("Mu", &pyCqcgl1d::Mu)
	.def_readonly("Br", &pyCqcgl1d::Br)
	.def_readonly("Bi", &pyCqcgl1d::Bi)
	.def_readonly("Dr", &pyCqcgl1d::Dr)
	.def_readonly("Di", &pyCqcgl1d::Di)
	.def_readonly("Gr", &pyCqcgl1d::Gr)
	.def_readonly("Gi", &pyCqcgl1d::Gi)
	.def_readonly("Ndim", &pyCqcgl1d::Ndim)
	.def_readonly("b", &pyCqcgl1d::b)
	.def_readonly("c", &pyCqcgl1d::c)
	.def_readonly("dr", &pyCqcgl1d::dr)
	.def_readonly("di", &pyCqcgl1d::di)
	.def("K", &pyCqcgl1d::PYK)
	.def("L", &pyCqcgl1d::PYL)
	.def(bp::init<int, double, double, bool, int, double, double, double, double, int>()) /* second constructor */
	.def("changeh", &pyCqcgl1d::PYchangeh)
	.def("velocity", &pyCqcgl1d::PYvelocity)
	.def("velSlice", &pyCqcgl1d::PYvelSlice)
	.def("velPhase", &pyCqcgl1d::PYvelPhase)
	.def("velocityReq", &pyCqcgl1d::PYvelocityReq)
	.def("Lyap", &pyCqcgl1d::PYLyap)
	.def("LyapVel", &pyCqcgl1d::PYLyapVel)
	.def("pad", &pyCqcgl1d::PYpad)
	.def("generalPadding", &pyCqcgl1d::PYgeneralPadding)
	.def("unpad", &pyCqcgl1d::PYunpad)
	.def("intg", &pyCqcgl1d::PYintg)
	.def("intgj", &pyCqcgl1d::PYintgj)
	.def("intgv", &pyCqcgl1d::PYintgv)
	.def("intgvs", &pyCqcgl1d::PYintgvs)
	.def("Fourier2Config", &pyCqcgl1d::PYFourier2Config)
	.def("Config2Fourier", &pyCqcgl1d::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCqcgl1d::PYFourier2ConfigMag)
	.def("calPhase", &pyCqcgl1d::PYcalPhase)
	.def("Fourier2Phase", &pyCqcgl1d::PYFourier2Phase)
	.def("orbit2sliceWrap", &pyCqcgl1d::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCqcgl1d::PYorbit2slice)
	.def("stab", &pyCqcgl1d::PYstab)
	.def("stabReq", &pyCqcgl1d::PYstabReq)
	.def("reflect", &pyCqcgl1d::PYreflect)
	.def("reduceReflection", &pyCqcgl1d::PYreduceReflection)
	.def("refGradMat", &pyCqcgl1d::PYrefGradMat)
	.def("reflectVe", &pyCqcgl1d::PYreflectVe)
	.def("reflectVeAll", &pyCqcgl1d::PYreflectVeAll)
	.def("ve2slice", &pyCqcgl1d::PYve2slice)
	.def("reduceAllSymmetries", &pyCqcgl1d::PYreduceAllSymmetries)
	.def("reduceIntg", &pyCqcgl1d::PYreduceIntg)
	.def("reduceVe", &pyCqcgl1d::PYreduceVe)
	.def("findReq", &pyCqcgl1d::PYfindReq)
	.def("optThPhi", &pyCqcgl1d::PYoptThPhi)
	.def("transRotate", &pyCqcgl1d::PYtransRotate) 
	.def("transTangent", &pyCqcgl1d::PYtransTangent)
	.def("phaseRotate", &pyCqcgl1d::PYphaseRotate)
	.def("phaseTangent", &pyCqcgl1d::PYphaseTangent)
	.def("Rotate", &pyCqcgl1d::PYRotate)
	.def("rotateOrbit", &pyCqcgl1d::PYrotateOrbit)
	.def("planeWave", &pyCqcgl1d::PYplaneWave)
	.def("powIt", &pyCqcgl1d::PYpowIt)
	.def("powEigE", &pyCqcgl1d::PYpowEigE)
	;

    bp::class_<pyCgl1d, bp::bases<Cgl1d> >("pyCgl1d", bp::init<
					       int, double, double,
					       bool, int,
					       double, double,
					       int >())
	.def_readonly("N", &pyCgl1d::N)
	.def_readonly("d", &pyCgl1d::d)
	.def_readonly("h", &pyCgl1d::h)
	.def_readonly("trueNjacv", &pyCgl1d::trueNjacv)
	.def_readonly("Mu", &pyCgl1d::Mu)
	.def_readonly("Br", &pyCgl1d::Br)
	.def_readonly("Bi", &pyCgl1d::Bi)
	.def_readonly("Dr", &pyCgl1d::Dr)
	.def_readonly("Di", &pyCgl1d::Di)
	.def_readonly("Gr", &pyCgl1d::Gr)
	.def_readonly("Gi", &pyCgl1d::Gi)
	.def_readonly("Ndim", &pyCgl1d::Ndim)
	.def("changeh", &pyCgl1d::PYchangeh)
	.def("velocity", &pyCgl1d::PYvelocity)
	.def("velocityReq", &pyCgl1d::PYvelocityReq)
	.def("pad", &pyCgl1d::PYpad)
	.def("generalPadding", &pyCgl1d::PYgeneralPadding)
	.def("unpad", &pyCgl1d::PYunpad)
	.def("intg", &pyCgl1d::PYintg)
	.def("intgj", &pyCgl1d::PYintgj)
	.def("intgv", &pyCgl1d::PYintgv)
	.def("Fourier2Config", &pyCgl1d::PYFourier2Config)
	.def("Config2Fourier", &pyCgl1d::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCgl1d::PYFourier2ConfigMag)
	.def("orbit2sliceWrap", &pyCgl1d::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCgl1d::PYorbit2slice)
	.def("stab", &pyCgl1d::PYstab)
	.def("stabReq", &pyCgl1d::PYstabReq)
	.def("reflect", &pyCgl1d::PYreflect)
	.def("reduceReflection", &pyCgl1d::PYreduceReflection)
	.def("refGradMat", &pyCgl1d::PYrefGradMat)
	.def("reflectVe", &pyCgl1d::PYreflectVe)
	.def("reflectVeAll", &pyCgl1d::PYreflectVeAll)
	.def("ve2slice", &pyCgl1d::PYve2slice)
	.def("reduceAllSymmetries", &pyCgl1d::PYreduceAllSymmetries)
	.def("reduceIntg", &pyCgl1d::PYreduceIntg)
	.def("reduceVe", &pyCgl1d::PYreduceVe)
	.def("findReq", &pyCgl1d::PYfindReq)
	.def("transRotate", &pyCgl1d::PYtransRotate)
	.def("phaseRotate", &pyCgl1d::PYphaseRotate)
	.def("Rotate", &pyCgl1d::PYRotate)
	.def("rotateOrbit", &pyCgl1d::PYrotateOrbit)
	;

    bp::class_<pyCqcglRPO, bp::bases<CqcglRPO> >("pyCqcglRPO", bp::init<
						 int, int, int, double, double,
						 double, double, double,
						 double, double, double,
						 double, int >())
	.def_readonly("N", &pyCqcglRPO::N)
	.def_readonly("Ndim", &pyCqcglRPO::Ndim)
	.def("findRPO", &pyCqcglRPO::PYfindRPO)
	.def("findRPOM", &pyCqcglRPO::PYfindRPOM)
	;

}
