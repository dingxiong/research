#ifndef MYBOOSTPYTHON_H
#define MYBOOSTPYTHON_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;

typedef std::complex<double> dcp;

// ============================================================

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
template<class MyAry = ArrayXXd, class MyType = double>
bn::ndarray copy2bn(const Ref<const MyAry> &x){
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
    bn::ndarray px = bn::empty(ndim, dims, bn::dtype::get_builtin<MyType>());
    memcpy((void*)px.get_data(), (void*)x.data(), sizeof(MyType) * m * n);
	    
    return px;
}

/* non-template function overloading for real matrix */
inline bn::ndarray copy2bn(const Ref<const ArrayXXd> &x) {
    return copy2bn<ArrayXXd, double>(x);
}

/* non-template function overloading for imaginary matrix */
inline bn::ndarray copy2bnc(const Ref<const ArrayXXcd> &x) {
    return copy2bn<ArrayXXcd, dcp>(x);
}


/**
 * @brief std::vector to bp::list
 */
template <class T>
inline bp::list toList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
	list.append(*iter);
    }
    return list;
}


#endif /* MYBOOSTPYTHON_H */
