/** \mainpage trail the fftw routines for my purpose 
 *
 *  \section sec_intro Introduction
 *
 *  \section sec_use usage
 *  Example:
 *  \code
 *  g++ yourfile.cc -I/path/to/fft.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 */

#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include <complex>
#include <Eigen/Dense>

/** @brief Structure for convenience of rfft.   */  
class FFT {
    
protected:
    static long NumOfInstance;
    fftw_plan p, rp; // plan for fft/ifft.
    fftw_complex *c1, *c2, *c3; // c1 = v, c2 = ifft(v), c3 = fft(g(ifft(v)))
    
public:
    
    typedef std::complex<double> dcp;
    
    const int N;
    const int M;
    const int threadNum;
    
    Eigen::Map<Eigen::ArrayXXcd> v1, v2, v3; // mapping for c1, c2, c3 respectively

    //////////////////////////////////////////////////////////////////////
    //                      member functions                            //
    //////////////////////////////////////////////////////////////////////

    FFT(const int N, const int M, const int threadNum = 4);
    ~FFT();    
    void fft();    
    void ifft();    
    
};




#endif	/* FFT_H */
