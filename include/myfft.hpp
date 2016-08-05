/** \mainpage trail the fftw routines for my purpose 
 *
 *  \section sec_intro Introduction
 *
 *  \section sec_use usage
 *  Example:
 *  \code
 *  g++ yourfile.cc -I/path/to/fft.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 *
 *  \note Also use shared_ptr of FFT, not define it directly.
 */

#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include <complex>
#include <Eigen/Dense>

namespace MyFFT {

    extern long NumOfInstance;
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //                                     FFT                                              //
    //////////////////////////////////////////////////////////////////////////////////////////
    /** @brief Structure for convenience of fft.   */  
    class FFT {
    
    protected:
	fftw_plan p, rp; // plan for fft/ifft.
	fftw_complex *c1, *c2, *c3; // c1 = v, c2 = ifft(v), c3 = fft(c2)
    
    public:
    
	typedef std::complex<double> dcp;
	
	int N;
	int M;
	bool hasInit = false;
	
	Eigen::Map<Eigen::ArrayXXcd> v1, v2, v3; // mapping for c1, c2, c3 respectively
	
	//////////////////////////////////////////////////////////////////////
	// member functions	
	FFT(const int N, const int M);
	FFT();
	~FFT();
	void init(const int N, const int M);

	void fft();
	void ifft();
	
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //                                    RFFT                                              //
    //////////////////////////////////////////////////////////////////////////////////////////
    class RFFT {

    public:
	typedef std::complex<double> dcp;
	
	const int N;
	const int M;
	const int threadNum;
	
	Eigen::Map<Eigen::ArrayXXd> vr2;
	Eigen::Map<Eigen::ArrayXXcd> vc1, vc3;
	
	fftw_plan p, rp;
	fftw_complex *c1, *c3;
	double *r2;
	
	////////////////////////////////////////////////////////////
	RFFT(const int N, const int M, const int threadNum = 4);
	~RFFT();

	void fft();
	void ifft();

    };

    //////////////////////////////////////////////////////////////////////////////////////////
    //                                     FFT2d                                            //
    //////////////////////////////////////////////////////////////////////////////////////////
    class FFT2d {
    
    public:
	typedef std::complex<double> dcp;

	const int N;
	const int M;
	const int threadNum;

	fftw_plan p, rp; // plan for fft/ifft.
	fftw_complex *c1, *c2, *c3; // c1 = v, c2 = ifft(v), c3 = fft(c2)
    
	Eigen::Map<Eigen::ArrayXXcd> v1, v2, v3; // mapping for c1, c2, c3 respectively

	//////////////////////////////////////////////////////////////////////
	// member functions                           

	FFT2d(const int N, const int M, const int threadNum = 4);
	~FFT2d();    
	void fft();    
	void ifft();    
	
    };
    

}



#endif	/* FFT_H */
