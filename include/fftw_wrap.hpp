/**
 * Note :
 *
 * 1) the "many" scheme assumes performing FFT in each column of input matrix. Eigen is
 * column-wise in memory layout, while FFTW is assuming row-wise memory layout, so
 * caution is exerted there.
 * 
 * 2)   
 */
#ifndef FFTW_WRAP_H
#define FFTW_WRAP_H

#include <fftw3.h>
#include <complex>
#include <Eigen/Dense>

namespace FFTW_wrap {
    
    using namespace std;
    using namespace Eigen;

    //////////////////////////////////////////////////////////////////////////////////////////
    //                                   1d  FFT                                            //
    //////////////////////////////////////////////////////////////////////////////////////////
    class fft {
    public:
	const int N, M;
	MatrixXcd time, freq;
	fftw_plan p, ip;
	
	fft(const int N, const int M, bool doInit = true) : N(N), M(M) {
	    if (doInit) init();
	}
	
	~fft(){
	    if(p) fftw_destroy_plan(p);
	    if(ip) fftw_destroy_plan(ip);
	}

	inline void init(){
	    assert (M > 0);

	    time.resize(N, M);
	    freq.resize(N, M);

	    if (1 == M){
		p = fftw_plan_dft_1d(N, (fftw_complex*)time.data(), (fftw_complex*)freq.data(),
				     FFTW_FORWARD, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
		ip = fftw_plan_dft_1d(N, (fftw_complex*)freq.data(), (fftw_complex*)time.data(),
				      FFTW_BACKWARD, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    }
	    else{
		int n[] = { N };
		p = fftw_plan_many_dft(1, n, M, (fftw_complex*)time.data(), 
				       n, 1, N, (fftw_complex*)freq.data(), n, 1, N,
				       FFTW_FORWARD, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
		ip = fftw_plan_many_dft(1, n, M, (fftw_complex*)freq.data(),
					n, 1, N, (fftw_complex*)time.data(), n, 1, N,
					FFTW_BACKWARD, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    }
	}
	
	inline void fwd(){
	    fftw_execute(p);  
	}
	
	inline void inv(){
	    fftw_execute(ip);
	    time *= 1.0/N;
	}
	
    };

    //////////////////////////////////////////////////////////////////////////////////////////
    //                                   1d  real FFT                                       //
    //////////////////////////////////////////////////////////////////////////////////////////
    class rfft {
    public:
	const int N, M;
	MatrixXd time;
	MatrixXcd freq;
	fftw_plan p, ip;
	
	fft(const int N, const int M, bool doInit = true) : N(N), M(M) {
	    if (doInit) init();
	}
	
	~fft(){
	    if(p) fftw_destroy_plan(p);
	    if(ip) fftw_destroy_plan(ip);
	}

	inline void init(){
	    assert (M > 0);

	    time.resize(N, M);
	    freq.resize(N/2+1, M);

	    if (1 == M){
		p = fftw_plan_dft_r2c_1d(N, time.data(), (fftw_complex*)freq.data(), FFTW_MEASURE|FFTW_PRESERVE_INPUT);
		ip = fftw_plan_dft_c2r_1d(N, (fftw_complex*)freq.data(), time.data(), FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    }
	    else{
		int n[] = { N };
		p = fftw_plan_many_dft_r2c(1, n, M, time.data(), n, 1, N,
					   (fftw_complex*)freq.data(), n, 1, N/2+1,
					   FFTW_MEASURE|FFTW_PRESERVE_INPUT);
		ip = fftw_plan_many_dft_r2c(1, n, M, (fftw_complex*)freq.data(), n, 1, N/2+1,
					    time.data(), n, 1, N,
					    FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    }
	}
	
	inline void fwd(){
	    fftw_execute(p);  
	}
	
	inline void inv(){
	    fftw_execute(ip);
	    time *= 1.0/N;
	}
	
    }
	
}

#endif	/* FFTW_WRAP_H */
