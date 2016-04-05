#include "myfft.hpp"
#include <iostream>
using namespace Eigen;
using namespace std;

namespace MyFFT {
    
    long NumOfInstance = 0;

    //////////////////////////////////////////////////////////////////////////////////////////
    //                                     FFT                                              //
    //////////////////////////////////////////////////////////////////////////////////////////

    FFT::FFT(const int N, const int M, const int threadNum) : 
	N(N), M(M), threadNum(threadNum),
	v1(NULL, 0, 0), v2(NULL, 0, 0), v3(NULL, 0, 0)
    {
	// only the first instance do some initialization
	if(++NumOfInstance == 1){
#ifdef TFFT  // mutlithread fft.
	    if(!fftw_init_threads()){
		fprintf(stderr, "error create MultiFFT.\n");
		exit(1);
	    }
	    // fftw_plan_with_nthreads(omp_get_max_threads());
	    fftw_plan_with_nthreads(threadNum);    
#endif	/* TFFT */
	}

	if(M > 0){
	    // initialize fft/ifft plan
	    c1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	    c2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	    c3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	    //build the maps.
	    new (&v1) Eigen::Map<Eigen::ArrayXXcd>( (dcp*)&(c1[0][0]), N, M );
	    new (&v2) Eigen::Map<Eigen::ArrayXXcd>( (dcp*)&(c2[0][0]), N, M );
	    new (&v3) Eigen::Map<Eigen::ArrayXXcd>( (dcp*)&(c3[0][0]), N, M );

	    if (1 == M){
		p = fftw_plan_dft_1d(N, c2, c3, FFTW_FORWARD, FFTW_MEASURE);
		rp = fftw_plan_dft_1d(N, c1, c2, FFTW_BACKWARD, FFTW_MEASURE);
	    } else{
		int n[] = { N };
		p = fftw_plan_many_dft(1, n, M, c2, n, 1, N,
				       c3, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
		rp = fftw_plan_many_dft(1, n, M, c1, n, 1, N,
					c2, n, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
	    }
	}
	
    }

    FFT::~FFT(){

	if(M > 0){
	    fftw_destroy_plan(p);
	    fftw_destroy_plan(rp);
	    fftw_free(c1);
	    fftw_free(c2);
	    fftw_free(c3);
	    /* releae the map */
	    new (&(v1)) Eigen::Map<Eigen::ArrayXXcd>(NULL, 0, 0);
	    new (&(v2)) Eigen::Map<Eigen::ArrayXXcd>(NULL, 0, 0);
	    new (&(v3)) Eigen::Map<Eigen::ArrayXXcd>(NULL, 0, 0);
	}
	
	// only the last instance cleans up
	if(--NumOfInstance == 0){
#ifdef CLEAN
	    fftw_cleanup();
#endif	/* CLEAN */
#ifdef TFFT
	    fftw_cleanup_threads();
#endif	/* TFFT */
	}
	

    }

    void FFT::fft() {
	fftw_execute(p);  
    }

    void FFT::ifft() {
	fftw_execute(rp);
	v2 /= N;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    //                                    RFFT                                              //
    //////////////////////////////////////////////////////////////////////////////////////////
    RFFT::RFFT(const int N, const int M, const int threadNum) : 
	N(N), M(M), threadNum(threadNum),
	vc1(NULL, 0, 0), vr2(NULL, 0, 0), vc3(NULL, 0, 0) 
    {
	if(++NumOfInstance == 1) {
#ifdef TFFT  // mutlithread fft.
	    if(!fftw_init_threads()){
		fprintf(stderr, "error create MultiFFT.\n");
		exit(1);
	    }
	    // fftw_plan_with_nthreads(omp_get_max_threads());
	    fftw_plan_with_nthreads(threadNum);    
#endif	/* TFFT */
	}
	    
	if(M > 0) {
	    r2 = (double*) fftw_malloc(sizeof(double) * N * M);
	    c1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1) * M);
	    c3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1) * M);
	    
	    new (&vr2) Map<ArrayXXd>( &(r2[0]), N, M);
	    new (&vc1) Map<ArrayXXcd>( (dcp*)&(c1[0][0]), N/2+1, M );
	    new (&vc3) Map<ArrayXXcd>( (dcp*)&(c3[0][0]), N/2+1, M );
	    
	    if (1 == M){
		p = fftw_plan_dft_r2c_1d(N, r2, c3, FFTW_MEASURE);
		rp = fftw_plan_dft_c2r_1d(N, c1, r2, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    } else{
		int n[]={N};
		p = fftw_plan_many_dft_r2c(1, n, N-1, r2, n, 1, N, 
					   c3, n, 1, N/2+1, FFTW_MEASURE);
		rp = fftw_plan_many_dft_c2r(1, n, N-1, c1, n, 1, N/2+1,
					    r2, n, 1, N, FFTW_MEASURE|FFTW_PRESERVE_INPUT);
	    }
	}

    }

    RFFT::~RFFT(){
	
	if(M > 0){	    
	    /* free the memory */
	    fftw_destroy_plan(p);
	    fftw_destroy_plan(rp);
	    fftw_free(c1);
	    fftw_free(r2);
	    fftw_free(c3);
	    /* release the maps */
	    new (&(vc1)) Map<ArrayXXcd>(NULL, 0, 0);
	    new (&(vr2)) Map<ArrayXXd>(NULL, 0, 0);
	    new (&(vc3)) Map<ArrayXXcd>(NULL, 0, 0);	   
	}
	
	if(--NumOfInstance == 0 ){
#ifdef CLEAN
	    fftw_cleanup();
#endif	/* CLEAN */
#ifdef TFFT
	    fftw_cleanup_threads();
#endif	/* TFFT */
	}

    }
    
    void RFFT::fft() {
	fftw_execute(p);
    }

    void RFFT::ifft() { //cout << vc1 << endl << endl;
	fftw_execute(rp); // cout << vr2 << endl;
	vr2 /= N;
    }	
    
}
