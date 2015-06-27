#include "myfft.hpp"
using namespace Eigen;

long FFT::NumOfInstance = 0;	// initialize the instance number

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
