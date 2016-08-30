/* to comiple:
 * first use
 * mpicxx --showme -O3 test_CQCGL1dReq_mpi.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dReq -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm
 * to get the actual linkages. Then change g++ to h5c++
 *
 * h5c++ -O3 test_CQCGL1dReq_mpi.cc -L../../lib -I../../include -I/usr/local/home/xiong/apps/eigen/include/eigen3 -std=c++11 -lCQCGL1dReq -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -pthread -L/usr//lib -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl -lhwloc 
 *
 * execute : mpiexec -np 4 ./a.out
 */
#include "CQCGL1dReq.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>
#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

typedef std::complex<double> dcp;


#define N30

int main(int argc, char **argv){

#ifdef N10
    //======================================================================
    // extend the soliton solution in the Bi-Gi plane
    iterMethod::LM_OUT_PRINT = false;
    iterMethod::LM_IN_PRINT = false;
    iterMethod::CG_PRINT = false;

    const int N = 1024;
    const int L = 50;
    double Bi = 3.4;
    double Gi = -5.6;

    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    string fileName = "../../data/cgl/reqBiGi";

    double stepB = 0.1;
    int NsB = 24;

    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = NsB / num;
    int rem = NsB - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////
    H5File file(fileName + "_" + to_string(rank) + ".h5", H5F_ACC_RDWR);

    int ids[] = {1, 2};
    for (int i = 0; i < 2; i++){	
	int id = ids[i];
	// cgl.findReqParaSeq(file, id, stepB, NsB, true);
	for (int i = p_start; i < p_end; i++){
	    cgl.Bi = Bi+i*stepB;
	    cgl.findReqParaSeq(file, id, 0.1, 53, false);
	}
    }
    
    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////

#endif
#ifdef N20
    //======================================================================
    // try to calculate the eigenvalue and eigenvector
    const int N = 1024;
    const int L = 50;
    double Bi = 5.7;
    double Gi = -5.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    
    // string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGiEV.h5";
    string fileName = "../../data/cgl/reqBiGiEV_add_";
    ArrayXd a0;
    double wth0, wphi0, err0;

    VectorXcd e;
    MatrixXcd v;
    
    std::vector<double> Bis, Gis;
    for(int i = 0; i < 55; i++) Gis.push_back(Gi+0.1*i);
    
    int NsB = 23;
    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = NsB / num;
    int rem = NsB - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    for (int i = p_start; i < p_end; i++) Bis.push_back(Bi-0.1*i);
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////
    
    H5File file(fileName + to_string(rank) + ".h5", H5F_ACC_RDWR);
    cgl.calEVParaSeq(file, std::vector<int>{1, 2}, Bis, Gis, true);

    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////

#endif
#ifdef N30
    //======================================================================
    // try to calculate the eigenvalue and eigenvector. Parallel for Gi.
    const int N = 1024;
    const int L = 50;
    double Bi = 3.6;
    double Gi = -5.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    
    // string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGiEV.h5";
    string fileName = "../../data/cgl/reqBiGiEV_add_";
    ArrayXd a0;
    double wth0, wphi0, err0;

    VectorXcd e;
    MatrixXcd v;
    
    std::vector<double> Bis, Gis;
    Bis.push_back(Bi);
    
    int NsG = 55;
    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = NsG / num;
    int rem = NsG - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    for (int i = p_start; i < p_end; i++) Gis.push_back(Gi+0.1*i);
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////
    
    H5File file(fileName + to_string(rank) + ".h5", H5F_ACC_RDWR);
    cgl.calEVParaSeq(file, std::vector<int>{1, 2}, Bis, Gis, true);

    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////

#endif
    
    return 0;
}
