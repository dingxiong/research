#ifndef KSSOLVER_H
#define KSSOLVER_H

const int N = 32; // truncation number.

#ifdef __cplusplus
extern "C"{
#endif

/********         function declaration               ***********/

/**
   @brief KS solver without calculating Jacobian matrix.
   
   @param[in] a0 initial condition, size N-2 array
   @param[in] d the lenth of the KS system
   @param[in] h time step
   @param[in] nstp number of steps to be integrated
   @param[in] np state saving spacing.
   @param[out] aa saved state vector size = (nstp/np)*(N-2)
   eg. if state column vector is v0, v1, ... vn-1, then
   aa is a row vector [ v0^{T}, v1^{T}, ... vn-1^{T}].
*/
  void
  ksf(double *a0, double d, double h, int nstp, int np, double *aa);

/**
   @brief KS solver with calculating Jacobian (size (N-2)*(N-2)).

   @param[in] nqr Jacobian saving spacing spacing
   @param[out] daa saved Jacobian matrix. size = (nstp/nqr)*(N-2)*(N-2).
               eg. If Jacobian matrix is J=[v1, v2,..., vn] each vi is a
               column vector,  then 
               daa is a row vector [vec(J1), vec(J2), vec(Jn)], where
               vec(J)= [v1^{T}, v2^{T},...,vn^{T}] with each element
               visted column-wise.
*/
  void
  ksfj(double *a0, double d, double h, int nstp, int np, int nqr, double *aa, double *daa);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* KSSOLVER_H */
