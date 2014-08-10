#ifndef KSSYM_H
#define KSSYM_H

#include <eigen3/Eigen/Dense>
using Eigen::MatrixXd; 
using Eigen::Matrix2d; 

class Kssym {
  
public:
  const int px; 

  /** @brief SO2 reduction of a sequence of column vectors stored in aa.
   *
   *  @param[in] aa matrix storing the vectors
   *  @param[out] ang pointer to the theta array. If you do not need
   *                  angle information, then just pass NULL to angle
   *  @return the group transformed vectors 
   * */
  MatrixXd redSO2(const MatrixXd &aa, double *ang);
  
  /** @brief calculate g(th) * aa
   *
   *  @param[in] aa [n,m] matrix
   *  @param[in] th SO2 group angle
   *  @return rotated matrix
   */
  MatrixXd rotate(const MatrixXd &aa, const double th);
  
  /* ------------   constructor/ destructor  ----------------- */
  Kssym(int p = 1): px(p) {}
  Kssym(const Kssym &k): px(k.px) {}
  Kssym & operator=(const Kssym &x ){ return *this; }
  ~Kssym(){}
};

#endif	/* KSSYM_H */
