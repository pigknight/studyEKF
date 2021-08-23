#ifndef __ODOM_ESTIMATION_EKF_INCLUDE__
#define __ODOM_ESTIMATION_EKF_INCLUDE__

#include "TinyEKF.h"

namespace ekf
{

    class OdomEstimationEKF : public TinyEKF
    {
    private:
        int stateCount = 1;
        int measurementCount = 1;

        SymmetricMatrix F_k;
        Matrix H_k;

    public:
        OdomEstimationEKF(int _numOfState, int _numOfMeasurement, float pVal, float qVal, float rVal);
        ~OdomEstimationEKF(){};

        bool stateTransitionFunction(ColumnVector &x, ColumnVector &x_new, SymmetricMatrix &f_k);
        bool stateToMeasurementTransitionFunction(ColumnVector &x, ColumnVector &zz_k, Matrix &h_k);
    };

}
#endif