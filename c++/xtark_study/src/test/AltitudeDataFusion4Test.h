#ifndef __ALTITUDE_DATE_FUSION_4_TEST_INCLUDE__
#define __ALTITUDE_DATE_FUSION_4_TEST_INCLUDE__

#include "TinyEKF.h"

namespace ekf
{

class AltitudeDataFusion :  public TinyEKF
{
private:
    int stateCount = 1;
    int measurementCount = 1;
    int interval = 10;   // 预测更新时间间隔,单位 ms

    float last_state_altitude = 0.0; //上次状态值：海拔高度值   做非线性函数线性化,用于计算连续差分
    float last_measure_barometers = 0.0; //上次测量气压值
public:
    AltitudeDataFusion(int _numOfState, int _numOfMeasurement, float pVal, float qVal, float rVal,int _interval);
    ~AltitudeDataFusion(){};

    bool stateTransitionFunction(ColumnVector& x, ColumnVector& x_new, SymmetricMatrix& f_k);
    bool stateToMeasurementTransitionFunction(ColumnVector& x, ColumnVector& zz_k, Matrix& h_k);
};

}
#endif