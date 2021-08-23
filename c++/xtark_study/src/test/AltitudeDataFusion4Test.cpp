/*
AltitudeDataFusion 自己实现的一个基于EKF的海拔高度数据融合算法,

算法中模拟一个飞机飞行中,确定自己的海拔高度: state_x( x_altitude ), 有且仅有此一个状态
飞机上安装有三个海拔高度传感器:
1.GPS定位系统,可以读取一个带有噪声的海拔高度值: v_gps 单位: 米;
2.气压计(barometers), 可以读取一个带有噪声的受温度和高度影响的气压值: v_barometers 单位: kPa;
3.IMU惯性传感器,可以利用内部差分累加算法获取一个带有噪音的高度值: v_imu 单位: 米.

****************************************************************************
实际工作中往往是根据测得的温度和压强垂直分布去计算位势高度，因几何高度难以直接测量。
大气压同海拔高度的关系：
　　P=P0×（1-H/44300)^5.256
计算高度公式为:
   H=44300*(1- (P/P0)^(1/5.256) )
式中：H——海拔高度，P0=大气压（0℃，101.325kPa）
**************************************************************************

Note:  以下: 当前时刻: cur or k-1   下一时刻: next or k
1.状态向量定义:
根据上述,系统有1个状态向量state_x( x ),即: X(x);
估计值向量: X = [ x ]
估计值协方差矩阵: P =[ Cov(x,x) ]    # Note:  Cov(x,y)参见协方差公式

2.根据数据变换关系分析出预测方程:
x_next = x_cur  #--->  [ 1 ]   默认状态保持不变,

用矩阵表示估计向量
(1) X_next = [ 1 ]  * X_cur  = F_k * X_cur
上面,预测矩阵 或 估计矩阵 或 状态转移矩阵:
F_k = [ 1 ]
根据协方差推到公式: Cov(Ax) = A * Cov(x) * A_T   依据此,就可以更新估计值协方差矩阵P
(1) X_next = F_k * X_cur                  # 状态向量 方程
(2) P_next = F_k * P_cur * F_k_T          # 估计值协方差矩阵 方程
因为机器人行进过程中,会受到外部噪声影响,添加噪声协方差矩阵Q_k后:
#根据上一时刻系统状态和当前时刻系统控制量所得到的系统估计值,该估计值又叫做先验估计值,为各变量高斯分布的均值.
(1) X_next = F_k * X_cur               # 状态向量预测方程,先验估计值,为各变量高斯分布的均值
#协方差矩阵,代表了不确定性,它是由上一时刻的协方差矩阵和外部噪声的协方差一起计算得到的.
(2) P_next = F_k * P_cur * F_k_T  + Q_k        # 估计值协方差矩阵 方程
这两个公式代表了卡尔曼滤波器中的预测部分,是卡尔曼滤波器五个基础公式中的前两个.
(3),(4),(5)详见tinyekf包


重点是: 状态值与测量值之间的函数关系,及其雅各比矩阵
ZZ_k(mx1) = H_k(mxn) * X(nx1)
          = [
             x <- f(v_gps)=v_gps   这是个线性计算符合高斯分布
             x <- f(v_barometers) = 这是个非线性计算?  线性?
             x <- f(v_imu)=v_imu  这是个线性计算符合高斯分布
            ] * [ x]
这里f(v_barometers)是什么?
根据大气压和海拔公式有:
H = f(v_barometers) = 44300*(1- (P/P0)^(1/5.256) )    (P0=大气压（0℃，101.325kPa）)
f(v_barometers)无法直接便是为: ZZ_k(mx1) = H_k(mxn) * X(nx1)形式,
故,采取z_k和x_k的连续差分表示:
  z_k = P = P0×（1-x_k/44300)^5.256     (P0=大气压（0℃，101.325kPa）)

  H_k(mxn) = [
             1
             (z_k - z_k-1) / (x_k - x_k-1)
             1
            ]
*/
#include <bfl/wrappers/matrix/matrix_wrapper.h>
#include "AltitudeDataFusion4Test.h"
#include <math.h>

using namespace std;

namespace ekf
{

    AltitudeDataFusion::AltitudeDataFusion(int _numOfState, int _numOfMeasurement, float pVal = 0.1, float qVal = 1e-4, float rVal = 0.1, int _interval = 10)
        : TinyEKF(_numOfState, _numOfMeasurement, pVal, qVal, rVal)
    {
        stateCount = _numOfState;
        measurementCount = _numOfMeasurement;
        interval = _interval; // 预测更新时间间隔,单位 ms

        last_state_altitude = 0.0;     //上次状态值： 海拔高度值   做非线性函数线性化,用于计算连续差分
        last_measure_barometers = 0.0; //上次测量气压值
    }

    /*
    根据系统运动学方程或几何学方程,更新预测状态向量(nx1)
    # (1). X_k = F_k * X_k-1 + B_k * u_k
    # F_k: 根据系统模型设计的预测矩阵或状态变量转移矩阵nxn
    # B_k: 系统外部控制矩阵
    # u_k: 系统外部控制向量,例如加速度,转向等.
    # 这里注意: B_k * u_k的结果维数要保持和X_k同样的维数,以便做矩阵相加
        
    参数x是含有n个元素的当前(k-1时刻)状态值(Current state)向量,
    返回:
    X_k: 依据方程(1)计算的含有n个元素的新的(k)预测状态向量(nx1)
    F_k: 一个依据系统运动学方程或几何学方程设计的预测状态变换模型(n*n矩阵)
    */
    bool AltitudeDataFusion::stateTransitionFunction(ColumnVector &x, ColumnVector &x_new, SymmetricMatrix &f_k)
    {
        //状态转换矩阵设为I 因为k-1状态到K状态之间没有明确的转换关系, 我们假设为恒等关系
        SymmetricMatrix _f_k(stateCount); //矩阵nxn
        _f_k = 0;
        for (unsigned int i = 1; i <= stateCount; i++)
            _f_k(i, i) = 1;

        f_k = _f_k; // 返回一个单位矩阵.

        x_new = _f_k * x; // 这里直接返回了当前状态值得相同值

        //cout<<"stateTransitionFunction:   "<<"f_k="<<f_k<<",  x_new="<<x_new<<"\n";

        return true;
    }

    /*
    预测值状态变换函数(即测量值转状态值得反函数),把预测值状态向量转换为与测量值向量同样的单位或度量,
    必要时需要非线性函数线性化(返回雅各比矩阵)
    方程(4). X^_k = X_k + K_k * (z_k - zz)     # zz = H_k * X_k
    
    参数x是含有n个元素的当前(k-1时刻)状态值(Current state)向量,
    返回:
    1. 预估测量值向量ZZ_K: 一个m个元素的预估测量值向量: 经 经雅各比矩阵 * 当前状态值向量 得到的 "预估测量值向量"
    2. H_k: 一个m*n雅各比矩阵H_k,矩阵的元素为: 状态值转换为观测值的非线性函数的一阶导数,或有限差分, 或连续差分的比值
    m为测量值个数, n为状态量个数, 用处1: H_k(mxn) * X(nx1) = ZZ_k(mx1),以便下一步方程(4)执行: Z_k - ZZ_K
    例如，您的函数可能包含一个将气压转换为以米为单位的海拔高度的组件,或者需要非线性函数线性化。
    
    Note: 
    1.一般情况下,如果不涉及到单位,度量或线性变换. 直接返回(m,n)矩阵: 对角线的地方为1，其余的地方为0.
    2.需要做基本的单位,度量或线性变换的,返回一个固定的(m,n)变换矩阵
    3.特殊情况,如果需要做非线性函数线性化的,需要返回一个元素为连续差分((z_k2 - z_k1) / (x_k2-x_k1))的(m,n)变换矩阵
    */
    bool AltitudeDataFusion::stateToMeasurementTransitionFunction(ColumnVector &x, ColumnVector &zz_k, Matrix &h_k)
    {
        /*
        大气压同海拔高度的关系：
        　　P=P0×（1-H/44300)^5.256
        计算高度公式为:
           H=44300*(1- (P/P0)^(1/5.256) )
        式中：H——海拔高度，P0=大气压（0℃，101.325kPa）
        */

        //基于大气压值: 把海拔高度估计值x,转换为大气压值,以便跟大气压的测量值做公式(4)中差运算
        float altitude = x[1]; //状态向量(1) only one
        float z_barometers = 101.325 * pow((1 - altitude / 44300), 5.256);
        float finite_difference = 1; // 默认变化率为1,  类似于 y = f(x) = x
        if (last_state_altitude != 0 && last_measure_barometers != 0 && (last_state_altitude - altitude) != 0)
            finite_difference = (z_barometers - last_measure_barometers) / (altitude - last_state_altitude);

        Matrix _h_k(measurementCount, stateCount); //测量值个数为3,  3行,1列
        _h_k = 0;
        _h_k(1, 1) = 1;
        _h_k(2, 1) = finite_difference;
        _h_k(3, 1) = 1;

        //返回经状态转换函数变换后的测量值zz_k: H_k(mxn) * X(nx1) = ZZ_k(mx1)
        zz_k = _h_k * x;
        h_k = _h_k;

        last_state_altitude = altitude;
        last_measure_barometers = z_barometers;

        return true;
    }

}