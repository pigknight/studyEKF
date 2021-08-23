/*

*/
#include <bfl/wrappers/matrix/matrix_wrapper.h>
#include "OdomEstimationEKF.h"
#include <math.h>

using namespace std;

namespace ekf
{

    OdomEstimationEKF::OdomEstimationEKF(int _numOfState = 6, int _numOfMeasurement = 9, float pVal = 0.1, float qVal = 1e-4, float rVal = 0.1)
        : TinyEKF(_numOfState, _numOfMeasurement, pVal, qVal, rVal)
    {
        stateCount = _numOfState; 
        measurementCount = _numOfMeasurement;
        
        /*
        Note:  在robot_pose_ekf中，F_k的设置采用了非线性方程线性化矩阵，即雅各比矩阵。对输入控制求偏导函数
        1。  状态转换方程描述为：
        //进行非线性计算,更新状态
        //近似执行        X_k = F_k * X_k-1 + B_k * u_k
	    //查看系统预测模型更新(update)时,控制向量输入都为方向为vel(1,2)均为0
        //state(1) += cos(state(6)/theta/) * vel(1)/v/; //根据速度v更新x轴位移
        //state(2) += sin(state(6) /theta/) * vel(1) /v/; //根据速度v更新y轴位移
        //state(6) += vel(2);                                 //更新theta,即方向.
        /*2。雅各比矩阵F_k :
        //密度函数矩阵,雅各比矩阵F_k.  对参数1即状态向量X_k(3)求雅各比矩阵
        for (unsigned int i = 1; i <= 6; i++){
            for (unsigned int j = 1; j <= 6; j++){
                if (i == j)  df(i, j) = 1;
                else  df(i, j) = 0;
            }
        }
        df(1, 6) = -vel_trans * sin(yaw); //求x关于yaw的偏导:             x =  vel(1)*cos(state(3))   x` = -vel(1)*sin(state(3))
        df(2, 6) = vel_trans * cos(yaw);  //求y关于yaw的偏导              y =  vel(1)*sin(state(3))   y` = vel(1)*cos(state(3))

        3。查看robot_pose_ekf系统预测模型更新(update)时,控制向量输入都为0,所以这里用对角线为1方阵即可！
        */

        //状态转换矩阵设为I   只是进行odom融合,且没有输入控制u_k（0，0），因为k-1状态到K状态之间没有明确的转换关系, 我们假设为恒等关系
        F_k.resize(stateCount); //矩阵nxn
        F_k = 0;
        for (unsigned int i = 1; i <= stateCount; i++)
            F_k(i, i) = 1;

        H_k.resize(measurementCount, stateCount); //测量值个数为9*6
        H_k = 0;

        /*
        hk = [
            (1, 0, 0, 0, 0, 0),  p_x
            (0, 1, 0, 0, 0, 0),  p_y
            (0, 0, 0, 0, 0, 0),  p_z = 0
            (0, 0, 0, 0, 0, 0),  d_roll = 0
            (0, 0, 0, 0, 0, 0),  d_pitch = 0
            (0, 0, 0, 0, 0, 1),  d_yaw
            (0, 0, 0, 0, 0, 0),  d_roll = 0
            (0, 0, 0, 0, 0, 0),  d_pitch = 0
            (0, 0, 0, 0, 0, 1)   d_yae
            ] */
        
        H_k(1, 1) = 1;
        H_k(2, 2) = 1;
        H_k(6, 6) = 1;
        H_k(9, 6) = 1;
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
    bool OdomEstimationEKF::stateTransitionFunction(ColumnVector &x, ColumnVector &x_new, SymmetricMatrix &f_k)
    {
        //状态转换矩阵设为I   只是进行odom融合,且没有输入控制u_k（0，0），因为k-1状态到K状态之间没有明确的转换关系, 我们假设为恒等关系
        /*SymmetricMatrix _f_k(stateCount); //矩阵nxn
        _f_k = 0;
        for (unsigned int i = 1; i <= stateCount; i++)
            _f_k(i, i) = 1;*/

        f_k = F_k; // 返回一个单位矩阵.

        x_new = F_k * x; // 这里直接返回了当前状态值得相同值

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
    bool OdomEstimationEKF::stateToMeasurementTransitionFunction(ColumnVector &x, ColumnVector &zz_k, Matrix &h_k)
    {
        //Matrix _h_k(measurementCount,stateCount); //测量值个数为9*6
        //_h_k = 0;
        /*
        hk = [
            (1, 0, 0, 0, 0, 0),  p_x
            (0, 1, 0, 0, 0, 0),  p_y
            (0, 0, 0, 0, 0, 0),  p_z = 0
            (0, 0, 0, 0, 0, 0),  d_roll = 0
            (0, 0, 0, 0, 0, 0),  d_pitch = 0
            (0, 0, 0, 0, 0, 1),  d_yaw
            (0, 0, 0, 0, 0, 0),  d_roll = 0
            (0, 0, 0, 0, 0, 0),  d_pitch = 0
            (0, 0, 0, 0, 0, 1)   d_yae
            ] */
        /*for (unsigned int i=1; i<=stateCount; i++)
            _h_k(i,i) =1;*/
        /*
        _h_k(1,1) = 1; 
        _h_k(2,2) = 1; 
        _h_k(6,6) = 1;
        _h_k(7,4) = 1; 
        _h_k(8,5) = 1; 
        _h_k(9,6) = 1;*/

        //返回经状态转换函数变换后的测量值zz_k: H_k(mxn) * X(nx1) = ZZ_k(mx1)
        zz_k = H_k * x;
        h_k = H_k;

        return true;
    }
}