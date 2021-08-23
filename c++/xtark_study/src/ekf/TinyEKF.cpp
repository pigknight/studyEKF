#include "TinyEKF.h"

using namespace std;

namespace ekf
{
    // constructor
    TinyEKF::TinyEKF(int _numOfState, int _numOfMeasurement, float pVal = 0.1, float qVal = 1e-4, float rVal = 0.1) : numOfState(_numOfState),
                                                                                                                      numOfMeasurement(_numOfMeasurement)
    {
        X_k.resize(_numOfState);
        X_k = 0;

        Q_k.resize(_numOfState, true);
        Q_k = 0;
        for (unsigned int i = 1; i <= _numOfState; i++)
            Q_k(i, i) = 1;
        Q_k *= qVal;

        P_current.resize(_numOfState, true);
        P_current = 0;

        P_result.resize(_numOfState, true);
        P_result = 0;
        for (unsigned int i = 1; i <= _numOfState; i++)
            P_result(i, i) = 1;
        P_result *= pVal;

        R_k.resize(_numOfMeasurement, true);
        R_k = 0;
        for (unsigned int i = 1; i <= _numOfMeasurement; i++)
            R_k(i, i) = 1;
        R_k *= rVal;

        Identity.resize(_numOfState, true);
        Identity = 0;
        for (unsigned int i = 1; i <= _numOfState; i++)
            Identity(i, i) = 1;
    }

    // destructor
    TinyEKF::~TinyEKF()
    {
    }

    void TinyEKF::setInitState(ColumnVector& _x_k, SymmetricMatrix& _p_k)
    {
        for (unsigned int i = 1; i <= numOfState; i++)
            X_k(i) = _x_k(i);
        for (unsigned int i = 1; i <= 6; i++)
            for (unsigned int j = 1; j <= 6; j++)
                P_result(i, j) = _p_k(i, j);
    }

    void TinyEKF::updateRk(SymmetricMatrix &newRk)
    {
        for (unsigned int i = 1; i <= numOfMeasurement; i++)
            for (unsigned int j = 1; j <= numOfMeasurement; j++)
                R_k(i, j) = newRk(i, j);
    }

    void TinyEKF::setAdditiveMeasureNoiseRk(SymmetricMatrix &noiseMatrix /*measurementCount * measurementCount*/)
    {
        for (unsigned int i = 1; i <= numOfMeasurement; i++)
            for (unsigned int j = 1; j <= numOfMeasurement; j++)
                R_k(i, j) = noiseMatrix(i, j);
    }

    void TinyEKF::getAdditiveMeasureNoiseRk(SymmetricMatrix &noiseMatrix /*measurementCount * measurementCount*/)
    {
        for (unsigned int i = 1; i <= numOfMeasurement; i++)
            for (unsigned int j = 1; j <= numOfMeasurement; j++)
                noiseMatrix(i, j) = R_k(i, j);
    }

    void TinyEKF::getPostCovariance(SymmetricMatrix &matrix)
    {
        for (unsigned int i = 1; i <= 6; i++)
            for (unsigned int j = 1; j <= 6; j++)
                matrix(i, j) = P_result(i, j);
    }

    void TinyEKF::doStep(ColumnVector &z, ColumnVector &x_k)
    {
        //Predict ----------------------------------------------------

        /*  根据系统运动学方程或几何学方程,更新预测状态
    参数: 
        x 前一时刻(k-1)状态向量
    返回:
        x: 新时刻(k)状态向量
        F_K:一个针对新状态的雅各比行列式(n*n矩阵),矩阵的元素为: 状态值和观测值之间非线性函数的一阶导数,或有限差分, 或连续差分的比值
    */
        /*# 预测状态方程
    # (1). X_k = F_k * X_k-1 + B_k * u_k
    # F_k: 根据系统模型设计的预测矩阵或状态变量转移矩阵nxn
    # B_k: 系统外部控制矩阵 维数确保可以和前面F_k相乘
    # u_k: 系统外部控制向量,例如加速度,转向等.  维数确保可以和前面F_k相乘
    */
        ColumnVector x_k_new(numOfState);
        SymmetricMatrix F_k(numOfState);
        stateTransitionFunction(X_k, x_k_new, F_k);

        /*# 预测协方差矩阵
    # (2). P_k = F_k * P_k-1 * F_k^T + Q_k
    # Q_k: 各状态量对应的噪音协方差矩阵nxn
    */
        P_current = (SymmetricMatrix)(((SymmetricMatrix)(F_k * P_result)) * F_k.transpose() + Q_k); //P_k

        //Update -----------------------------------------------------
        /*# 预测值状态转测量值变换函数(即测量值转状态值得反函数)矩阵,把预测值状态向量转换为与测量值向量同样的单位或度量,
    # 必要时需要非线性函数线性化(返回雅各比矩阵)
    # 1. 一个m个元素的预估测量值向量: 经经雅各比矩阵 * 当前状态值向量得到的 "预估测量值向量"
    # H_k: 预测值状态转测量值变换函数的雅各比矩阵
    */
        ColumnVector zz_k(numOfMeasurement);
        Matrix H_k(numOfMeasurement, numOfState);
        stateToMeasurementTransitionFunction(x_k_new, zz_k, H_k);

        /* 卡尔曼增益: K_k
    # (3). K_k = P_k * H_k^T * (H_k * P_k * H_k^T + R_k)^-1
    # R_k: 传感器噪声(不确定性)协方差矩阵mxm
    */
        //Note:  这里建议都采用Matrix类型，因为掺杂着SymmetricMatrix类型会使对矩阵就逆矩阵（inverse()）时，运算结果不正确
        Matrix K_k;
        Matrix tmp1 = (Matrix)((Matrix)P_current * H_k.transpose());
        Matrix tmp2 = (Matrix)(H_k * (Matrix)P_current);
        Matrix tmp3 = (Matrix)(tmp2 * H_k.transpose());
#if true //分步计算
        Matrix tmp31 = (Matrix)(tmp3 + R_k);
        Matrix tmp4 = tmp31.inverse();
        K_k = tmp1 * (Matrix)tmp4;
#else
        K_k = (Matrix)(tmp1 * (Matrix)(((Matrix)(tmp3 + R_k)).inverse()));
#endif

        /*# 最终,最优预测状态向量值
    # z_k: 传感器读数或均值
    # (4). X^_k = X_k + K_k * (z_k - H_k * X_k)
    # self.x += np.dot(K_k, (np.array(z) - H_k.dot(self.x)))
    */

        /*# zz_k: 由状态转测量值函数stateToMeasurementTransitionFunction返回的值
    # (4). X^_k = X_k + K_k * (z_k - zz_k)    # Note:  此处的zz_k = H_k * X_k)
    */
        X_k = x_k_new + (K_k * (z - zz_k));

        /*# 最后,最优预测协方差矩阵
    # (5). P^_k = P_k - K_k * H_k * P_k = (I - K_k * H_k) * P_k
    */
        P_result = (SymmetricMatrix)(((SymmetricMatrix)(Identity - (K_k * H_k))) * P_current);

        x_k = X_k;
    }

}