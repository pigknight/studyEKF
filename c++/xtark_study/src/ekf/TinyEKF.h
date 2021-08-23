#ifndef __TINY_EKF_INCLUDE__
#define __TINY_EKF_INCLUDE__

#include <bfl/wrappers/matrix/matrix_wrapper.h>

using namespace MatrixWrapper;

namespace ekf
{

    class TinyEKF
    {
    public:
        // constructor
        TinyEKF(int numOfState, int numOfMeasurement, float pVal, float qVal, float rVal);

        // destructor
        ~TinyEKF();

        void updateRk(SymmetricMatrix &newRk);

        void setAdditiveMeasureNoiseRk(SymmetricMatrix &noiseMatrix);
        void getAdditiveMeasureNoiseRk(SymmetricMatrix &noiseMatrix);

        void setInitState(ColumnVector& _x_k, SymmetricMatrix& p_current);

        void getPostCovariance(SymmetricMatrix &matrix);

        //输入当前测量值,  返回新的最优评估值
        void doStep(ColumnVector &z, ColumnVector &x_k);

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
        virtual bool stateTransitionFunction(ColumnVector &x, ColumnVector &x_new, SymmetricMatrix &f_k)
        {
            return false;
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
        virtual bool stateToMeasurementTransitionFunction(ColumnVector &x, ColumnVector &zz_k, Matrix &h_k)
        {
            return false;
        }

    private:
        int numOfState;       //状态向量个数
        int numOfMeasurement; //测量值向量个数
        /*
    # 预测状态方程
        # (1). X_k = F_k * X_k-1 + B_k * u_k
        # F_k: 根据系统模型设计的预测矩阵或状态变量转移矩阵nxn
        # B_k: 系统外部控制矩阵 维数确保可以和前面F_k相乘
        # u_k: 系统外部控制向量,例如加速度,转向等.  维数确保可以和前面F_k相乘
    # 预测协方差矩阵
        # (2). P_k = F_k * P_k-1 * F_k^T + Q_k
        # Q_k: 各状态量对应的噪音协方差矩阵nxn
    # 卡尔曼增益: K_k
        # (3). K_k = P_k * H_k^T * (H_k * P_k * H_k^T + R_k)^-1
        # R_k: 传感器噪声(不确定性)协方差矩阵mxm
    # (4). X^_k = X_k + K_k * (z_k - H_k * X_k)
        #z_k: 传感器读数或均值
    # 最后,最优预测协方差矩阵
        # (5). P^_k = P_k - K_k * H_k * P_k = (I - K_k * H_k) * P_k
    */
        ColumnVector X_k;          //上一时刻(k-1)或当前时刻k的状态向量:  n个元素向量,或nx1矩阵
        SymmetricMatrix Q_k;       //各状态变量的预测噪声协方差矩阵nxn
        SymmetricMatrix P_current; //前一时刻(k-1)预测协方差矩阵: 预测P_k
        SymmetricMatrix P_result;  //最优P_k: 当前时刻最优估计协方差矩阵

        /*
        For update
    
        # 传感器测量值向量与预测值向量之间的非线性转换矩阵
        # mxn矩阵的元素为: 状态值和观测值之间非线性函数的一阶导数,或有限差分, 或连续差分的比值
        # m为测量值个数, n为状态量个数, 用处1: H_k(mxn) 乘 X(nx1) = ZZ_k(mx1)
        # H_k = np.eye(n)  # 返回的是一个二维的数组(m,n)，对角线的地方为1，其余的地方为0.

        # 传感器自身测量噪声带来的不确定性协方差矩阵
        # Set up covariance matrices for measurement noise
    */
        SymmetricMatrix R_k;

        //单位矩阵I, 这里当数字1使用.  P_k = P_k - K_k*H_k*P_k = (I - G_k*H_k)*P_k
        //Identity matrix(单位矩阵) will be useful later
        SymmetricMatrix Identity;
    };

}
#endif