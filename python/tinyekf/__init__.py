"""
Extended Kalman Filter in Python

Copyright (C) 2016 Simon D. Levy

MIT License
"""

import numpy as np
from abc import ABCMeta, abstractmethod


class EKF(object):
    """
    A abstrat class for the Extended Kalman Filter, based on the tutorial in
    http://home.wlu.edu/~levys/kalman_tutorial.
    """
    __metaclass__ = ABCMeta

    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1):
        """
        Creates a KF object with n states, m observables, and specified values for 
        prediction noise covariance pval, process noise covariance qval, and 
        measurement noise covariance rval.
        """
        # 预测状态变换矩阵,比如依据系统运动学方程或几何学方程得出的变换矩阵,维数n,取自状态向量个数n
        # F_k = np.eye(n)  # 返回的是一个二维的数组(N,N)，对角线的地方为1，其余的地方为0.

        # Set up covariance matrices for process noise
        self.Q_k = np.eye(n) * qval  # 各状态变量的预测噪声协方差矩阵

        # Current state is zero,
        # numpy.zeros(shape, dtype = float, order = 'C')
        # np.zeros(5) --> [0. 0. 0. 0. 0.]
        # F_k(n,n) * X^_k-1(n,1) --> x^_k(n,1) 新时刻状态向量
        self.x = np.zeros(n)  # 上一时刻(k-1)或当前时刻k的状态向量:  n个元素向量,或nx1矩阵

        # 前一时刻(k-1)预测协方差矩阵: 预测P_k
        # No current prediction noise covariance
        self.P_current = None
        # 最优P_k: 当前时刻最优估计协方差矩阵,is diagonal(对角线) noise covariance matrix
        self.P_result = np.eye(n) * pval

        """
        For update
        """
        # 传感器测量值向量与预测值向量之间的非线性转换矩阵
        # mxn矩阵的元素为: 状态值和观测值之间非线性函数的一阶导数,或有限差分, 或连续差分的比值
        # m为测量值个数, n为状态量个数, 用处1: H_k(mxn) 乘 X(nx1) = ZZ_k(mx1)
        # H_k = np.eye(n)  # 返回的是一个二维的数组(m,n)，对角线的地方为1，其余的地方为0.

        # 传感器自身测量噪声带来的不确定性协方差矩阵
        # Set up covariance matrices for measurement noise
        self.R_k = np.eye(m) * rval

        # 单位矩阵I, 这里当数字1使用.  P_k = P_k - K_k*H_k*P_k = (I - G_k*H_k)*P_k
        # Identity matrix(单位矩阵) will be useful later
        self.I = np.eye(n)

    def updateR(self, R):
        self.R_k = R

    # 输入当前鼠标位置测量值,  返回新的鼠标位置最优评估值
    def step(self, z):  # z:  (mouse_info.x, mouse_info.y)
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """

        # Predict ----------------------------------------------------

        """  根据系统运动学方程或几何学方程,更新预测状态
        参数: 
            x 前一时刻(k-1)状态向量
        返回:
            x: 新时刻(k)状态向量
            F_K:一个针对新状态的雅各比行列式(n*n矩阵),矩阵的元素为: 状态值和观测值之间非线性函数的一阶导数,或有限差分, 或连续差分的比值
        """
        # 预测状态方程
        # (1). X_k = F_k * X_k-1 + B_k * u_k
        # F_k: 根据系统模型设计的预测矩阵或状态变量转移矩阵nxn
        # B_k: 系统外部控制矩阵 维数确保可以和前面F_k相乘
        # u_k: 系统外部控制向量,例如加速度,转向等.  维数确保可以和前面F_k相乘
        self.x, F_k = self.stateTransitionFunction(self.x)

        # 预测协方差矩阵
        # (2). P_k = F_k * P_k-1 * F_k^T + Q_k
        # Q_k: 各状态量对应的噪音协方差矩阵nxn
        self.P_current = F_k * self.P_result * F_k.T + self.Q_k  # P_k

        # Update -----------------------------------------------------
        # 预测值状态转测量值变换函数(即测量值转状态值得反函数)矩阵,把预测值状态向量转换为与测量值向量同样的单位或度量,
        # 必要时需要非线性函数线性化(返回雅各比矩阵)
        # 1. 一个m个元素的预估测量值向量: 经经雅各比矩阵 * 当前状态值向量得到的 "预估测量值向量"
        # H_k: 预测值状态转测量值变换函数的雅各比矩阵
        zz_k, H_k = self.stateToMeasurementTransitionFunction(self.x)

        # 卡尔曼增益: K_k
        # (3). K_k = P_k * H_k^T * (H_k * P_k * H_k^T + R_k)^-1
        # R_k: 传感器噪声(不确定性)协方差矩阵mxm
        K_k = np.dot(self.P_current.dot(H_k.T), np.linalg.inv(H_k.dot(self.P_current).dot(H_k.T) + self.R_k))

        # 最终,最优预测状态向量值
        # z_k: 传感器读数或均值
        # (4). X^_k = X_k + K_k * (z_k - H_k * X_k)
        # self.x += np.dot(K_k, (np.array(z) - H_k.dot(self.x)))

        # zz_k: 由状态转测量值函数stateToMeasurementTransitionFunction返回的值
        # (4). X^_k = X_k + K_k * (z_k - zz_k)    # Note:  此处的zz_k = H_k * X_k)
        self.x += np.dot(K_k, (np.array(z) - zz_k))

        # 最后,最优预测协方差矩阵
        # (5). P^_k = P_k - K_k * H_k * P_k = (I - K_k * H_k) * P_k
        self.P_result = np.dot(self.I - np.dot(K_k, H_k), self.P_current)

        # return self.x.asarray()
        return self.x

    """
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
    """

    @abstractmethod
    def stateTransitionFunction(self, x):
        """
        Your implementing class should define this method for the state-transition function f(x).
        Your state-transition fucntion should return a NumPy array of n elements representing the
        new state, and a NumPy array of n x n elements representing the predict model
        """
        raise NotImplementedError()

    """
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
    """

    @abstractmethod
    def stateToMeasurementTransitionFunction(self, x):
        """
        Your implementing class should define this method for the observation function h(x), returning
        a NumPy array of m elements, and a NumPy array of m x n elements representing the Jacobian matrix H of the observation function
        with respect to the observation. For example, your function might include a component that
        turns barometric pressure into altitude in meters.
        """
        raise NotImplementedError()
