"""
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
"""

import numpy as np
import math
from tinyekf import EKF


class AltitudeDataFusion(EKF):
    """
    An EKF for mouse tracking
    """

    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1, interval=10):
        self.stateCount = n
        self.measurementCount = m
        self.interval = interval  # 预测更新时间间隔,单位 ms

        self.last_x_altitude = 0.  # 海拔高度
        self.last_z_barometers = 0.
        EKF.__init__(self, n, m, pval, qval, rval)

    # 返回:  X_k, F_k
    def stateTransitionFunction(self, x):
        # State-transition function is identity
        # F_k = np.array([self.stateCount])  # 状态转换矩阵设为I 因为k-1状态到K状态之间没有明确的转换关系, 我们假设为恒等关系
        # F_k.shape = (self.stateCount, self.stateCount)  # 1行,1列
        F_k = np.eye(self.stateCount)  # 状态转换矩阵设为I 因为k-1状态到K状态之间没有明确的转换关系, 我们假设为恒等关系

        # 这里直接返回了当前状态值得相同值
        return F_k.dot(x), F_k  # 返回的是一个二维的数组(N,N)，对角线的地方为1，其余的地方为0.

    # 返回: 预估测量值向量ZZ_K, H_k
    def stateToMeasurementTransitionFunction(self, x):
        """
        大气压同海拔高度的关系：
        　　P=P0×（1-H/44300)^5.256
        计算高度公式为:
           H=44300*(1- (P/P0)^(1/5.256) )
        式中：H——海拔高度，P0=大气压（0℃，101.325kPa）
        """

        # 基于大气压值: 把海拔高度估计值x,转换为大气压值,以便跟大气压的测量值做公式(4)中差运算
        altitude = x[0]  # 状态向量(1) only one
        z_barometers = 101.325 * math.pow((1-altitude/44300), 5.256)
        finite_difference = 1  # 默认变化率为1,  类似于 y = f(x) = x
        if self.last_x_altitude != 0 and self.last_z_barometers != 0 and (self.last_x_altitude-altitude) != 0:
            finite_difference = (z_barometers - self.last_z_barometers) / (altitude - self.last_x_altitude)
        H_k = np.array([1, finite_difference, 1])   # 测量值个数为3
        H_k.shape = (self.measurementCount, 1)  # 3行,1列

        new_x = H_k.dot(x)

        self.last_x_altitude = altitude
        self.last_z_barometers = z_barometers

        """
        # 基于大气压转换前的 海拔高度
        H_k = np.array([1, 1, 1])  # 测量值个数为3
        H_k.shape = (self.measurementCount, 1)  # 3行,1列
        new_x = H_k.dot(x)
        """

        return new_x, H_k   # 同时返回经状态转换函数变换后的测量值zz_k: H_k(mxn) * X(nx1) = ZZ_k(mx1)

