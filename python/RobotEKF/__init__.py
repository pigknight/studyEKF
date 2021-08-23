"""
MouseRobotEKF 自己实现的一个EKF算法,
此算法中把鼠标作为机器人控制信号,为机器人提供运动需要的加速度和方向向量: M_vector,(加速度就用鼠标的加速度a,方向也是鼠标运动的方向);
同时,鼠标的当前位置和当前速度,也作为机器人的测量传感器使用Z_k(P,V)
Note:  以下: 当前时刻: cur or k-1   下一时刻: next or k
1.状态向量定义:
机器人根据鼠标的运动产生自己的位置和速度两个状态向量state_x(Position(x,y),Velocity(x,y)),即: X(P,V);
估计值向量: X= [ Px
               Vx
               Py
               Vy]
估计值协方差矩阵: P =[ Cov(Px,Px) Cov(Px,Vx) Cov(Px,Py)  Cov(Px,Vy)
                    ....
                    ....
                    .... ]    # Note:  Cov(x,y)参见协方差公式
2.根据运动学分析出预测方程:
p_next_x = p_cur_x + delta_t * v_cur_x   #--->  [1, delta_t ]
v_next_x =  0      +  v_cur_x            #--->  [0,  1      ] 先假设匀速运动
p_next_y = p_cur_y + delta_t * v_cur_y   #--->  [1, delta_t ]
v_next_y =  0      +  v_cur_y            #--->  [0,  1      ] 先假设匀速运动
用矩阵表示估计向量
(1) X_next = [ 1  delta_t 0 0
               0  1       0 0
               0    0     1 delta_t
               0    0     0 1]  * X_cur  = F_k * X_cur
上面,预测矩阵 或 估计矩阵 或 状态转移矩阵:
F_k = [ 1    delta_t 0 0
        0    1       0 0
        0    0       1 delta_t
        0    0       0 1 ]
根据协方差推到公式: Cov(Ax) = A * Cov(x) * A_T   依据此,就可以更新估计值协方差矩阵P
(1) X_next = F_k * X_cur                  # 状态向量 方程
(2) P_next = F_k * P_cur * F_k_T          # 估计值协方差矩阵 方程
*************暂不添加外部控制向量,比如这里的加速度a(a_x,a_y),  通过EKF的变量ENABLE_ACCELERATION设置***************
进一步,设鼠标的加速度为a,分为x方向和y方向两个加速度分量, 根据前面控制要求: 实际控制加速度u是鼠标加速度a(a_x,a_y), 有:
p_next_x = p_cur_x + delta_t * v_cur_x + 1/2 * a_x * delta_t^2  #--->  [ 1/2 * delta_t^2 ]  运动学位移公式: S1 = s0 +V0 + 1/2at^2
v_next_x =  0    +  v_cur_x          +   a_x * delta_t        #--->    [    delta_t    ]  运动学速度公式: V1 = V0 + at
p_next_y = p_cur_y + delta_t * v_cur_y + 1/2 * a_y * delta_t^2  #--->  [ 1/2 * delta_t^2 ]  运动学位移公式: S1 = s0 +V0 + 1/2at^2
v_next_y =  0    +  v_cur_y          +   a_y * delta_t        #--->    [    delta_t      ]  运动学速度公式: V1 = V0 + at
得新的矩阵表示估计向量形式:
(1) X_next = F_k * X_cur + [delta_t^2/2
                            delta_t
                            delta_t^2/2
                            delta_t ] * a
           = F_k * X_cur + B_k * U_k
这里控制矩阵B_k: [ delta_t^2/2
                 delta_t
                 delta_t^2/2
                 delta_t    ]
   控制变量矩阵(n*1)U_k:  a
*******************************************************************************************
因为机器人行进过程中,会受到外部噪声影响,添加噪声协方差矩阵Q_k后:
#根据上一时刻系统状态和当前时刻系统控制量所得到的系统估计值,该估计值又叫做先验估计值,为各变量高斯分布的均值.
(1) X_next = F_k * X_cur + B_k * U_k               # 状态向量预测方程,先验估计值,为各变量高斯分布的均值
#协方差矩阵,代表了不确定性,它是由上一时刻的协方差矩阵和外部噪声的协方差一起计算得到的.
(2) P_next = F_k * P_cur * F_k_T  + Q_k        # 估计值协方差矩阵 方程
这两个公式代表了卡尔曼滤波器中的预测部分,是卡尔曼滤波器五个基础公式中的前两个.
(3),(4),(5)详见tinyekf包
"""
import math
import numpy as np
from tinyekf import EKF

# Note:  谨慎打开加速度选项,因为加速度并没有和速度方向相关联,所以加速度会导致朝某方向快速运动
ENABLE_ACCELERATION = False


class RobotEKF(EKF):
    """
    An EKF for mouse tracking
    """

    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1, interval=10):
        self.stateCount = n
        self.measurementCount = m
        self.interval = interval  # 预测更新时间间隔,单位 ms
        self.acceleration_x = 0  # pixels/ms^2
        self.acceleration_y = 0  # pixels/ms^2

        EKF.__init__(self, n, m, pval, qval, rval)

    # update 加速度
    def update_acceleration(self, a_x,a_y):
        self.acceleration_x = a_x  # pixels/ms^2
        self.acceleration_y = a_y  # pixels/ms^2

    # 返回:  X_k: 向量(nx1), F_k: (nxn)
    def stateTransitionFunction(self, x):
        # 执行预测方程 (1). X_k = F_k * X_k-1 + B_k * u_k
        F_k = np.array([(1, self.interval, 0, 0),
                        (0, 1,             0, 0),
                        (0, 0,             1, self.interval),
                        (0, 0,             0, 1)
                        ])
        F_k.shape = (4, 4)

        if ENABLE_ACCELERATION:
            # B_k * u_k
            # 这里注意: B_k * u_k的结果维数要保持和X_k同样的维数,以便做矩阵相加
            tt = math.pow(self.interval, 2)  # 不做 ms 转 s,单位按 pixels/ms算
            B_k_dot_u_k = np.array((int(tt/2 * self.acceleration_x),
                                   int(self.interval * self.acceleration_x),
                                   int(tt/2 * self.acceleration_y),
                                   int(self.interval * self.acceleration_y)
                                    ))
            print(f"stateTransitionFunction: B_k_dot_u_k={B_k_dot_u_k}")

            # X_k = F_k * X_k-1 + B_k * u_k
            new_x = F_k.dot(x) + B_k_dot_u_k
        else:
            new_x = F_k.dot(x)

        return new_x, F_k  # 返回的是一个二维的数组(N,N)，对角线的地方为1，其余的地方为0.

    # 返回: 预估测量值向量ZZ_K, H_k
    def stateToMeasurementTransitionFunction(self, x):
        # Observation function is identity
        H_k = np.eye(self.measurementCount)  # 状态值转换为测量值的函数为: y= f(x) = x,基本是恒等关系,故返回一个单位矩阵
        return H_k.dot(x), H_k   # 同时返回经状态转换函数变换后的测量值^: H_k(mxn) * X(nx1) = ZZ_k(mx1)

