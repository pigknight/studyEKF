#!/usr/bin/env python3

"""
ekf_mouseRobot.py - OpenCV mouse robot demo using TinyEKF

Copyright (C) 2016 Simon D. Levy

MIT License
"""
import cv2
import math
import time  # 引入time模块
import numpy as np
from sys import exit

from AltitudeDataFusion import AltitudeDataFusion

LINE_AA = cv2.LINE_AA  # if cv2.__version__[0] == '3' else cv2.CV_AA

measurement_gps = [1, 3.5, 5.6, 7.9, 10.87, 12.33, 15.33, 20.45, 30.22, 40.56, 55.67, 69.12, 82.83, 99.01]
measurement_barometers_H = [1.5, 2, 7, 7, 9.9, 13.33, 14, 19, 32.22, 38, 59.69, 67, 85, 105.01]
measurement_barometers = [101.30696865142431, 101.30095877932848, 101.24087593495592, 101.24087593495592,
                          101.20604110913699, 101.16485244748047, 101.15680843373597, 101.09679483337116,
                          100.93825774830097, 100.86900602460803, 100.60947532385319, 100.52212978044717,
                          100.30731356084152, 100.06894598949584]
measurement_imu = [1.1, 3.7, 5.0, 8.9, 11.87, 12.00, 15.99, 20.05, 31.22, 38.99, 56.01, 68.30, 84.03, 103.01]
stateXArray = []

"""
Note:  此案例仅仅依赖输入的模拟数据,产生模拟结果,无图形化界面显示.
需要查看图形曲线,请到目录下ekf_AltitudeDataFusion.xlsx查看

# 下面是某次运行本程序的输出结果
R = [[1.e-06 0.e+00 0.e+00]
    [0.e+00 1.e-02 0.e+00]
    [0.e+00 0.e+00 1.e-03]]
    
#基于 measurement_barometers_H:
stateXArray = [1.0000509290345925, 3.475566530730563, 5.578737668262674, 7.878164958440969, 10.84159024461897, 
               12.315194571920056, 15.300993357528029, 20.39903001736479, 30.124997419203158, 40.45599032992332, 
               55.5217233570598, 68.98579363559189, 82.69580727540483, 98.85476385062523] """
"""
#基于 measurement_barometers:
stateXArray = [1.0100196732766782, 3.4854862008594245, 5.5881563560552046, 7.887578739462468, 10.850713687883315, 
               12.323971830124972, 15.309700158561197, 20.40723563343255, 30.131874831770933, 40.46227617384393, 
               55.52583234011125, 68.98914961329632, 82.69735421828966, 98.85429027346373] """

if __name__ == '__main__':
    """
    # 测试数据中 大气压代表高度转为大气压值,  只是为了模拟测试
    values_barometers = []
    for index in range(len(measurement_barometers_H)):
        valueBarometers = 101.325 * math.pow((1 - measurement_barometers_H[index] / 44300), 5.256)
        values_barometers.append(valueBarometers)
    print(f"values_barometers={values_barometers}")
    """

    """
    # 某次测试的一组值
    values_barometers = [101.30696865142431, 101.30095877932848, 101.24087593495592, 101.24087593495592, 
                         101.20604110913699, 101.16485244748047, 101.15680843373597, 101.09679483337116, 
                         100.93825774830097, 100.86900602460803, 100.60947532385319, 100.52212978044717, 
                         100.30731356084152, 100.06894598949584]"""


    # Create a new Kalman filter for mouse tracking
    kalfilt = AltitudeDataFusion(1, 3, pval=0.01, qval=1e-4, rval=0.0001)

    # 更新传感器噪声矩阵,三个传感器具有不同的噪声(测量误差)
    R = np.eye(3)
    R[0, 0] = 0.000001  # GPS
    R[1, 1] = 0.01       # 大气压测高
    R[2, 2] = 0.001   # IMU 惯性传感器
    kalfilt.updateR(R)  # 更新传感器噪声矩阵,三个传感器具有不同的噪声(测量误差)

    print(f"R={R}")

    for index in range(len(measurement_barometers)):
        # 输入当前鼠标位置测量值,  返回新的鼠标位置最优评估值
        estimate = kalfilt.step((measurement_gps[index], measurement_barometers[index], measurement_imu[index]))  # 基于大气压
        # estimate = kalfilt.step((measurement_gps[index], measurement_barometers_H[index], measurement_imu[index]))  # 基于大气压转换前海拔高度
        stateXArray.append(estimate[0])

    print(f"stateXArray={stateXArray}")

