#!/usr/bin/env python3

"""
kalman_mousetracker.py - OpenCV mouse-tracking demo using TinyEKF

Adapted from

   http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/

Copyright (C) 2016 Simon D. Levy

MIT License
"""
import cv2
import numpy as np
from sys import exit

from tinyekf import EKF

# This delay will affect the Kalman update rate
DELAY_MSEC = 10

# Arbitrary display params
WINDOW_NAME = 'Kalman Mousetracker [ESC to quit]'
WINDOW_SIZE = 800

LINE_AA = cv2.LINE_AA  # if cv2.__version__[0] == '3' else cv2.CV_AA


class TrackerEKF(EKF):
    """
    An EKF for mouse tracking
    """

    def __init__(self):
        # Two state values (mouse coordinates), two measurement values (mouse coordinates)
        EKF.__init__(self, 2, 2)

    # 返回:  X_k, F_k
    def stateTransitionFunction(self, x):
        # State-transition function is identity
        # 这里直接返回了当前状态值得相同值
        return np.copy(x), np.eye(2)  # 返回的是一个二维的数组(N,N)，对角线的地方为1，其余的地方为0.

    # 返回: 预估测量值向量ZZ_K, H_k
    def stateToMeasurementTransitionFunction(self, x):
        # Observation function is identity
        H_k = np.eye(2)  # 状态值转换为测量值的函数为: y= f(x) = x,基本是恒等关系,故返回一个单位矩阵
        return H_k.dot(x), H_k   # 同时返回经状态转换函数变换后的测量值^: H_k(mxn) * X(nx1) = ZZ_k(mx1)


class MouseInfo(object):
    """
    A class to store X,Y points
    """

    def __init__(self):
        self.x, self.y = -1, -1

    def __str__(self):
        return '%4d %4d' % (self.x, self.y)


def mouseCallback(event, x, y, flags, mouse_info):
    """
    Callback to update a MouseInfo object with new X,Y coordinates
    """

    mouse_info.x = x
    mouse_info.y = y


def drawCross(img, center, r, g, b):
    """
    Draws a cross at the specified X,Y coordinates with color RGB
    """

    d = 5
    t = 2

    color = (r, g, b)

    ctrx = center[0]
    ctry = center[1]

    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def drawLines(img, points, r, g, b):
    '''
    Draws lines 
    '''

    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def newImage():
    '''
    Returns a new image
    '''

    return np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)


if __name__ == '__main__':

    # Create a new image in a named window
    img = newImage()
    cv2.namedWindow(WINDOW_NAME)

    # Create an X,Y mouse info object and set the window's mouse callback to modify it
    mouse_info = MouseInfo()
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback, mouse_info)

    # Loop until mouse inside window
    while True:
        if mouse_info.x > 0 and mouse_info.y > 0:
            break
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) == 27:
            exit(0)

    # These will get the trajectories(轨迹) for mouse location and Kalman estimate(估计)
    measured_points = []  # 测量值: 鼠标位置
    kalman_points = []  # 卡尔曼估计(预测)值:  估计鼠标位置

    # Create a new Kalman filter for mouse tracking
    kalfilt = TrackerEKF()

    # Loop till user hits escape
    while True:
        # Serve up a fresh image
        img = newImage()

        # Grab(抓取) current mouse position and add it to the trajectory
        measured = (mouse_info.x, mouse_info.y)  # 当前鼠标位置测量值
        measured_points.append(measured)  # 加入鼠标位置测量值到轨迹列表

        # Update the Kalman filter with the mouse point, getting the estimate.
        estimate = kalfilt.step((mouse_info.x, mouse_info.y))  # 输入当前鼠标位置测量值,  返回新的鼠标位置最优评估值

        # Add the estimate to the trajectory
        estimated = [int(c) for c in estimate]   # 最新最优估计值
        kalman_points.append(estimated)  # 加入鼠标位置估计(预测)值到估计轨迹列表

        # Display the trajectories and current points
        drawLines(img, kalman_points, 0, 255, 0)  # 绘制鼠标估计值轨迹
        drawCross(img, estimated, 255, 255, 255)  # 在最新估计位置绘制X图像
        drawLines(img, measured_points, 255, 255, 0)  # 绘制鼠标测量值轨迹
        drawCross(img, measured, 0, 0, 255)  # 在最新鼠标位置绘制鼠标图像

        # Delay for specified interval, quitting on ESC
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
            break
