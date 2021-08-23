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

from RobotEKF import RobotEKF

# This delay will affect the Kalman update rate
DELAY_MSEC = 50

# Arbitrary display params
WINDOW_NAME = 'Kalman Mousetracker [ESC to quit]'
WINDOW_SIZE = 800

LINE_AA = cv2.LINE_AA  # if cv2.__version__[0] == '3' else cv2.CV_AA

class MouseInfo(object):
    """
    A class to store X,Y points
    """

    def __init__(self):
        self.x, self.y = -1, -1
        self.last_x, self.last_y = -1, -1
        self.last_time = time.time()
        self.v_x = 0
        self.v_y = 0
        self.last_v_x = 0
        self.last_v_y = 0
        self.a_x = 0   # V1 = V0 + at   -> a = (v1-v0)/t
        self.a_y = 0

    def __str__(self):
        return '%4d %4d' % (self.x, self.y)


def mouseCallback(event, x, y, flags, mouse_info):
    """
    Callback to update a MouseInfo object with new X,Y coordinates
    """

    mouse_info.x = x
    mouse_info.y = y

    now = time.time()
    if mouse_info.last_x > 0 and mouse_info.last_y > 0:
        t = (now*1000 - mouse_info.last_time)  # 转ms
        # print(f"mouseCallback: t={t}")
        # print(f"mouseCallback: now={now}")
        # print(f"mouseCallback: mouse_info.last_time={mouse_info.last_time}")

        if t > DELAY_MSEC:
            # s = math.sqrt(math.pow((x-mouse_info.last_x), 2) + math.pow((y-mouse_info.last_y), 2))
            # t = t/1000  # 不做ms 转 s,  单位按 pixels/ms算
            mouse_info.v_x = (x-mouse_info.last_x) / t
            mouse_info.v_y = (y - mouse_info.last_y) / t

            # print(f"mouseCallback:  mouse_info.v_x={mouse_info.v_x}")
            # print(f"mouseCallback:  mouse_info.v_y={mouse_info.v_y}")

            mouse_info.a_x = (mouse_info.v_x - mouse_info.last_v_x) / t  # a = (v1-v0)/t
            mouse_info.a_y = (mouse_info.v_y - mouse_info.last_v_y) / t  # a = (v1-v0)/t

            # print(f"mouseCallback:  mouse_info.a_x={mouse_info.a_x}")
            # print(f"mouseCallback:  mouse_info.a_y={mouse_info.a_y}")

            mouse_info.last_v_x = mouse_info.v_x
            mouse_info.last_v_y = mouse_info.v_y
            mouse_info.last_time = now * 1000  # 转ms

    mouse_info.last_x = x
    mouse_info.last_y = y


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
    kalfilt = RobotEKF(4, 4, pval=0.1, qval=1e-4, rval=0.1, interval=DELAY_MSEC)

    # Loop till user hits escape
    while True:
        # Serve up a fresh image
        img = newImage()

        # Grab(抓取) current mouse position and add it to the trajectory
        measured = (mouse_info.x, mouse_info.y)  # 当前鼠标位置测量值
        measured_points.append(measured)  # 加入鼠标位置测量值到轨迹列表

        # calculate and update acceleration
        kalfilt.update_acceleration(mouse_info.a_x, mouse_info.a_y)

        # Update the Kalman filter with the mouse point, getting the estimate.
        estimate = kalfilt.step((mouse_info.x, mouse_info.v_x, mouse_info.y, mouse_info.v_y))  # 输入当前鼠标位置测量值,  返回新的鼠标位置最优评估值

        print(f"estimate={estimate}")
        # Add the estimate to the trajectory
        estimated = [int(c) for c in estimate]  # 最新最优估计值
        kalman_points.append([estimated[0], estimated[2]])  # 加入鼠标位置估计(预测)值到估计轨迹列表

        # Display the trajectories and current points
        drawLines(img, kalman_points, 0, 255, 0)  # 绘制鼠标估计值轨迹
        drawCross(img, estimated, 255, 255, 255)  # 在最新估计位置绘制X图像
        drawLines(img, measured_points, 255, 255, 0)  # 绘制鼠标测量值轨迹
        drawCross(img, measured, 0, 0, 255)  # 在最新鼠标位置绘制鼠标图像

        # Delay for specified interval, quitting on ESC
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
            break
