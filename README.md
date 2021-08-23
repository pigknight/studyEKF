# studyEKF

#### 介绍
学习和实践扩展卡尔曼滤波器

#### 软件架构
项目分两个目录C++和python,分别对应于两个开发语言版本;

 **一. python目录:**  

1.tinyekf

是一个简单的ekf算法基类,其源码来自于上面参考文献4.
本人对其代码做了必要的中文注释和简单的结构调整;

其测试程序为: kalman_mousetracker.py

2.RobotEKF

这是一个从tinyekf继承来的子类.模拟一个小车,小车运动的加速度和方向由鼠标提供,
根据小车的运动学方程分析,对应于卡尔曼滤波方程扩展了F_k,增加了B_k(控制矩阵)和u_k(控制向量).
运行效果和tinyekf差不多,但鼠标移动过快时,加速度a过大,模拟小车就会飞出屏幕外.
总的来说模拟的效果并不好.但主要目的是扩展tinyekf,学习引入运动学方程;

其测试程序为: ekf_mouseRobot.py

3.AltitudeDataFusion
这是一个模拟飞机飞行时,基于气压计,陀螺仪(或IMU),GPS确定飞机海拔高度的例子;
也就是卡尔曼滤波器对三个传感器的数据融合;

主要目的有两个:

1>.学习多传感器数据融合;

2>.学习传感器非线性数据线性化处理了过程,即构建雅各比矩阵过程;

其测试程序为: ekf_AltitudeDataFusion.py

其测试数据效果图为: ekf_AltitudeDataFusion.xlsx

 **二. C++目录:** 

此目录是一个ROS功能包:xtark_study(此包名无意义,自己学习用的).

背景介绍:

先介绍ROS系统下的robot_pose_ekf 扩展卡尔曼滤波算法包这个包用于评估机器人的 3D 位姿，使用了来自不同源的位姿测量信息，它使用带有 6D（3D position and 3D orientation）模型信息的扩展卡尔曼滤波器来整合来自轮子里程计， IMU 传感器和视觉里程计的数据信息。 基本思路就是用松耦合方式融合不同传感器信息实现位姿估计。
该功能包依赖于一个BFL(贝叶斯老滤波库),robot_pose_ekf的主要滤波算法也是由BFL完成.

出于学习的目的,自己想写一个自己的滤波器,用于完成BFL一样的滤波功能;

于是工作开始了:

第一步:  ekf/TinyEKF.cpp  把上面python版本tinyekf用C++语言重新以便,作为EKF核心基类;

第二步: 为了先测试,编译了一个和上面python版本类似的多传感器数据融合计算海拔高度的例子: AltitudeDataFusion4Test.
GPS海拔高度,大气压值,IMU惯性传感器海拔高度,三者数据融合,得到最优海拔高度

扩展子类: test/AltitudeDataFusion4Test.cpp

可执行main测试: test/test_AltitudeDataFusion.cpp

融合数据图形化效果: test/ekf_AltitudeDataFusion.xlsx

注:  如不在ros环境, 可以运行此测试样例. 但要注意对orocos-bfl库的依赖,因为用到了其中的SymmetricMatrix, Matrix, ColumnVector等

第三步:  也是重点

1. 开发针对位姿估计的ekf子类: OdomEstimationEKF.cpp   我的滤波器的主要代码这在此类及其基类中.

设计思路上和robot_pose_ekf中的BLF有所不同: 

* BLF采用松耦合的方式为每种传感器(odom和imu)设计了独立的PDF(概率密度函数),并独立
设计了sysModel,及各传感器的测量模型mesurementModel. 获取到传感器数据后,也是各部分独立计算后,汇总到总的estimate状态中;

* 我的EKF思路则相对简单,设计了一个整体的状态方程模型和测量值模型.

2. 开发ROS系统节点(node)入口程序(main),包括ros初始化,odom,imu等消息的订阅,tf发布等.
需要说明的事,此入口程序,直接复制了ROS系统自带的robot_pose_ekf的入口程序源码,所以代码是一样的.
odom_estimation_node.cpp odom_estimation.cpp  nonlinearanalyticconditionalgaussianodo.cpp
会订阅和发布同样的tipoc,发布同样的transform.

3. 在节点入口程序中,删除对BFL的引用及相关滤波器代码; 替换为自己的滤波器代码:OdomEstimationEKF.

4. 上机测试,根据自己的机器环境不同,执行不同命令,我的小车运行命令如下:

   1>. shell 窗口1: ssh连接小车, 执行:  roscore
   
   2>. shell 窗口2: ssh连接小车, 执行:  roslaunch xtark_driver xtark_driver.launch

   3>. shell 窗口3: ssh连接小车, 执行:  roslaunch xtark_study my_robot_pose_ekf.launch

5. 上面第三步我是修改了我的系统中SLAM(动态定位和地图构建)中对robot_pose_ekf.launch的调用,改为
我的ekf: my_robot_pose_ekf.launch,直接启动小车去SLAM.

结果,使用自己的ekf构建出来的地图效果也非常不错,效果如图: [机器人小车上实际测试效果.png](https://gitee.com/okgwf/studyEKF/blob/master/c++/xtark_study/%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B0%8F%E8%BD%A6%E4%B8%8A%E5%AE%9E%E9%99%85%E6%B5%8B%E8%AF%95%E6%95%88%E6%9E%9C.png)

最后. 如何使用xtark_study包,  直接复制xtark_study目录到你的ROS系统/src目录下, 并在需要调用的的地方引用my_robot_pose_ekf.launch,
然后catkin_make一下即可.


参考文献:

1.卡尔曼滤波器工作原理

https://blog.csdn.net/michaelhan3/article/details/85458054

2.扩展卡尔曼滤波新手教程（一）----中文版

https://blog.csdn.net/xiaolong361/article/details/82912256

3.扩展卡尔曼滤波新手教程----英文版

The Extended Kalman Filter: An Interactive Tutorial for Non-Experts

http://home.wlu.edu/~levys/kalman_tutorial/

4. 基础源码来自: 
https://github.com/simondlevy/TinyEKF


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
