#include <bfl/wrappers/matrix/matrix_wrapper.h>
#include "AltitudeDataFusion4Test.h"

using namespace std;
using namespace ekf;

/*================================================================================================================
Note:  使用此test_AltitudeDataFusion请在安装好以来后，在linux下单独编译或放在Ros某软件包下编译。
并在CMakeList.txt文件中add_executable（）添加：
src/AltitudeDataFusion4Test.cpp
src/test_AltitudeDataFusion.cpp
//=================================================================================================================
*/

const int DATA_COUNT = 14;
float measurement_gps[DATA_COUNT] = {1, 3.5, 5.6, 7.9, 10.87, 12.33, 15.33, 20.45, 30.22, 40.56, 55.67, 69.12, 82.83, 99.01};
float measurement_barometers_H[DATA_COUNT] = {1.5, 2, 7, 7, 9.9, 13.33, 14, 19, 32.22, 38, 59.69, 67, 85, 105.01};
float measurement_barometers[DATA_COUNT] = {101.30696865142431, 101.30095877932848, 101.24087593495592, 101.24087593495592,
                                            101.20604110913699, 101.16485244748047, 101.15680843373597, 101.09679483337116,
                                            100.93825774830097, 100.86900602460803, 100.60947532385319, 100.52212978044717,
                                            100.30731356084152, 100.06894598949584};
float measurement_imu[DATA_COUNT] = {1.1, 3.7, 5.0, 8.9, 11.87, 12.00, 15.99, 20.05, 31.22, 38.99, 56.01, 68.30, 84.03, 103.01};
float stateXArray[DATA_COUNT];

/*
Note:  此案例仅仅依赖输入的模拟数据,产生模拟结果,无图形化界面显示.
需要查看图形曲线,请到目录下ekf_AltitudeDataFusion.xlsx查看

# 下面是某次运行本程序的输出结果
R = [[1.e-06 0.e+00 0.e+00]
    [0.e+00 1.e-02 0.e+00]
    [0.e+00 0.e+00 1.e-03]]
    
#基于 measurement_barometers_H:
stateXArray = [1.0000509290345925, 3.475566530730563, 5.578737668262674, 7.878164958440969, 10.84159024461897, 
               12.315194571920056, 15.300993357528029, 20.39903001736479, 30.124997419203158, 40.45599032992332, 
               55.5217233570598, 68.98579363559189, 82.69580727540483, 98.85476385062523] 

#基于 measurement_barometers:
stateXArray =[1.01002,3.48549,5.58816,7.88758,10.8507,12.324,15.3097,20.4072,30.1319,40.4623,55.5258,68.9892,82.6974,98.8543,]
*/

int main(int argc, char **argv)
{
  /*
    # 测试数据中 大气压代表高度转为大气压值,  只是为了模拟测试,  in python
    values_barometers = []
    for index in range(len(measurement_barometers_H)):
        valueBarometers = 101.325 * math.pow((1 - measurement_barometers_H[index] / 44300), 5.256)
        values_barometers.append(valueBarometers)
    print(f"values_barometers={values_barometers}")
    */

  /*
    # 某次测试的一组值
    values_barometers = [101.30696865142431, 101.30095877932848, 101.24087593495592, 101.24087593495592, 
                                                  101.20604110913699, 101.16485244748047, 101.15680843373597, 101.09679483337116, 
                                                  100.93825774830097, 100.86900602460803, 100.60947532385319, 100.52212978044717, 
                                                  100.30731356084152, 100.06894598949584]
    */

  int state_count = 1;
  int measurement_count = 3;

  cout << "my_robot_pose_ekf              start\n";
  //Create a new Kalman filter for mouse tracking
  AltitudeDataFusion kalfilt(state_count, measurement_count, 0.01, 1e-4, 0.0001, 30);

  //更新传感器噪声矩阵,三个传感器具有不同的噪声(测量误差)
  SymmetricMatrix r_k(measurement_count); //矩阵3x3
  r_k = 0;
  r_k(1, 1) = 0.000001;  // GPS
  r_k(2, 2) = 0.01;      // 大气压测高
  r_k(3, 3) = 0.001;     // IMU 惯性传感器
  kalfilt.updateRk(r_k); // 更新传感器噪声矩阵,三个传感器具有不同的噪声(测量误差)
  cout << "my_robot_pose_ekf             R_k： " << r_k << "\n";

  ColumnVector estimateX;
  for (int index = 0; index < DATA_COUNT; index++)
  {
    ColumnVector measurementZ(measurement_count);
    measurementZ(1) = measurement_gps[index];
    measurementZ(2) = measurement_barometers[index];
    measurementZ(3) = measurement_imu[index];

    //输入当前鼠标位置测量值,  返回新的鼠标位置最优评估值
    kalfilt.doStep(measurementZ, estimateX);

    stateXArray[index] = estimateX(1);
  }
  cout << "my_robot_pose_ekf              end\n";

  cout << "Result: \nstateXArray=[";
  for (int index = 0; index < DATA_COUNT; index++)
    cout << stateXArray[index] << ",";
  cout << "]\n";

  return 0;
}
