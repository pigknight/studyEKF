// Copyright (C) 2008 Wim Meeussen <meeussen at willowgarage com>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//

#include <nonlinearanalyticconditionalgaussianodo.h>
#include <bfl/wrappers/rng/rng.h> // Wrapper around several rng libraries
#define NUMCONDARGUMENTS_MOBILE 2

namespace BFL
{
  using namespace MatrixWrapper;

  NonLinearAnalyticConditionalGaussianOdo::NonLinearAnalyticConditionalGaussianOdo(const Gaussian &additiveNoise)
      : AnalyticConditionalGaussianAdditiveNoise(additiveNoise, NUMCONDARGUMENTS_MOBILE),
        df(6, 6)
  {
    // initialize df matrix
    //密度函数矩阵,雅各比矩阵F_k.  对参数1即状态向量X_k(3)求雅各比矩阵
    for (unsigned int i = 1; i <= 6; i++)
    {
      for (unsigned int j = 1; j <= 6; j++)
      {
        if (i == j)
          df(i, j) = 1;
        else
          df(i, j) = 0;
      }
    }
  }

  NonLinearAnalyticConditionalGaussianOdo::~NonLinearAnalyticConditionalGaussianOdo() {}

  ColumnVector NonLinearAnalyticConditionalGaussianOdo::ExpectedValueGet() const
  {
    //条件概率密度的条件参数，状态与输入都是条件概率密度的私有变量。
    ColumnVector state = ConditionalArgumentGet(0); //获取状态向量X_k(3)
    ColumnVector vel = ConditionalArgumentGet(1);   //获取输入向量z_k(2)  ,这里是速度和方向

    //进行非线性计算,更新状态
    //近似执行        X_k = F_k * X_k-1 + B_k * u_k
    //查看系统预测模型更新(update)时,控制向量输入都为方向为vel(1,2)均为0
    state(1) += cos(state(6) /*theta*/) * vel(1) /*v*/; //根据速度v更新x轴位移
    state(2) += sin(state(6) /*theta*/) * vel(1) /*v*/; //根据速度v更新y轴位移
    state(6) += vel(2);                                 //更新theta,即方向.

    // Get the mean Value of the Additive Gaussian uncertainty,即:_additiveNoise_Mu:   X_k-1(6)
    return state + AdditiveNoiseMuGet();
  }

  //对第i个条件参数求偏导数. 相当于F_k和B_k的结合,第0个表状态向量X_k(6)
  /*
  卡尔曼滤波只能针对线性模型，包括线性运动模型、线性观测模型并且随机误差满足高斯分布。当处理的模型是非线性的，
  比如运动方程是非线性的，那么从上一次状态（用高斯分布描述）经过转换矩阵到当前状态就不再是高斯分布能描述的了。
  而到底用什么分布函数来描述都很难说的清。对于线性模型来说，上一次状态用高斯分布来描述，线性转换后的状态依然
  可以用高斯分布来描述。扩展卡尔曼滤波是实现了一种将非线性模型线性化的方法。除此之外，其它运算过程与卡尔曼滤波
  是一致的。扩展卡尔曼滤波的思路是: 1）在工作点附近，用泰勒展开式去线性近似。泰勒展开式的高阶项被忽略了所以得到下面的公式。
  原文链接：https://blog.csdn.net/shoufei403/article/details/102655696
  */
  /* 根据导数常识:
  y = sin(theta)   导函数: y` = cos(theta)
  y = cos(theta) 导函数: y` = -sin(theta)
  */
  Matrix NonLinearAnalyticConditionalGaussianOdo::dfGet(unsigned int i) const
  {
    if (i == 0) //derivative to the first conditional argument (x)  状态向量X_k(6)
    {
      double vel_trans = ConditionalArgumentGet(1)(1); //输入向量的速度值
      double yaw = ConditionalArgumentGet(0)(6);       //状态向量的偏航角theta

      //这里的第3列 是不是 应该为 第6列???
      //查看系统预测模型更新(update)时,控制向量输入都为0,所以这里3或6都没影响
      df(1, 3) = -vel_trans * sin(yaw); //求x关于yaw的偏导:             x =  vel(1)*cos(state(3))   x` = -vel(1)*sin(state(3))
      df(2, 3) = vel_trans * cos(yaw);  //求y关于yaw的偏导              y =  vel(1)*sin(state(3))   y` = vel(1)*cos(state(3))
      /*
		df = 
		[
		 1 0 0 0 0 x`
		 0 1 0 0 0 y`
		 0 0 1 0 0 0
		 0 0 0 1 0 0
		 0 0 0 0 1 0
		 0 0 0 0 0 1
		]
		执行 F_k(6x6) * X_k-1(   x
							   y
							   z
							   r_theta
							   p_theta
							   y_theta )  -->  (6x1)
		*/

      return df;
    }
    else
    {
      if (i >= NumConditionalArgumentsGet())
      {
        cerr << "This pdf Only has " << NumConditionalArgumentsGet() << " conditional arguments\n";
        exit(-BFL_ERRMISUSE);
      }
      else
      {
        cerr << "The df is not implemented for the" << i << "th conditional argument\n";
        exit(-BFL_ERRMISUSE);
      }
    }
  }

} //namespace BFL
