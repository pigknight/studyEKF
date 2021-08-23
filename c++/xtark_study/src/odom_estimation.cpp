/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Wim Meeussen */

#include <odom_estimation.h>

using namespace MatrixWrapper;
//using namespace BFL;
using namespace ekf;
using namespace tf;
using namespace std;
using namespace ros;

namespace estimation
{
  // constructor
  OdomEstimation::OdomEstimation() : //prior_(NULL),
                                     filter_(NULL),
                                     estimateX(6),
                                     measurementZ(9),
                                     filter_initialized_(false),
                                     odom_initialized_(false),
                                     imu_initialized_(false),
                                     vo_initialized_(false),
                                     gps_initialized_(false),
                                     output_frame_(std::string("odom_combined")),
                                     base_footprint_frame_(std::string("base_footprint")){

                                     };

  // destructor
  OdomEstimation::~OdomEstimation()
  {
    if (filter_)
      delete filter_;
    /*if (prior_)
      delete prior_;
    delete odom_meas_model_;
    delete odom_meas_pdf_;
    delete imu_meas_model_;
    delete imu_meas_pdf_;
    delete vo_meas_model_;
    delete vo_meas_pdf_;
    delete gps_meas_model_;
    delete gps_meas_pdf_;
    delete sys_pdf_;
    delete sys_model_;*/
  };

  // initialize prior density of filter
  void OdomEstimation::initialize(const Transform &prior, const Time &time)
  {
    /****************************
    * Linear prior DENSITY     *
    <三>.初始估计与不确定度的先验分布，同样包含了均值以及协方差
    ***************************/
    // set prior of filter
    ColumnVector prior_Mu(6); prior_Mu = 0;//--->X_k-1 状态向量(6)先验(初始)估计值(x,y,z,roll,pitch,yaw)
    decomposeTransform(prior, prior_Mu(1), prior_Mu(2), prior_Mu(3), prior_Mu(4), prior_Mu(5), prior_Mu(6));
    SymmetricMatrix prior_Cov(6); //--->P_k-1   初始预测矩阵6x6,
    for (unsigned int i = 1; i <= 6; i++)
    {
      for (unsigned int j = 1; j <= 6; j++)
      {
        if (i == j)
          prior_Cov(i, j) = pow(0.001, 2);
        else
          prior_Cov(i, j) = 0;
      }
    }

    /*
    prior_ = new Gaussian(prior_Mu, prior_Cov); //关于先验状态值的二维高斯分布
    filter_ = new ExtendedKalmanFilter(prior_); //Construction of the Filter
    */
    float qVal = pow(1000.0,2);
    filter_ = new OdomEstimationEKF(6, 9, 0.1, qVal, 1);
    filter_->setInitState(prior_Mu, prior_Cov);

    SymmetricMatrix noiseMatrix(9); noiseMatrix = 0;//--->R_k    测量值个数为9*9
    for (unsigned int i=1; i<=9; i++) noiseMatrix(i,i) = 1;
    filter_->updateRk(noiseMatrix);

    // remember prior
    addMeasurement(StampedTransform(prior, time, output_frame_, base_footprint_frame_));
    filter_estimate_old_vec_ = prior_Mu;
    filter_estimate_old_ = prior;
    filter_time_old_ = time;

    // filter initialized
    filter_initialized_ = true;
  }

  // update filter
  bool OdomEstimation::update(bool odom_active, bool imu_active, bool gps_active, bool vo_active, const Time &filter_time, bool &diagnostics_res)
  {
    // only update filter when it is initialized
    if (!filter_initialized_)
    {
      ROS_INFO("Cannot update filter when filter was not initialized first.");
      return false;
    }

    // only update filter for time later than current filter time
    double dt = (filter_time - filter_time_old_).toSec();
    if (dt == 0)
      return false;
    if (dt < 0)
    {
      ROS_INFO("Will not update robot pose with time %f sec in the past.", dt);
      return false;
    }
    ROS_DEBUG("Update filter at time %f with dt %f", filter_time.toSec(), dt);

    // process odom measurement
    // ------------------------
    ROS_DEBUG("Process odom meas");
    if (odom_active)
    {
      if (!transformer_.canTransform(base_footprint_frame_, "wheelodom", filter_time))
      {
        ROS_ERROR("filter time older than odom message buffer");
        return false;
      }
      transformer_.lookupTransform("wheelodom", base_footprint_frame_, filter_time, odom_meas_);
      if (odom_initialized_)
      {
        // convert absolute odom measurements to relative odom measurements in horizontal plane
        Transform odom_rel_frame = Transform(tf::createQuaternionFromYaw(filter_estimate_old_vec_(6)),
                                             filter_estimate_old_.getOrigin()) *
                                   odom_meas_old_.inverse() * odom_meas_;
        ColumnVector odom_rel(6);
        decomposeTransform(odom_rel_frame, odom_rel(1), odom_rel(2), odom_rel(3), odom_rel(4), odom_rel(5), odom_rel(6));
        angleOverflowCorrect(odom_rel(6), filter_estimate_old_vec_(6));

        //ROS_INFO("odom data: (%f,%f,%f,%f,%f,%f)", odom_rel(1), odom_rel(2), odom_rel(3), odom_rel(4), odom_rel(5), odom_rel(6));
        // update filter
        //odom_meas_pdf_->AdditiveNoiseSigmaSet(odom_covariance_ * pow(dt, 2)); //更新odom协方差,即更新前面measNoiseOdom_Cov
        ROS_DEBUG("Update filter with odom measurement %f %f %f %f %f %f",
                  odom_rel(1), odom_rel(2), odom_rel(3), odom_rel(4), odom_rel(5), odom_rel(6));
        //filter_->Update(odom_meas_model_, odom_rel );//测量向量:z_k); //分步更新odom测量模型

        
        for (unsigned int i = 1; i <= 6; i++)
          measurementZ(i) = odom_rel(i);
        measurementZ(9) = 0;

        SymmetricMatrix noiseMatrix(9); //测量值个数为9*9
        filter_->getAdditiveMeasureNoiseRk(noiseMatrix);

        SymmetricMatrix noiseCovariance = odom_covariance_ * pow(dt, 2);
        for (unsigned int i = 1; i <= 6; i++)
          for (unsigned int j = 1; j <= 6; j++)
            noiseMatrix(i, j) = noiseCovariance(i, j);
        
        for (unsigned int i = 1; i <= 6; i++)
          for (unsigned int j = 1; j <= 3; j++)
            noiseMatrix(i, 6 + j) = 0;
        for (unsigned int i = 1; i <= 3; i++)
          for (unsigned int j = 1; j <= 6; j++)
            noiseMatrix(6+i, j) = 0;

        filter_->setAdditiveMeasureNoiseRk(noiseMatrix /*measurementCount * stateCount*/);

        diagnostics_odom_rot_rel_ = odom_rel(6);
      }
      else
      {
        odom_initialized_ = true;
        diagnostics_odom_rot_rel_ = 0;
      }
      odom_meas_old_ = odom_meas_;
    }
    // sensor not active
    else
      odom_initialized_ = false;

    // process imu measurement
    // -----------------------
    if (imu_active)
    {
      if (!transformer_.canTransform(base_footprint_frame_, "imu", filter_time))
      {
        ROS_ERROR("filter time older than imu message buffer");
        return false;
      }
      transformer_.lookupTransform("imu", base_footprint_frame_, filter_time, imu_meas_);
      if (imu_initialized_)
      {
        // convert absolute imu yaw measurement to relative imu yaw measurement
        Transform imu_rel_frame = filter_estimate_old_ * imu_meas_old_.inverse() * imu_meas_;
        ColumnVector imu_rel(3);
        double tmp;
        decomposeTransform(imu_rel_frame, tmp, tmp, tmp, tmp, tmp, imu_rel(3));
        decomposeTransform(imu_meas_, tmp, tmp, tmp, imu_rel(1), imu_rel(2), tmp);
        angleOverflowCorrect(imu_rel(3), filter_estimate_old_vec_(6));
        diagnostics_imu_rot_rel_ = imu_rel(3);

        //ROS_INFO("imu data: (%f,%f,%f)", imu_rel(1), imu_rel(2), imu_rel(3));
        // update filter
        //imu_meas_pdf_->AdditiveNoiseSigmaSet(imu_covariance_ * pow(dt, 2)); //更新imu协方差, 即更新前面measNoiseImu_Cov
        //filter_->Update(imu_meas_model_, imu_rel /*测量向量:z_k*/);         //分步更新imu测量模型

        for (unsigned int i = 1; i <= 3; i++)
          measurementZ(6 + i) = imu_rel(i);

        SymmetricMatrix noiseMatrix(9); //测量值个数为9*9
        filter_->getAdditiveMeasureNoiseRk(noiseMatrix);
        SymmetricMatrix noiseCovariance = imu_covariance_ * pow(dt, 2);
        for (unsigned int i = 1; i <= 3; i++)
          for (unsigned int j = 1; j <= 3; j++)
            noiseMatrix(6 + i, 6 + j) = noiseCovariance(i, j);
        
        for (unsigned int i = 1; i <= 6; i++)
          for (unsigned int j = 1; j <= 3; j++)
            noiseMatrix(i, 6 + j) = 0;
        for (unsigned int i = 1; i <= 3; i++)
          for (unsigned int j = 1; j <= 6; j++)
            noiseMatrix(6+i, j) = 0;
        
        filter_->setAdditiveMeasureNoiseRk(noiseMatrix /*measurementCount * stateCount*/);
      }
      else
      {
        imu_initialized_ = true;
        diagnostics_imu_rot_rel_ = 0;
      }
      imu_meas_old_ = imu_meas_;
    }
    // sensor not active
    else
      imu_initialized_ = false;

    //ROS_INFO("my_robot_pose_ekf: measurementZ: (x,x,x,x,x,%f,x,x,%f)", measurementZ(6),measurementZ(9));

    ColumnVector _estimateX;
    filter_->doStep(measurementZ, _estimateX);
    estimateX = _estimateX;

    //ROS_INFO("my_robot_pose_ekf: estimateX: (%f,%f,%f,%f,%f,%f)", _estimateX(1), _estimateX(2), _estimateX(3), _estimateX(4), _estimateX(5), _estimateX(6));

    // remember last estimate
    filter_estimate_old_vec_ = estimateX;
    tf::Quaternion q;
    q.setRPY(filter_estimate_old_vec_(4), filter_estimate_old_vec_(5), filter_estimate_old_vec_(6));
    filter_estimate_old_ = Transform(q,
                                     Vector3(filter_estimate_old_vec_(1), filter_estimate_old_vec_(2), filter_estimate_old_vec_(3)));
    filter_time_old_ = filter_time;
    addMeasurement(StampedTransform(filter_estimate_old_, filter_time, output_frame_, base_footprint_frame_));

    return true;
  };

  void OdomEstimation::addMeasurement(const StampedTransform &meas)
  {
    ROS_DEBUG("AddMeasurement from %s to %s:  (%f, %f, %f)  (%f, %f, %f, %f)",
              meas.frame_id_.c_str(), meas.child_frame_id_.c_str(),
              meas.getOrigin().x(), meas.getOrigin().y(), meas.getOrigin().z(),
              meas.getRotation().x(), meas.getRotation().y(),
              meas.getRotation().z(), meas.getRotation().w());
    transformer_.setTransform(meas);
  }

  void OdomEstimation::addMeasurement(const StampedTransform &meas, const MatrixWrapper::SymmetricMatrix &covar)
  {
    // check covariance
    for (unsigned int i = 0; i < covar.rows(); i++)
    {
      if (covar(i + 1, i + 1) == 0)
      {
        ROS_ERROR("Covariance specified for measurement on topic %s is zero", meas.child_frame_id_.c_str());
        return;
      }
    }
    // add measurements
    addMeasurement(meas);
    if (meas.child_frame_id_ == "wheelodom")
      odom_covariance_ = covar;
    else if (meas.child_frame_id_ == "imu")
      imu_covariance_ = covar;
    else if (meas.child_frame_id_ == "vo")
      vo_covariance_ = covar;
    else if (meas.child_frame_id_ == "gps")
      gps_covariance_ = covar;
    else
      ROS_ERROR("Adding a measurement for an unknown sensor %s", meas.child_frame_id_.c_str());
  };

  // get latest filter posterior as vector
  void OdomEstimation::getEstimate(MatrixWrapper::ColumnVector &estimate)
  {
    estimate = filter_estimate_old_vec_;
  };

  // get filter posterior at time 'time' as Transform
  void OdomEstimation::getEstimate(Time time, Transform &estimate)
  {
    StampedTransform tmp;
    if (!transformer_.canTransform(base_footprint_frame_, output_frame_, time))
    {
      ROS_ERROR("Cannot get transform at time %f", time.toSec());
      return;
    }
    transformer_.lookupTransform(output_frame_, base_footprint_frame_, time, tmp);
    estimate = tmp;
  };

  // get filter posterior at time 'time' as Stamped Transform
  void OdomEstimation::getEstimate(Time time, StampedTransform &estimate)
  {
    if (!transformer_.canTransform(output_frame_, base_footprint_frame_, time))
    {
      ROS_ERROR("Cannot get transform at time %f", time.toSec());
      return;
    }
    transformer_.lookupTransform(output_frame_, base_footprint_frame_, time, estimate);
  };

  // get most recent filter posterior as PoseWithCovarianceStamped
  void OdomEstimation::getEstimate(geometry_msgs::PoseWithCovarianceStamped &estimate)
  {
    // pose
    StampedTransform tmp;
    if (!transformer_.canTransform(output_frame_, base_footprint_frame_, ros::Time()))
    {
      ROS_ERROR("Cannot get transform at time %f", 0.0);
      return;
    }
    transformer_.lookupTransform(output_frame_, base_footprint_frame_, ros::Time(), tmp);
    poseTFToMsg(tmp, estimate.pose.pose);

    // header
    estimate.header.stamp = tmp.stamp_;
    estimate.header.frame_id = output_frame_;

    // covariance
    //SymmetricMatrix covar = filter_->PostGet()->CovarianceGet();
    SymmetricMatrix covar(6);
    filter_->getPostCovariance(covar);
    for (unsigned int i = 0; i < 6; i++)
      for (unsigned int j = 0; j < 6; j++)
        estimate.pose.covariance[6 * i + j] = covar(i + 1, j + 1);
  };

  // correct for angle overflow
  void OdomEstimation::angleOverflowCorrect(double &a, double ref)
  {
    while ((a - ref) > M_PI)
      a -= 2 * M_PI;
    while ((a - ref) < -M_PI)
      a += 2 * M_PI;
  };

  // decompose Transform into x,y,z,Rx,Ry,Rz
  void OdomEstimation::decomposeTransform(const StampedTransform &trans,
                                          double &x, double &y, double &z, double &Rx, double &Ry, double &Rz)
  {
    x = trans.getOrigin().x();
    y = trans.getOrigin().y();
    z = trans.getOrigin().z();
    trans.getBasis().getEulerYPR(Rz, Ry, Rx);
  };

  // decompose Transform into x,y,z,Rx,Ry,Rz
  void OdomEstimation::decomposeTransform(const Transform &trans,
                                          double &x, double &y, double &z, double &Rx, double &Ry, double &Rz)
  {
    x = trans.getOrigin().x();
    y = trans.getOrigin().y();
    z = trans.getOrigin().z();
    trans.getBasis().getEulerYPR(Rz, Ry, Rx);
  };

  void OdomEstimation::setOutputFrame(const std::string &output_frame)
  {
    output_frame_ = output_frame;
  };

  void OdomEstimation::setBaseFootprintFrame(const std::string &base_frame)
  {
    base_footprint_frame_ = base_frame;
  };

}; // namespace
