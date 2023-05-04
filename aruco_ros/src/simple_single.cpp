/*****************************
 Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are
 permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list
 of conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
 WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those of the
 authors and should not be interpreted as representing official policies, either expressed
 or implied, of Rafael Muñoz Salinas.
 ********************************/
/**
 * @file simple_single.cpp
 * @author Bence Magyar
 * @date June 2012
 * @version 0.1
 * @brief ROS version of the example named "simple" in the ArUco software package.
 */

#include <iostream>

#include "aruco/aruco.h"
#include "aruco/cvdrawingutils.h"
#include "aruco_ros/aruco_ros_utils.hpp"

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcpputils/asserts.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "capella_ros_service_interfaces/msg/charge_marker_visible.hpp"
#include "Eigen/Dense"
#include "nav_msgs/msg/odometry.hpp"


class ArucoSimple : public rclcpp::Node
{
private:
  rclcpp::Node::SharedPtr subNode;
  cv::Mat inImage;
  aruco::CameraParameters camParam;
  tf2::Stamped<tf2::Transform> rightToLeft;
  bool useRectifiedImages;
  aruco::MarkerDetector mDetector;
  std::vector<aruco::Marker> markers;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
  bool cam_info_received;
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_pub;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr position_pub;
  // rviz visualization marker
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pixel_pub;
  rclcpp::Publisher<capella_ros_service_interfaces::msg::ChargeMarkerVisible>::SharedPtr detect_status;
  rclcpp::TimerBase::SharedPtr marker_timer;  
  std::string marker_frame;
  std::string camera_frame;
  std::string reference_frame;

  double marker_size;
  int marker_id;

  std::unique_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber image_sub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  double linear_x_;
  double angular_z_;
  rclcpp::Time vel_time_;
  double intervel;
  bool init_first_time;
  int p_uncertain, r_uncertain;
  bool odom_pub_;
  struct kalman_info
  {
	Eigen::Vector4d kalman_output;  
	Eigen::Matrix4d kalman_gain;   
	Eigen::Matrix4d A;   
  Eigen::Vector4d B;    
	Eigen::Matrix4d H;   
	Eigen::Matrix4d Q;   
	Eigen::Matrix4d R;   
	Eigen::Matrix4d P;   
  double u;
  };
  kalman_info camera_pose_info;
  void Init_kalman_info(kalman_info* info, double t, Eigen::Vector4d measurement)
  {
    info->A << 1,0,t,0, 0,1,0,t, 0,0,1,0, 0,0,0,1; 
    info->B << pow(t, 2)/2, pow(t, 2)/2, t, t;
    info->Q << pow(t, 2)/4,0,pow(t, 3)/2,0, 0,pow(t, 2)/4,0,pow(t, 3)/2, pow(t, 3)/2,0,pow(t, 2),0, 0,pow(t, 3)/2,0,pow(t, 2); //预测方差
    if (init_first_time)
    {
      info->P.Identity();  //后验状态估计值方差的初值
      info->P *= p_uncertain;
      info->R << 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1; //观测噪声方差
      info->R *= r_uncertain;
      info->H << 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1;
      info->kalman_output << measurement;
      init_first_time = false;
      // 测量的初始值
      info->u = 1e-3;
    }
     
  }

  void marker_visible_callback()
  {
    if (markers.size() == 0)
    {
      capella_ros_service_interfaces::msg::ChargeMarkerVisible marker_detect_status;
      marker_detect_status.marker_visible = false;
      detect_status->publish(marker_detect_status);
    }
    else if (markers.size() > 0)
    {
      for (int i = 0; i < markers.size(); i++)
      {
        if (markers[i].id == marker_id)
        {
          capella_ros_service_interfaces::msg::ChargeMarkerVisible marker_detect_status;
          marker_detect_status.marker_visible = true;
          detect_status->publish(marker_detect_status);
        }
      }
    }

    
  }
  void odom_callback(const nav_msgs::msg::Odometry &odom_sub)
  {
    linear_x_ = odom_sub.twist.twist.linear.x;
    angular_z_ = odom_sub.twist.twist.angular.z;
    vel_time_ = odom_sub.header.stamp;
    odom_pub_ = true;
  }
  
  

public:
  ArucoSimple()
  : Node("aruco_single"), cam_info_received(false)
  {
    detect_status = this->create_publisher<capella_ros_service_interfaces::msg::ChargeMarkerVisible>("marker_visible", 1);
    marker_visible = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&ArucoSimple::marker_visible_callback, this));
    linear_x_ = 0;
    angular_z_ = 0;
    intervel = 0.1;
    init_first_time = true;
    odom_pub_ = false;
  }

  bool setup()
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    subNode = this->create_sub_node(this->get_name());

    it_ = std::make_unique<image_transport::ImageTransport>(shared_from_this());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    if (this->has_parameter("corner_refinement")) {
      RCLCPP_WARN(
        this->get_logger(),
        "Corner refinement options have been removed in ArUco 3.0.0, "
        "corner_refinement ROS parameter is deprecated");
    }

    aruco::MarkerDetector::Params params = mDetector.getParameters();
    std::string thresh_method;
    switch (params.thresMethod) {
      case aruco::MarkerDetector::ThresMethod::THRES_ADAPTIVE:
        thresh_method = "THRESH_ADAPTIVE";
        break;
      case aruco::MarkerDetector::ThresMethod::THRES_AUTO_FIXED:
        thresh_method = "THRESH_AUTO_FIXED";
        break;
      default:
        thresh_method = "UNKNOWN";
        break;
    }

    // Print parameters of ArUco marker detector:
    RCLCPP_INFO_STREAM(this->get_logger(), "Threshold method: " << thresh_method);

    // Declare node parameters
    this->declare_parameter<double>("marker_size", 0.05);
    this->declare_parameter<int>("marker_id", 300);
    this->declare_parameter<std::string>("reference_frame", "");
    this->declare_parameter<std::string>("camera_frame", "");
    this->declare_parameter<std::string>("marker_frame", "");
    this->declare_parameter<bool>("image_is_rectified", true);
    this->declare_parameter<float>("min_marker_size", 0.02);
    this->declare_parameter<std::string>("detection_mode", "");

    this->declare_parameter<int>("P_uncertain", 1);
    this->declare_parameter<int>("R_uncertain", 1);

    
    this->get_parameter_or<int>("P_uncertain", p_uncertain, 1);
    this->get_parameter_or<int>("R_uncertain", r_uncertain, 1);

    float min_marker_size;  // percentage of image area
    this->get_parameter_or<float>("min_marker_size", min_marker_size, 0.02);

    std::string detection_mode;
    this->get_parameter_or<std::string>("detection_mode", detection_mode, "DM_FAST");
    if (detection_mode == "DM_FAST") {
      mDetector.setDetectionMode(aruco::DM_FAST, min_marker_size);
    } else if (detection_mode == "DM_VIDEO_FAST") {
      mDetector.setDetectionMode(aruco::DM_VIDEO_FAST, min_marker_size);
    } else {
      // Aruco version 2 mode
      mDetector.setDetectionMode(aruco::DM_NORMAL, min_marker_size);
    }

    RCLCPP_INFO_STREAM(
      this->get_logger(), "Marker size min: " << min_marker_size << " of image area");
    RCLCPP_INFO_STREAM(this->get_logger(), "Detection mode: " << detection_mode);
    image_sub = it_->subscribe("/image", 1, &ArucoSimple::image_callback, this);
    cam_info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info",  rclcpp::QoS{1}.best_effort(), std::bind(
        &ArucoSimple::cam_info_callback, this,
        std::placeholders::_1));

    odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("odom", 1, std::bind(&ArucoSimple::odom_callback, this, std::placeholders::_1));

    image_pub = it_->advertise(this->get_name() + std::string("/result"), 1);
    debug_pub = it_->advertise(this->get_name() + std::string("/debug"), 1);
    pose_pub = subNode->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 100);
    transform_pub =
      subNode->create_publisher<geometry_msgs::msg::TransformStamped>("transform", 100);
    position_pub = subNode->create_publisher<geometry_msgs::msg::Vector3Stamped>("position", 100);
    marker_pub = subNode->create_publisher<visualization_msgs::msg::Marker>("marker", 10);
    pixel_pub = subNode->create_publisher<geometry_msgs::msg::PointStamped>("pixel", 10);



    this->get_parameter_or<double>("marker_size", marker_size, 0.05);
    this->get_parameter_or<int>("marker_id", marker_id, 300);
    this->get_parameter_or<std::string>("reference_frame", reference_frame, "");
    this->get_parameter_or<std::string>("camera_frame", camera_frame, "");
    this->get_parameter_or<std::string>("marker_frame", marker_frame, "");
    this->get_parameter_or<bool>("image_is_rectified", useRectifiedImages, true);

    detect_status = this->create_publisher<capella_ros_service_interfaces::msg::ChargeMarkerVisible>("marker_visible", 10);
    marker_timer = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&ArucoSimple::marker_timer_callback, this));

    rcpputils::assert_true(
      camera_frame != "" && marker_frame != "",
      "Found the camera frame or the marker_frame to be empty!. camera_frame : " +
      camera_frame + " and marker_frame : " + marker_frame);

    if (reference_frame.empty()) {
      reference_frame = camera_frame;
    }

    RCLCPP_INFO(
      this->get_logger(), "ArUco node started with marker size of %f m and marker id to track: %d",
      marker_size, marker_id);
    RCLCPP_INFO(
      this->get_logger(), "ArUco node will publish pose to TF with %s as parent and %s as child.",
      reference_frame.c_str(), marker_frame.c_str());

    // dyn_rec_server.setCallback(boost::bind(&ArucoSimple::reconf_callback, this, _1, _2));
    RCLCPP_INFO(this->get_logger(), "Setup of aruco_simple node is successful!");
    return true;
  }

  bool getTransform(
    const std::string & refFrame, const std::string & childFrame,
    geometry_msgs::msg::TransformStamped & transform)
  {
    std::string errMsg;

    if (!tf_buffer_->canTransform(
        refFrame, childFrame, tf2::TimePointZero,
        tf2::durationFromSec(0.5), &errMsg))
    {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Unable to get pose from TF: " << errMsg);
      return false;
    } else {
      try {
        transform = tf_buffer_->lookupTransform(
          refFrame, childFrame, tf2::TimePointZero, tf2::durationFromSec(
            0.5));
      } catch (const tf2::TransformException & e) {
        RCLCPP_ERROR_STREAM(
          this->get_logger(),
          "Error in lookupTransform of " << childFrame << " in " << refFrame << " : " << e.what());
        return false;
      }
    }
    return true;
  }

  void kalman_filter(kalman_info* kalman_info, Eigen::Vector4d last_measurement)
  {
    //预测下一时刻的值
    auto predict_value = kalman_info->A * kalman_info->kalman_output + kalman_info->B * kalman_info->u;   //x的先验估计由上一个时间点的后验估计值和输入信息给出，此处需要根据基站高度做一个修改
    
    //求协方差
    kalman_info->P = kalman_info->A * kalman_info->P * kalman_info->A.transpose() + kalman_info->Q;  //计算先验均方差 p(n|n-1)=A^2*p(n-1|n-1)+q  
    //计算kalman增益
    kalman_info->kalman_gain = kalman_info->P * kalman_info->H.transpose() * (kalman_info->H * kalman_info->P * kalman_info->H.transpose() + kalman_info->R).inverse();  //Kg(k)= P(k|k-1) H’ / (H P(k|k-1) H’ + R)
    //修正结果，即计算滤波值
    kalman_info->kalman_output = predict_value + kalman_info->kalman_gain * (last_measurement - kalman_info->H * predict_value);  //利用残余的信息改善对x(t)的估计，给出后验估计，这个值也就是输出  X(k|k)= X(k|k-1)+Kg(k) (Z(k)-H X(k|k-1))
    //更新后验估计
    kalman_info->P = (kalman_info->P - kalman_info->kalman_gain * kalman_info->H) * kalman_info->P;//计算后验均方差  P[n|n]=(1-K[n]*H)*P[n|n-1]
  }

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    if ((image_pub.getNumSubscribers() == 0) &&
      (debug_pub.getNumSubscribers() == 0) &&
      (pose_pub->get_subscription_count() == 0) &&
      (transform_pub->get_subscription_count() == 0) &&
      (position_pub->get_subscription_count() == 0) &&
      (marker_pub->get_subscription_count() == 0) &&
      (pixel_pub->get_subscription_count() == 0))
    {
      RCLCPP_DEBUG(this->get_logger(), "No subscribers, not looking for ArUco markers");
      // return;
    }

    if (cam_info_received) {
      builtin_interfaces::msg::Time curr_stamp = msg->header.stamp;
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::RGB8);
        inImage = cv_ptr->image;

        // detection results will go into "markers"
        markers.clear();
        // ok, let's detect
        mDetector.detect(inImage, markers, camParam, marker_size, false);
        // for each marker, draw info and its boundaries in the image
        for (std::size_t i = 0; i < markers.size(); ++i) {
          // only publishing the selected marker
          if (markers[i].id == marker_id) {
            tf2::Transform transform = aruco_ros::arucoMarker2Tf2(markers[i]);
            tf2::Stamped<tf2::Transform> cameraToReference;
            cameraToReference.setIdentity();
            tf2::Quaternion q;
            q.setRPY(M_PI/2, 0.0, M_PI/2);
            cameraToReference.setRotation(q);

            if (reference_frame != camera_frame) {
              geometry_msgs::msg::TransformStamped transform_stamped;
              getTransform(reference_frame, camera_frame, transform_stamped);
              tf2::fromMsg(transform_stamped, cameraToReference);
            }

            transform = static_cast<tf2::Transform>(cameraToReference) *
              static_cast<tf2::Transform>(rightToLeft) *
              transform;
            
            if (odom_pub_)
            {
              intervel = (rclcpp::Clock().now().seconds() - vel_time_.seconds());
              Eigen::Vector4d pose_and_vel_mea;
              pose_and_vel_mea << transform.getOrigin()[0], transform.getRotation()[2], linear_x_, angular_z_;
              Init_kalman_info(&camera_pose_info, intervel, pose_and_vel_mea);
              kalman_filter(&camera_pose_info, pose_and_vel_mea);
              transform.getOrigin()[0] = camera_pose_info.kalman_output[0];
              transform.getRotation()[2] = camera_pose_info.kalman_output[1];
              odom_pub_ = false;
            }
            else
            {
              RCLCPP_WARN(this->get_logger(), "No odom pose and vel received");
            }
            
            
            geometry_msgs::msg::TransformStamped stampedTransform;
            stampedTransform.header.frame_id = reference_frame;
            stampedTransform.header.stamp = curr_stamp;
            stampedTransform.child_frame_id = marker_frame;
            tf2::toMsg(transform, stampedTransform.transform);
            tf_broadcaster_->sendTransform(stampedTransform);
            geometry_msgs::msg::PoseStamped poseMsg;
            poseMsg.header = stampedTransform.header;
            tf2::toMsg(transform, poseMsg.pose);
            poseMsg.header.frame_id = reference_frame;
            poseMsg.header.stamp = curr_stamp;
            pose_pub->publish(poseMsg);

            transform_pub->publish(stampedTransform);

            geometry_msgs::msg::Vector3Stamped positionMsg;
            positionMsg.header = stampedTransform.header;
            positionMsg.vector = stampedTransform.transform.translation;
            position_pub->publish(positionMsg);

            geometry_msgs::msg::PointStamped pixelMsg;
            pixelMsg.header = stampedTransform.header;
            pixelMsg.point.x = markers[i].getCenter().x;
            pixelMsg.point.y = markers[i].getCenter().y;
            pixelMsg.point.z = 0;
            pixel_pub->publish(pixelMsg);

            // publish rviz marker representing the ArUco marker patch
            visualization_msgs::msg::Marker visMarker;
            visMarker.header = stampedTransform.header;
            visMarker.id = 1;
            visMarker.type = visualization_msgs::msg::Marker::CUBE;
            visMarker.action = visualization_msgs::msg::Marker::ADD;
            visMarker.pose = poseMsg.pose;
            visMarker.scale.x = marker_size;
            visMarker.scale.y = marker_size;
            visMarker.scale.z = 0.001;
            visMarker.color.r = 1.0;
            visMarker.color.g = 0;
            visMarker.color.b = 0;
            visMarker.color.a = 1.0;
            visMarker.lifetime = builtin_interfaces::msg::Duration();
            visMarker.lifetime.sec = 3;
            marker_pub->publish(visMarker);
          }
          
          
          // but drawing all the detected markers
          markers[i].draw(inImage, cv::Scalar(0, 0, 255), 2);
        }

        // draw a 3d cube in each marker if there is 3d info
        if (camParam.isValid() && marker_size > 0) {
          for (std::size_t i = 0; i < markers.size(); ++i) {
            aruco::CvDrawingUtils::draw3dAxis(inImage, markers[i], camParam);
          }
        }

        if (image_pub.getNumSubscribers() > 0) {
          // show input with augmented information
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = curr_stamp;
          out_msg.encoding = sensor_msgs::image_encodings::RGB8;
          out_msg.image = inImage;
          image_pub.publish(out_msg.toImageMsg());
        }

        if (debug_pub.getNumSubscribers() > 0) {
          // show also the internal image resulting from the threshold operation
          cv_bridge::CvImage debug_msg;
          debug_msg.header.stamp = curr_stamp;
          debug_msg.encoding = sensor_msgs::image_encodings::MONO8;
          debug_msg.image = mDetector.getThresholdedImage();
          debug_pub.publish(debug_msg.toImageMsg());
        }
      } catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
      }
    }
  }

  // wait for one camerainfo, then shut down that subscriber
  void cam_info_callback(const sensor_msgs::msg::CameraInfo & msg)
  {
    if (!cam_info_received) {
      camParam = aruco_ros::rosCameraInfo2ArucoCamParams(msg, useRectifiedImages);

      // handle cartesian offset between stereo pairs
      // see the sensor_msgs/CameraInfo documentation for details
      rightToLeft.setIdentity();
      rightToLeft.setOrigin(tf2::Vector3(-msg.p[3] / msg.p[0], -msg.p[7] / msg.p[5], 0.0));

      cam_info_received = true;
    }
  }

//  void reconf_callback(aruco_ros::ArucoThresholdConfig & config, uint32_t level)
//  {
//    mDetector.setDetectionMode(
//      aruco::DetectionMode(config.detection_mode),
//      config.min_image_size);
//    if (config.normalizeImage) {
//      RCLCPP_WARN("normalizeImageIllumination is unimplemented!");
//    }
//  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<ArucoSimple> aruco_simple = std::make_shared<ArucoSimple>();
  aruco_simple->setup();
  rclcpp::spin(aruco_simple);
  rclcpp::shutdown();
}
