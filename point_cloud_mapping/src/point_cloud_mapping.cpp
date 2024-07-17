#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <signal.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>());
ros::Publisher map_pub;
ros::Publisher marker_pub;
float filter_size;
tf::TransformBroadcaster* tf_broadcaster;
std::string output_pcd_file;
std::string temp_dir;
size_t max_points_per_file = 5000000; 
size_t current_file_index = 0;

struct Pose {
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

std::vector<Pose> pose_vec;

Eigen::Matrix3d R_Tr;
Eigen::Vector3d t_Tr;
Eigen::Matrix4d Tr;
Eigen::Matrix4d Tr_inv;

void setupTransformation() {
    R_Tr << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03,
            -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01,
            9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03;
    t_Tr << -1.198459927713e-02, -5.403984729748e-02, -2.921968648686e-01;

    Tr << R_Tr(0,0), R_Tr(0,1), R_Tr(0,2), t_Tr(0),
          R_Tr(1,0), R_Tr(1,1), R_Tr(1,2), t_Tr(1),
          R_Tr(2,0), R_Tr(2,1), R_Tr(2,2), t_Tr(2),
          0, 0, 0, 1;

    Tr_inv = Tr.inverse();
}

void publishTFAndMarker(const Pose& pose, const ros::Time& timestamp, int index) {
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pose.t.x(), pose.t.y(), pose.t.z()));
    transform.setRotation(tf::Quaternion(pose.q.x(), pose.q.y(), pose.q.z(), pose.q.w()));
    std::string child_frame_id = "camera_init_" + std::to_string(index);
    ros::Time updated_timestamp = timestamp + ros::Duration(index * 0.0001); // Ensure each TF transform has a unique timestamp.
    tf_broadcaster->sendTransform(tf::StampedTransform(transform, updated_timestamp, "map", child_frame_id));

    // Marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = timestamp;
    marker.ns = "vehicle_pose";
    marker.id = index;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    //Avoid Marker overlap with TF coordinates.
    marker.pose.position.x = pose.t.x() - 2.0;
    marker.pose.position.y = pose.t.y() - 2.0;
    marker.pose.position.z = pose.t.z() - 2.0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.z = 1.5; 
    marker.color.a = 1.0; 
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.text = std::to_string(index); // Display the text content.

    marker_pub.publish(marker);
}

void saveCurrentMapWithPoses() {
    std::stringstream ss;
    ss << temp_dir << "/temp_map_" << current_file_index++ << ".pcd";
    std::cout << "Saving temporary map with poses to " << ss.str() << std::endl;

    pcl::PointCloud<pcl::PointXYZ> map_with_poses;
    map_with_poses = *global_map;

    for (const auto& pose : pose_vec) {
        pcl::PointXYZ pose_point;
        pose_point.x = pose.t.x();
        pose_point.y = pose.t.y();
        pose_point.z = pose.t.z();
        map_with_poses.push_back(pose_point);
    }

    pcl::io::savePCDFileBinary(ss.str(), map_with_poses);
    global_map->clear();
}

void callback(const nav_msgs::Odometry::ConstPtr& odom_msg, const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    Pose pose;
    pose.t = Eigen::Vector3d(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
    pose.q = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);

    Eigen::Matrix4d pose_T;
    pose_T.block<3, 3>(0, 0) = pose.q.toRotationMatrix();
    pose_T.block<3, 1>(0, 3) = pose.t;
    pose_T.block<1, 3>(3, 0) = Eigen::RowVector3d(0, 0, 0);
    pose_T(3, 3) = 1;

    Eigen::Matrix4d transformed_pose = pose_T * Tr;
    
    pose.t = transformed_pose.block<3, 1>(0, 3);
    pose.q = Eigen::Quaterniond(transformed_pose.block<3, 3>(0, 0));
    pose_vec.push_back(pose);

    // Publish all TFs and Markers.
    for (size_t i = 0; i < pose_vec.size(); ++i) {
        publishTFAndMarker(pose_vec[i], odom_msg->header.stamp, i);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Transform the point cloud using the current pose.
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = pose.q.toRotationMatrix().cast<float>();
    transform.block<3, 1>(0, 3) = pose.t.cast<float>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

    *global_map += *transformed_cloud;

    if (global_map->size() > max_points_per_file) {
        saveCurrentMapWithPoses();
    }

    // Apply voxel grid filter for downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(global_map);
    voxel_filter.setLeafSize(filter_size, filter_size, filter_size);
    voxel_filter.filter(*filtered_cloud);

    global_map = filtered_cloud;

    // Publish the global map
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*global_map, output);
    output.header.frame_id = "map";
    map_pub.publish(output);
}

void saveFinalMap(int signum) {
    if (!global_map->empty()) {
        saveCurrentMapWithPoses();
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr final_map(new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < current_file_index; ++i) {
        std::stringstream ss;
        ss << temp_dir << "/temp_map_" << i << ".pcd";
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(ss.str(), *temp_cloud) == -1) {
            std::cerr << "Couldn't read file " << ss.str() << std::endl;
            continue;
        }
        *final_map += *temp_cloud;
    }

    for (const auto& pose : pose_vec) {
        pcl::PointXYZ pose_point;
        pose_point.x = pose.t.x();
        pose_point.y = pose.t.y();
        pose_point.z = pose.t.z();
        final_map->push_back(pose_point);
    }

    std::cout << "Saving final global map to " << output_pcd_file << std::endl;
    pcl::io::savePCDFileBinary(output_pcd_file, *final_map);
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "point_cloud_mapping");
    ros::NodeHandle nh("~");

    tf_broadcaster = new tf::TransformBroadcaster;

    setupTransformation();

    std::string odom_topic, cloud_topic, map_topic;
    nh.param<std::string>("odom_topic", odom_topic, "/odometry_gt");
    nh.param<std::string>("cloud_topic", cloud_topic, "/velodyne_points");
    nh.param<std::string>("map_topic", map_topic, "/global_map");
    nh.param<float>("filter_size", filter_size, 0.8);
    nh.param<std::string>("output_pcd_file", output_pcd_file, "global_map.pcd");
    nh.param<std::string>("temp_dir", temp_dir, "/tmp");

    ROS_INFO("odom_topic: %s", odom_topic.c_str());
    ROS_INFO("cloud_topic: %s", cloud_topic.c_str());
    ROS_INFO("map_topic: %s", map_topic.c_str());
    ROS_INFO("filter_size: %f", filter_size);
    ROS_INFO("output_pcd_file: %s", output_pcd_file.c_str());
    ROS_INFO("temp_dir: %s", temp_dir.c_str());

    if (!boost::filesystem::exists(temp_dir)) {
        boost::filesystem::create_directory(temp_dir);
    }

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, odom_topic, 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, cloud_topic, 1);
    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, cloud_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    map_pub = nh.advertise<sensor_msgs::PointCloud2>(map_topic, 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("pose_markers", 1);

    // // Capture SIGINT signal (e.g., Ctrl+C) to save point cloud map.
    signal(SIGINT, saveFinalMap);

    ros::spin();

    delete tf_broadcaster;
    return 0;
}
