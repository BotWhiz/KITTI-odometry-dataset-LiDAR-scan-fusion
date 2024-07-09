cmake_minimum_required(VERSION 3.0.2)
project(point_cloud_mapping)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  tf
)

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs pcl_ros pcl_conversions tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(point_cloud_mapping src/point_cloud_mapping.cpp)

target_link_libraries(point_cloud_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_dependencies(point_cloud_mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

