cmake_minimum_required(VERSION 3.10)
project(PCL_CV_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify PCL directories
set(PCL_DIR "/root/install/pcl")
include_directories(${PCL_DIR}/include/pcl-1.13)  # 使用适合您的PCL版本
include_directories(${PCL_DIR}/include/pcl-1.13/pcl)

# 添加PCL的库目录
link_directories(${PCL_DIR}/lib)

# Specify OpenCV include directory
include_directories(/opt/linux/sdk/include/opencv_3413)

# Specify OpenCV library directory
link_directories(/opt/linux/sdk/lib/opencv_3413)

# 查找PCL依赖的Boost、Eigen等库
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# 添加Eigen库
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    # 如果find_package找不到Eigen，使用常见的默认路径
    include_directories(/usr/include/eigen3)
endif()

add_executable(my-project test.cpp)

# Link against the specific OpenCV libraries you need
target_link_libraries(my-project
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    
    # PCL库，根据需要添加更多
    pcl_common
    pcl_io
    pcl_io_ply
    pcl_kdtree
    pcl_search
    pcl_features
    pcl_filters
    pcl_segmentation
    pcl_sample_consensus
    # pcl_visualization
    # PCL依赖库
    ${Boost_LIBRARIES}
)