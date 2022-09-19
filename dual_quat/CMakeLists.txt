cmake_minimum_required(VERSION 3.10) #这里指定了最低要求的cmake 版本
# set the project name 项目名
project(dual_quat)

find_package(CUDA REQUIRED)
MESSAGE("Find CUDA ${CUDA_FOUND}, path: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES}, def: ${CUDA_DEFINITIONS}")
MESSAGE("")
include_directories(${CUDA_INCLUDE_DIRS})




find_package(Eigen3 REQUIRED)
MESSAGE("Find Eigen3 ${Eigen3_FOUND}, path: ${Eigen3_INCLUDE_DIRS}, lib: ${Eigen3_LIBRARIES}, def: ${Eigen3_DEFINITIONS}")
MESSAGE("")
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")


include_directories("~/open3d_install/include")
link_directories("~/open3d_install/lib") 
link_libraries(Open3D)



# add the executable 添加可执行文件
add_executable(dual_quat usage.cpp)
target_link_libraries(dual_quat ~/open3d_install/lib/libOpen3D.so)