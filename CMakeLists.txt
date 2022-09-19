cmake_minimum_required(VERSION 3.10) #这里指定了最低要求的cmake 版本
# set the project name 项目名
project(dual_quat)
# add the executable 添加可执行文件
add_executable(dual_quat usage.cpp)