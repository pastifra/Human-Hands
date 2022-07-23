cmake_minimum_required(VERSION 2.8)

project(project)

find_package(OPENCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_executable(${PROJECT_NAME} source/segmentation_.cpp source/functions.cpp source/accuracy.cpp include/functions.h)

target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBS})
