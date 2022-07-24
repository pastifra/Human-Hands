cmake_minimum_required(VERSION 2.8)

project(project)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} source/main.cpp source/detection.cpp source/IntersectionOverUnion.cpp source/segmentation.cpp source/IOFunctions.cpp source/accuracy.cpp   include/IOfunctions.h include/detection.h include/segmentation.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
