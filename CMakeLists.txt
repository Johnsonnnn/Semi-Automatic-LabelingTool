cmake_minimum_required (VERSION 3.16)

project (SemiAutomaticLabelingTool)

set (executable_name SemiAutomaticLabelingTool)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV include at ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "Found OpenCV libraries: ${OpenCV_LIBRARIES}")
endif()

set (CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${OpenCV_INCLUDE_DIRS} includes)

add_executable ( ${executable_name}
    SemiAutomaticLabelingTool.cpp
    SemiAutomaticLabelingTool.h
)

target_link_libraries (${executable_name} ${OpenCV_LIBRARIES} yaml-cpp)
