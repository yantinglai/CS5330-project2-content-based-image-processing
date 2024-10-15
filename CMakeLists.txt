# Minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(Project2)

# Use C++11 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add all source files
add_executable(imgRetrieval
    imgRetrieval.cpp
    readfiles.cpp
    features.cpp
    csv_util.cpp
)

# Link OpenCV libraries
target_link_libraries(imgRetrieval ${OpenCV_LIBS})
