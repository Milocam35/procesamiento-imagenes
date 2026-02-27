cmake_minimum_required(VERSION 3.10)
project(TallerImagenSatelital)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(TallerImagenSatelital ${SOURCES})

target_include_directories(TallerImagenSatelital PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(TallerImagenSatelital ${OpenCV_LIBS})