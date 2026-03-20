cmake_minimum_required(VERSION 3.10)
project(DeteccionFigurasGeometricas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(DeteccionFigurasGeometricas ${SOURCES})

target_include_directories(DeteccionFigurasGeometricas PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DeteccionFigurasGeometricas ${OpenCV_LIBS})