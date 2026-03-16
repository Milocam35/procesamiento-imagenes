cmake_minimum_required(VERSION 3.10)
project(Taller1_Corte2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(Taller1_Corte2 ${SOURCES})

target_include_directories(Taller1_Corte2 PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Taller1_Corte2 ${OpenCV_LIBS})