cmake_minimum_required(VERSION 3.5)
project(GraphCut LANGUAGES CXX)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/maxflow/*.cpp
                  ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${PROJECT_SOURCE_DIR}/maxflow)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
