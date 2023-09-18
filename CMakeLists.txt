
cmake_minimum_required (VERSION 3.10.3)
project(OvOCR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(gflags)

file (GLOB SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
file (GLOB HDRS 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hppp
)

set (TARGET_NAME "OvOCR")

add_executable(${TARGET_NAME} ${SRCS} ${HDRS})

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

if (OpenVINO_FOUND)
        message(STATUS "OpenVINO Found")
endif()

list(APPEND INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} build/gflags/include)
list(APPEND DEPENDENCIES ${OpenCV_LIBS} openvino::runtime gflags)
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDES})
target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCIES})