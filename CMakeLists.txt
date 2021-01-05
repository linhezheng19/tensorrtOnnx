cmake_minimum_required(VERSION 3.10)

project(engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -g -fno-inline")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

message(STATUS "Project source: " ${PROJECT_SOURCE_DIR})

#-------------- Set variable for different ENV ----------#
if ("${CMAKE_SYSTEM}" MATCHES ".*tegra")  # on nx/nano/tx2
    set(TRT_ROOT /usr/local)
    set(CUDA_DIR /usr/local/cuda)
else()  # on GPU server
    set(TRT_ROOT /home/user/Software/TensorRT-7.1.3.4)  # set for user's install directory
    set(CUDA_DIR /usr/local/cuda-10.2)  # set for user's install directory
endif()

message(STATUS "TensorRT source: " ${TRT_ROOT})

#------------- Package ---------------------------------#
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else ()
    message(STATUS "CUDA Compiler not find")
endif()

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv not found")
endif (NOT OpenCV_FOUND)

#----------- Set architecture and CUDA -----------#
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_72,code=sm_72
        )

find_library(CUDART cudart HINTS ${CUDA_DIR}/lib64)

#----------- TensorRT ---------------------------#
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

#----------- Project source ---------------------#
# Common sources
include_directories(${TRT_ROOT}/include
                    ${CUDA_DIR}/include
                    /usr/local/include
                    ${CUDA_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/ops
                    ${PROJECT_SOURCE_DIR}/common/utils
                    )

file(GLOB COMMON_SRC ${PROJECT_SOURCE_DIR}/common/*.cpp
                     ${PROJECT_SOURCE_DIR}/common/ops/*.cpp
                     ${PROJECT_SOURCE_DIR}/common/utils/*.cpp
                     )

file(GLOB COMMON_CUDA_SRC ${PROJECT_SOURCE_DIR}/common/*.cu
                          ${PROJECT_SOURCE_DIR}/common/ops/*.cu
                          ${PROJECT_SOURCE_DIR}/common/utils/*.cu
                          )

# Tasks sources
include_directories(${PROJECT_SOURCE_DIR}/tasks/cls
                    ${PROJECT_SOURCE_DIR}/tasks/semseg
                    ${PROJECT_SOURCE_DIR}/tasks/fcos
                    ${PROJECT_SOURCE_DIR}/tasks/yolov5
                    ${PROJECT_SOURCE_DIR}/tasks/f_track
                    ${PROJECT_SOURCE_DIR}/tasks/fairmot
                    )

file(GLOB_RECURSE MODEL_SRC ${PROJECT_SOURCE_DIR}/tasks/*.cpp)

file(GLOB_RECURSE MODEL_CUDA_SRC ${PROJECT_SOURCE_DIR}/tasks/*.cu)

#---------- Library and Executable -----------#
link_directories(${TRT_ROOT}/lib
                 ${CUDA_DIR}/lib64
                 /usr/local/lib
                 )

#---------- CUDA Compiler --------------------#
# cuda_add_library(cuda_lib SHARED ${COMMON_CUDA_SRC} ${MODEL_CUDA_SRC})
cuda_add_executable(${PROJECT_NAME} main.cpp ${COMMON_CUDA_SRC} ${MODEL_CUDA_SRC} ${COMMON_SRC} ${MODEL_SRC})

#---------- G++ Compiler ---------------------#
# add_executable(${PROJECT_NAME} main.cpp ${COMMON_SRC} ${MODEL_SRC})

target_link_libraries(${PROJECT_NAME} nvinfer
                                      nvinfer_plugin
                                      nvparsers
                                      nvonnxparser
                                      nvcaffe_parser
                                      ${CUDART}
                                      ${OpenCV_LIBRARIES}
                                      yaml-cpp
                                    #   cuda_lib
                                      )
