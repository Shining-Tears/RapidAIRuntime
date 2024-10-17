cmake_minimum_required(VERSION 3.18)
project(RapidAIRuntime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")
set(CUDA_INCLUDE "/usr/local/cuda-12.3/include")
file(GLOB EXT_LIBS "${LIB_DIR}/tensorrt8616/lib/*.so")
# set(EXT_LIBS "${LIB_DIR}/tensorrt8616/lib/nvinfer_plugin.lib" 
#     "${LIB_DIR}/tensorrt8616/lib/nvinfer.lib" 
#     "${LIB_DIR}/tensorrt8616/lib/nvonnxparser.lib" 
#     "${LIB_DIR}/tensorrt8616/lib/nvparsers.lib" 
#     "${LIB_DIR}/tensorrt8616/lib/nvinfer_vc_plugin.lib"
#     "${LIB_DIR}/tensorrt8616/lib/nvinfer_dispatch.lib"
#     "${LIB_DIR}/tensorrt8616/lib/nvinfer_lean.lib")

file(GLOB SOURCE_CODE "${CMAKE_CURRENT_SOURCE_DIR}/RapidAIRuntime/src/*.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/RapidAIRuntime/src/*.cu")

add_library(RapidAIRuntime SHARED ${SOURCE_CODE})

target_include_directories(RapidAIRuntime PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/RapidAIRuntime/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/tensorrt8616/include
    ${CUDA_INCLUDE})

target_link_libraries(RapidAIRuntime PRIVATE
    ${EXT_LIBS} 
    ${CUDA_LIBRARIES})