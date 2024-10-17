cmake_minimum_required(VERSION 3.18)
project(RapidAIRuntimeTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(GRANDPARENT_DIR "${PARENT_DIR}" DIRECTORY)

message("Current directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message("Parent directory: ${PARENT_DIR}")
message("GrandParent directory: ${GRANDPARENT_DIR}")

find_package(CUDAToolkit REQUIRED)

set(CUDA_INCLUDE "/usr/local/cuda-12.3/include")
set(TRT_LIB_DIR "${GRANDPARENT_DIR}/3rd_party/tensorrt8616")
set(CUDNN_LIB_DIR "${GRANDPARENT_DIR}/3rd_party/cudnn89")
set(TEST_LIB_DIR "${GRANDPARENT_DIR}/3rd_party/googletest")
file(GLOB EXT_LIBS "${TRT_LIB_DIR}/lib/*" 
                   "${TEST_LIB_DIR}/lib/*"
                   "${CUDNN_LIB_DIR}/lib/*")

file(GLOB SOURCE_CODE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
                      "${PARENT_DIR}/src/*.cu" 
                      "${PARENT_DIR}/src/*.cpp")

add_executable(RapidAIRuntimeTest ${SOURCE_CODE})

target_include_directories(RapidAIRuntimeTest PRIVATE 
    ${PARENT_DIR}/include 
    ${GRANDPARENT_DIR}/3rd_party/tensorrt8616/include 
    ${GRANDPARENT_DIR}/3rd_party/googletest/include
    ${CUDA_INCLUDE})

target_link_libraries(RapidAIRuntimeTest PRIVATE
    ${EXT_LIBS} 
    ${CUDA_LIBRARIES})