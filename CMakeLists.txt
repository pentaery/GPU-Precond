cmake_minimum_required(VERSION 3.17.0)
project(GPU_Precond CXX CUDA)
find_package(CUDAToolkit)
add_subdirectory(src)
