cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# 添加可执行文件
cuda_add_executable(my_project main.cu)

# 链接 CUDA 库（如果需要）
target_link_libraries(my_project ${CUDA_LIBRARIES})