cmake_minimum_required(VERSION 3.20)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_PREFIX}/lib")  
set(CMAKE_CUDA_ARCHITECTURES 89)

set(cugraph_DIR "~/miniconda3/envs/cugraph/lib/cmake/cugraph")
set(raft_DIR "~/miniconda3/envs/cugraph/lib/cmake/raft")
set(rmm_DIR "~/miniconda3/envs/cugraph/lib/cmake/rmm")

find_package(cugraph REQUIRED)
find_package(raft REQUIRED)
find_package(rmm REQUIRED)

add_executable(test
    test.cc
    matCPU.cc
    )


target_link_libraries(test
    PRIVATE
    cugraph::cugraph
    raft::raft
    rmm::rmm
    CUDA::cudart
)

target_include_directories(test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include_directories(BEFORE ${CUDA_INCLUDE_DIRS})
include_directories("/home/ET/yjzhou/HPCSoft/amgx/include")
target_link_libraries(test PRIVATE "/home/ET/yjzhou/HPCSoft/amgx/build/libamgxsh.so")


# target_include_directories(your_executable
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )