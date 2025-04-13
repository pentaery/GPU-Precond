cmake_minimum_required(VERSION 3.18)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



message(STATUS "Looking for mfem ...")
set(MFEM_DIR "~/HPCSoft/mfem-4.7" CACHE PATH "Path to the MFEM build or install prefix.")
if (MFEM_DIR)
   find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}" "${MFEM_DIR}/build" NO_DEFAULT_PATH)
else()
   find_package(mfem REQUIRED NAMES MFEM)
endif()

# --- 检查 MFEM 是否支持 CUDA ---
if (NOT MFEM_USE_CUDA)
   message(WARNING "MFEM was not compiled with CUDA support! Rebuild MFEM with -D MFEM_USE_CUDA=ON")
else()
   message(STATUS "MFEM CUDA support enabled.")
endif()

# --- 设置 CUDA 相关编译选项 ---
if (MFEM_USE_CUDA)
   # 确保 CUDA 相关的头文件和库被包含
   include_directories(${CUDA_INCLUDE_DIRS})
   list(APPEND MFEM_LIBRARIES ${CUDA_LIBRARIES})
endif()

# --- 使用 MFEM 的编译器（如果 MFEM 是用 MPI 编译的） ---
message(STATUS "MFEM compiler is: ${MFEM_CXX_COMPILER}")
if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)
   set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")
endif()



add_executable(matgen test.cc)
include_directories(${MFEM_INCLUDE_DIRS})
target_link_libraries(matgen PRIVATE ${MFEM_LIBRARIES})





