cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/../.venv")
set(Python3_EXECUTABLE "${Python3_ROOT_DIR}/Scripts/python.exe")
find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development)

list(APPEND CMAKE_PREFIX_PATH
  "${Python3_ROOT_DIR}/Lib/site-packages/pybind11/share/cmake/pybind11"
)
find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)

pybind11_add_module(cuda_kernels
  src/bindings.cu
  src/kernels.cu
)

set_target_properties(cuda_kernels PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/../py_modules"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../py_modules"
)