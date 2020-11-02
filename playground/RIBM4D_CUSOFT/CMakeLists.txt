cmake_minimum_required(VERSION 2.8)
project(bm4d-gpu LANGUAGES CXX CUDA)

## Add support for C++14
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED core video videoio highgui)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CXX
        src/allreader.cpp
        src/main.cpp
        )
set(CU  src/kernels.cu
        src/cusoft_kernels.cu
        src/bm4d.cu
        src/cusoft_lib/CUcospmls.cu
        src/cusoft_lib/CUs2_primitive.cu
        src/cusoft_lib/CUnewFCT.cu
        src/cusoft_lib/CUOURperms.cu
        src/cusoft_lib/CUOURmods.cu
        )
set(H
        include/bm4d-gpu/allreader.h
        include/bm4d-gpu/bm4d.h
        include/bm4d-gpu/helper_cuda.h
        include/bm4d-gpu/helper_string.h
        include/bm4d-gpu/parameters.h
        include/bm4d-gpu/stopwatch.hpp
        include/bm4d-gpu/utils.h
        include/bm4d-gpu/kernels.cuh
        include/cusoft/cusoft_kernels.cuh
        include/cusoft/CUcospmls.cuh
        include/cusoft/CUs2_primitive.cuh
        include/cusoft/CUnewFCT.cuh
        include/cusoft/CUOURperms.cuh
        include/cusoft/CUOURmods.cuh
        )

add_library(ribm4d STATIC

        src/kernels.cu
        src/cusoft_kernels.cu
        src/bm4d.cu
        
        src/cusoft_lib/CUcospmls.cu
        src/cusoft_lib/CUs2_primitive.cu
        src/cusoft_lib/CUnewFCT.cu
        src/cusoft_lib/CUOURperms.cu
        src/cusoft_lib/CUOURmods.cu

        include/bm4d-gpu/allreader.h
        include/bm4d-gpu/bm4d.h
        include/bm4d-gpu/helper_cuda.h
        include/bm4d-gpu/helper_string.h
        include/bm4d-gpu/parameters.h
        include/bm4d-gpu/stopwatch.hpp
        include/bm4d-gpu/utils.h
        include/bm4d-gpu/kernels.cuh
        include/cusoft/cusoft_kernels.cuh
        include/cusoft/CUcospmls.cuh
        include/cusoft/CUs2_primitive.cuh
        include/cusoft/CUnewFCT.cuh
        include/cusoft/CUOURperms.cuh
        include/cusoft/CUOURmods.cuh

        src/allreader.cpp
)

set_target_properties(ribm4d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# cuda_add_executable(bm4d-gpu ${CXX} ${CU} ${H})
cuda_add_executable(bm4d-gpu src/main.cpp)
set_target_properties(bm4d-gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bm4d-gpu
                        ribm4d
                        ${OpenCV_LIBRARIES})
