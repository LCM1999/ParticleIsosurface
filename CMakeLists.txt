cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(COMPUTE_CAPABILITY 75)
set(CUDA_STANDARD 11)

set(PROJECT_NAME Isosurface_smiplify)
set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY})

include(CheckLanguage)
check_language(CUDA)


if(CMAKE_CUDA_COMPILER)
    message("Rendertoolengine found cuda")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # find_package(Thrust CONFIG)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    include_directories(${CUDA_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
    add_compile_definitions(__CUDACC__)  # 强制定义宏
endif()

################################################################################
# Find packages
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
if( UNIX )
    # set(MPI_DIR /usr/lib/x86_64-linux-gnu/openmpi)
    # set(HDF5_DIR /home/lcm/h5/CMake-hdf5-1.14.2/build/HDF5-1.14.2-Linux/HDF_Group/HDF5/1.14.2/cmake)
    # set(EIGEN3_DIR /home/lcm/eigen/eigen-3.4.0/build)
endif()

find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found!")
endif()

# set(HDF5_DIR E:/hdf5/HDF5-1.14.3-win64/cmake)
set(HDF5_PREFER_PARALLEL "ON")
find_package(HDF5 REQUIRED COMPONENTS C HL)
# find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS C static HL)
if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif ()

#find_package(Eigen3 REQUIRED)
#if(NOT EIGEN3_FOUND)
#    message(FATAL_ERROR "Eigen3 not found!")
#endif()

find_package(VTK REQUIRED)
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found!")
endif()

################################################################################
# Target
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options("-O3")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options("-openmp:llvm")
endif()

file(GLOB ORIGINAL_HEADER_FILES
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/*.cuh" 
)

file(GLOB ORIGINAL_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cu"
  "${PROJECT_SOURCE_DIR}/src/*.c"
)

file(GLOB ISO_HEADER_FILES
  "${PROJECT_SOURCE_DIR}/iso/include/*.h"
  "${PROJECT_SOURCE_DIR}/iso/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/iso/include/*.cuh"
)

file(GLOB ISO_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/iso/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/iso/src/*.cu"
  "${PROJECT_SOURCE_DIR}/iso/src/*.c"
)

file(GLOB CSTONEOCTREE_HEADER_FILES
  "${PROJECT_SOURCE_DIR}/cstoneOctree/include/*.h"
  "${PROJECT_SOURCE_DIR}/cstoneOctree/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/cstoneOctree/include/*.cuh"
)

file(GLOB CSTONEOCTREE_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/cstoneOctree/src/*.c"
  "${PROJECT_SOURCE_DIR}/cstoneOctree/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/cstoneOctree/src/*.cu"
)

set(ORIGINAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(ISO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/iso/include")
set(CSTONEOCTREE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cstoneOctree/include")

add_executable(${PROJECT_NAME}
    main.cpp
    ${ORIGINAL_SOURCE_FILES}
    ${CSTONEOCTREE_SOURCE_FILES}
    ${ISO_SOURCE_FILES}
)

include_directories(
    pugixml/src
    "${VTK_USE_FILE}"
    ${ORIGINAL_INCLUDE_DIR}
    ${ISO_INCLUDE_DIR}
    ${CSTONEOCTREE_INCLUDE_DIR}
    )

add_library(pugixml
    STATIC
    pugixml/src/pugixml.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    ${ORIGINAL_INCLUDE_DIR}
    ${ISO_INCLUDE_DIR}
    ${CSTONEOCTREE_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIR}
    #${EIGEN3_INCLUDE_DIR}
)

# cuda_add_library(node_calc_lib SHARED src/node_calc.cu)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${HDF5_C_STATIC_LIBRARY}
    ${VTK_LIBRARIES}
    #Eigen3::Eigen
    OpenMP::OpenMP_CXX
    HDF5::HDF5
    pugixml
    CUDA::cudart
    ${CUDA_LIBRARIES}
)


# add unit_test
# add_subdirectory(unit_test)