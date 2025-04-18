cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(COMPUTE_CAPABILITY 75)
set(CUDA_STANDARD 11)

set(CMAKE_PROJECT_NAME Isosurface_smiplify)
set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message("Rendertoolengine found cuda")
    # set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # find_package(Thrust CONFIG)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    # include_directories(${CUDA_INCLUDE_DIR})
    # include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

################################################################################
# Find packages
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found!")
endif()

find_package(Eigen3 REQUIRED)
if(NOT EIGEN3_FOUND)
   message(FATAL_ERROR "Eigen3 not found!")
endif()

set(HDF5_PREFER_PARALLEL "ON")
find_package(HDF5 REQUIRED COMPONENTS C HL)
if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif ()

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
    # add_compile_options("-openmp:llvm")
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
set(CSTONEOCTREE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cstoneOctree/include")

add_executable(${PROJECT_NAME}
    main.cpp
    ${ORIGINAL_SOURCE_FILES}
    ${CSTONEOCTREE_SOURCE_FILES}
)

include_directories(
    pugixml/src
    "${VTK_USE_FILE}"
    ${ORIGINAL_INCLUDE_DIR}
    ${CSTONEOCTREE_INCLUDE_DIR}
    )

add_library(pugixml
    STATIC
    pugixml/src/pugixml.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    ${ORIGINAL_INCLUDE_DIR}
    ${CSTONEOCTREE_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)


target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    CUDA::cudart
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
    HDF5::HDF5
    Eigen3::Eigen
    ${VTK_LIBRARIES}
    ${HDF5_C_STATIC_LIBRARY}
    pugixml
)

