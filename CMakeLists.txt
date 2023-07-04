cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(COMPUTE_CAPABILITY 75)


set(PROJECT_NAME Isosurface_smiplify)

################################################################################
# Find packages
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(EIGEN3_DIR C:/Lib/Eigen-3.4.0-MSVC/share/eigen3/cmake)
set(HDF5_DIR C:/Lib/HDF5-1.10.7-MSVC/cmake/hdf5)


# find_package(CUDA  REQUIRED)
# if(NOT CUDA_FOUND)
#     message(FATAL_ERROR "CUDA not found!")    
# endif(NOT CUDA_FOUND)

# find_package(Thrust REQUIRED)
# if(NOT Thrust_FOUND)
#     message(FATAL_ERROR "Thrust not found!")    
# endif(NOT Thrust_FOUND)


find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found!")
endif()

set(HDF5_PREFER_PARALLEL "ON")
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS C static HL)
if(NOT HDF5_FOUND)
   message(FATAL_ERROR "HDF5 not found!")
endif()

find_package(EIGEN3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()

#find_package(Metis REQUIRED)
#if(NOT Metis_FOUND)
#    message(FATAL_ERROR "metis not found!")
#endif()


################################################################################
# Target
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options("-O3")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options("-openmp:llvm")
endif()

# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_${COMPUTE_CAPABILITY},code=compute_${COMPUTE_CAPABILITY};-g;-G;)

add_executable(${PROJECT_NAME}
    src/iso_method_ours.cpp
    src/main.cpp
    src/timer.cpp
    src/utils.cpp
    src/visitorextract.cpp
    src/evaluator.cpp
    src/hash_grid.cpp
    src/surface_reconstructor.cpp
    src/global.cpp
    src/recorder.cpp
    src/pugixml.cpp
    src/hdf5Utils.cpp
    src/multi_level_reseacher.cpp
    # src/node_calc.cu
    src/rply.c
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    include
    #${METIS_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    # ${CUDA_INCLUDE_DIRS}
    # ${THRUST_INCLUDE_DIRS}
)

# cuda_add_library(node_calc_lib SHARED src/node_calc.cu)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    #Metis::metis
    ${HDF5_C_STATIC_LIBRARY}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    # node_calc_lib
    # ${CUDA_LIBRARIES}
    # ${THRUST_LIBRARIES}
)
