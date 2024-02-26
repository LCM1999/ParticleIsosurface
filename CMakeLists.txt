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
if( UNIX )
    set(MPI_DIR /usr/lib/x86_64-linux-gnu/openmpi)
    set(HDF5_DIR /home/lcm/h5/CMake-hdf5-1.14.2/build/HDF5-1.14.2-Linux/HDF_Group/HDF5/1.14.2/cmake)
    set(EIGEN3_DIR /home/lcm/eigen/eigen-3.4.0/build)
endif()

find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found!")
endif()

set(HDF5_DIR E:/hdf5/HDF5-1.14.3-win64/cmake)
set(HDF5_PREFER_PARALLEL "ON")
find_package(HDF5 REQUIRED COMPONENTS C HL)
# find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS C static HL)
if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif ()

find_package(EIGEN3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()

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

add_executable(${PROJECT_NAME}
    src/iso_method_ours.cpp
    src/main.cpp
    src/marching.cpp
    src/timer.cpp
    src/utils.cpp
    src/visitorextract.cpp
    src/evaluator.cpp
    src/hash_grid.cpp
    src/surface_reconstructor.cpp
    src/global.cpp
    src/recorder.cpp
    src/hdf5Utils.cpp
    src/multi_level_reseacher.cpp
    src/rply.c
)

include_directories(
    pugixml/src
    "${VTK_USE_FILE}"
    )

add_library(pugixml
    STATIC
    pugixml/src/pugixml.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    include
    ${MPI_CXX_INCLUDE_DIRS}
    # ${HDF5_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# cuda_add_library(node_calc_lib SHARED src/node_calc.cu)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${HDF5_C_STATIC_LIBRARY}
    ${VTK_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    HDF5::HDF5
    pugixml
)
