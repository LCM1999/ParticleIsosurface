cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME Isosurface_smiplify)
#SET(HDF5_USE_STATIC_LIBRARIES ON)

################################################################################
# Find packages
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)


find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

#find_package(MPI REQUIRED)
#if(NOT MPI_FOUND)
#    message(FATAL_ERROR "MPI not found!")
#endif()#

#set(HDF5_PREFER_PARALLEL "ON")
#find_package(HDF5 REQUIRED COMPONENTS C HL)
#if(NOT HDF5_FOUND)
#    message(FATAL_ERROR "HDF5 not found!")
#endif()

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


add_executable(${PROJECT_NAME}
    src/cube_arrays.cpp
    src/iso_method_ours.cpp
    src/main.cpp
    src/tet_arrays.cpp
    src/timer.cpp
    src/utils.cpp
    src/visitorextract.cpp
    src/evaluator.cpp
    src/hash_grid.cpp
    src/surface_reconstructor.cpp
    src/global.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    include
    #${METIS_INCLUDE_DIR}
    #${MPI_CXX_INCLUDE_DIRS}
    #${HDF5_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    #Metis::metis
    #${HDF5_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
