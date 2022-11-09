cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX)

set(METIS_INCLUDE_DIR "F:/RepoCpp/Isosurface_smiplify_cmake/include")
set(METIS_LIB "F:/RepoCpp/Isosurface_smiplify_cmake/lib/metis.lib")
#set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/1.10.7/")
#set(AEC_LIB "C:/Program Files/HDF_Group/HDF5/1.10.7/lib/libaec.lib")
#set(SZIP_LIB "C:/Program Files/HDF_Group/HDF5/1.10.7/lib/libsz.lib")
#set(ZLIB_LIB "C:/Program Files/HDF_Group/HDF5/1.10.7/lib/libzlib.lib")
#set(HDF5_INCLUDE_DIR "C:/Program Files/HDF_Group/HDF5/1.10.7/include")
#set(HDF5_LIB "C:/Program Files/HDF_Group/HDF5/1.10.7/hdf5.lib")

set(PROJECT_NAME Isosurface_smiplify)
#SET(HDF5_USE_STATIC_LIBRARIES ON)

################################################################################
# Find packages
################################################################################
find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

#find_package(MPI REQUIRED)
#if(NOT MPI_FOUND)
#    message(FATAL_ERROR "MPI not found!")
#endif()#

#find_package(HDF5 REQUIRED COMPONENTS CXX)
#if(NOT HDF5_FOUND)
#    message(FATAL_ERROR "HDF5 not found!")
#endif()

find_package(EIGEN3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()


################################################################################
# Target
################################################################################
add_compile_options("-openmp:llvm")

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
    ${METIS_INCLUDE_DIR}
    #${MPI_CXX_INCLUDE_DIRS}
    #${HDF5_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${METIS_LIB}
    #${AEC_LIB}
    #${SZIP_LIB}
    #${ZLIB_LIB}
    #${MPI_LIBRARIES}
    #${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
