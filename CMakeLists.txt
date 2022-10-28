cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX)

set(METIS_INCLUDE_DIR "F:/RepoCpp/Isosurface_smiplify_cmake/include")
set(METIS_LIB "F:/RepoCpp/Isosurface_smiplify_cmake/lib/metis.lib")
set(HDF5_ROOT "F:/HDF5")
#set(HDF5_INCLUDE_DIR "F:/HDF5/include")
#set(HDF5_LIB "F:/HDF5/lib/hdf5.lib")

set(PROJECT_NAME Isosurface_smiplify)

################################################################################
# Find packages
################################################################################
find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(HDF5 REQUIRED COMPONENTS CXX)
if(NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif()

find_package(EIGEN3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()


################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME}
    src/cube_arrays.cpp
    src/global.cpp
    src/iso_method_ours.cpp
    src/main.cpp
    src/tet_arrays.cpp
    src/timer.cpp
    src/utils.cpp
    src/visitorextract.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    include
    ${METIS_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${METIS_LIB}
    ${HDF5_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
