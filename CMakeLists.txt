cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(METIS_INCLUDE_DIR "F:/RepoCpp/Isosurface_smiplify_cmake/include")
#set(METIS_LIB "F:/RepoCpp/Isosurface_smiplify_cmake/lib/metis.lib")


set(PROJECT_NAME Isosurface_smiplify)

################################################################################
# Find packages
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)


find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

set(HDF5_PREFER_PARALLEL "ON")
find_package(HDF5 REQUIRED COMPONENTS C HL)
if(NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif()

find_package(EIGEN3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found!")
endif()

find_package(Metis REQUIRED)
if(NOT Metis_FOUND)
    message(FATAL_ERROR "metis not found!")
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
    Metis::metis
    ${HDF5_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
