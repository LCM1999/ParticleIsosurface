cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(Isosurface_smiplify CXX)

set(METIS_INCLUDE_DIR "F:/RepoCpp/Isosurface_smiplify_cmake/include")
set(METIS_LIB "F:/RepoCpp/Isosurface_smiplify_cmake/lib/metis.lib")
set(PROJECT_NAME Isosurface_smiplify)


################################################################################
# Find packages
################################################################################
find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found!")
endif()

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME}
    src/cube_arrays.cpp
    src/global.cpp
    src/intersect.cpp
    src/iso_method_ours.cpp
    src/main.cpp
    src/matrix.cpp
    src/mctable.cpp
    src/parse.cpp
    src/tet_arrays.cpp
    src/timer.cpp
    src/vect.cpp
    src/visitorextract.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    include
    ${METIS_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${METIS_LIB}
    OpenMP::OpenMP_CXX
)
