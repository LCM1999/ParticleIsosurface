find_package(GTest)
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -Xcompiler -fopenmp")


# if(CMAKE_CUDA_COMPILER)
#     message("unit test use cuda!")
#     file(GLOB_RECURSE CSTONEOCTREE_SOURCE_FILES CONFIGURE_DEPENDS
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.cpp"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.c"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.cu")

#     file(GLOB_RECURSE CSTONEOCTREE_HEADER_FILES CONFIGURE_DEPENDS
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.h"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.hpp"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.cuh")
# else()
#     file(GLOB_RECURSE CSTONEOCTREE_SOURCE_FILES CONFIGURE_DEPENDS
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.cpp"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.c")

#     file(GLOB_RECURSE CSTONEOCTREE_HEADER_FILES CONFIGURE_DEPENDS
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.h"
#             "${PROJECT_SOURCE_DIR}/engine/cstoneOctree/*.hpp")
# endif()

# print cstoneoctree_source_files
message("cstoneoctree_source_files: ${CSTONEOCTREE_SOURCE_FILES}")

set(CSTONEOCTREE_TEST cstone_octree_test.cpp)
add_executable(cstone_octree_test 
                ${CSTONEOCTREE_TEST}
                ${ISO_SOURCE_FILES}
                ${CSTONEOCTREE_SOURCE_FILES}
                )

set_target_properties(cstone_octree_test PROPERTIES
    BUILD_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib/gcc/x86_64-linux-gnu/12"
)

target_include_directories(cstone_octree_test 
                            PRIVATE 
                            ${HDF5_INCLUDE_DIR}
                            ${ISO_INCLUDE_DIR}
                            ${CSTONEOCTREE_INCLUDE_DIR}
                            ${MPI_CXX_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR}
                            )

target_link_libraries(cstone_octree_test 
                    PUBLIC
                    ${HDF5_C_STATIC_LIBRARY}
                    ${VTK_LIBRARIES}
                    GTest::gtest_main
                    Eigen3::Eigen
                    OpenMP::OpenMP_CXX
                    HDF5::HDF5
                    pugixml
                    CUDA::cudart
                    ${CUDA_LIBRARIES}
                    )

add_test(NAME CSTONE COMMAND cstone_octree_test)


set(CSTONE_OCTREE_TEST_GPU cstone_octree_test_gpu.cu)


add_executable(cstone_octree_test_gpu
                    ${CSTONE_OCTREE_TEST_GPU}                            
                    ${ISO_SOURCE_FILES}
                    ${CSTONEOCTREE_SOURCE_FILES}
                    )

set_target_properties(cstone_octree_test_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib/gcc/x86_64-linux-gnu/12"
    )  

target_include_directories(cstone_octree_test_gpu 
                    PRIVATE 
                    ${HDF5_INCLUDE_DIR}
                    ${ISO_INCLUDE_DIR}
                    ${CSTONEOCTREE_INCLUDE_DIR}
                    ${CSTONEOCTREE_HEADER_FILES}
                    ${MPI_CXX_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    )


target_link_libraries(cstone_octree_test_gpu 
                PUBLIC
                ${HDF5_C_STATIC_LIBRARY}
                ${VTK_LIBRARIES}
                GTest::gtest_main
                Eigen3::Eigen
                OpenMP::OpenMP_CXX
                HDF5::HDF5
                pugixml
                CUDA::cudart
                ${CUDA_LIBRARIES}
                )

add_test(NAME CSTONE_GPU COMMAND cstone_octree_test_gpu)