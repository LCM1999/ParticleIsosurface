#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"



/*
CUDA_NODE_CALC_CONST_R:
    // initial evaluator:
        particles: always means xmeans in anisotropic interpolation
        radiuses: particles' radiuses
        Gs: transform matrix in anisotropic interpolation
        splashs
        particles_gradients
    // initial hash_grid
    // TNode lists
    // output:
        nodes: tnodes' feature point
        recurs: flags to determine should this tnode be subdivision
*/
__global__ cuda_node_calc_const_r(
    float* particles, float* radiuses, float* Gs, bool* splashs, float* particles_gradients, int* particels_num,
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    float* bounding, float* cell_size,
    int* hash_list_size, int* index_list_size, int* start_end_list_size,
    short* types, short* depths, float* centers, float* half_lengthes, int* tnodes_num,
    float* nodes,
    bool* recurs,
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}