#ifndef NODE_CALC_CUH
#define NODE_CALC_CUH

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "../include/evaluator.h"
#include "../include/hash_grid.h"

class TNode;

__constant__ static float BOUNDING[6];
__constant__ static unsigned int XYZ_CELL_NUM[3];
__constant__ static size_t START_END_LIST_SIZE[2];
// __constant__ static int PARTICLES_NUM[1];
__constant__ static int MIN_DEPTH[1];
__constant__ static float RADIUS[3];
// __constant__ static float NEIGHBOR_FACTOR[1];
__constant__ static float INFLUNCE[3];
__constant__ static float ISO_VALUE[1];
__constant__ static float MAX_SCALAR[1];
__constant__ static float HASH_CELL_SIZE[1];
__constant__ static int TNODE_NUM[1];
__constant__ static float HALF_CELL_BORDER_STEP_SIZE[4];

float* particles_gpu;
float* Gs_gpu; 
bool* splashs_gpu; 
float* particles_gradients_gpu; 
int* index_list_gpu;
long long* start_list_keys_gpu;
int* start_list_values_gpu;
long long* end_list_keys_gpu;
int* end_list_values_gpu;
char* types_gpu; 
float* centers_gpu; 
float* nodes_gpu; 

extern "C" void cuda_node_calc_initialize_const_r(
    int GlobalParticlesNum, int DepthMin, float R, float InfFactor, float IsoValue, float MaxScalar,
    Evaluator* evaluator, HashGrid* hashgrid
);

extern "C" void cuda_node_calc_release_const_r();

extern "C" void cuda_node_calc_const_r_kernel(
    int QueueFlag, float half_length, char* types_cpu, float* centers_cpu, float* nodes_cpu
);
#endif