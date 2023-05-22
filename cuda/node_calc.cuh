#ifndef NODE_CALC_CUH
#define NODE_CALC_CUH

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "../include/evaluator.h"
#include "../include/hash_grid.h"

extern "C" void cuda_node_calc_const_r_kernel(
    int GlobalParticlesNum, int DepthMin, int QueueFlag,
    float R, float NeighborFactor, float IsoValue, float MinScalar, float MaxScalar,  
    Evaluator* evaluator, HashGrid* hashgrid);
#endif