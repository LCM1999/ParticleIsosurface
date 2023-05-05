#ifndef NODE_CALC_H
#define NODE_CALC_H

void cuda_node_calc_const_r_kernel(
	int blocks, int threads,
	float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values,
    char* types, char* depths, float* centers, float* half_lengthes, int* tnode_num,
    float* nodes
);

#endif