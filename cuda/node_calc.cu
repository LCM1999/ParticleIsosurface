#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math.h"

#include "node_calc.cuh"
#include "iso_method_ours.h"
#include "cuda_error.cuh"

#include "../include/timer.h"


#define TOLERANCE 1e-5f
#define FLATNESS 0.99f
#define INV_PI 0.31830988618379067153776752674503
#define SAMPLE 2
#define MIN_SCALAR 0
#define REGS_USED_BY_KERNEL 32

/*-----------------Device Math Util---------------------*/
__device__ float squared_norm(float* v, const int m)
{
    float sn = 0;
    for (size_t i = 0; i < m; i++)
    {
        sn += v[i] * v[i];
    }
    return sn;
}

__device__ float norm(float* v, const int m)
{
    return sqrt(squared_norm(v, m));
}

__device__ void normlize(float* v, const int m)
{
    float n = norm(v, m);
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

__device__ float dot(float* v1, float* v2, const int m) {
    float r = 0;
    for (size_t i = 0; i < m; i++)
    {
        r += v1[i] * v2[i];
    }
    return r;
}

__device__ float squared(float x)
{
    return x * x;
}

__device__ void mat_mult_vec(float* mat, float* vec, const int m, float* r)
{
    for (int i = 0; i < m; i++)
    {{
        for (int j = 0; j < m; j++)
        {{
            r[i] += mat[i * m + j] * vec[j]; 
        }}
    }}
}

__device__ float determine3(float* mat) 
{
     return  mat[0] * mat[4] * mat[8] -
            mat[0] * mat[5] * mat[7] +
            mat[1] * mat[5] * mat[6] -
            mat[1] * mat[3] * mat[8] +
            mat[2] * mat[3] * mat[7] -
            mat[2] * mat[4] * mat[6];
}

__device__ void qef_combine(float* qef_normal, float* eqn) {
    int index = 0;
    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = i; j < 5; j++)
        {
            qef_normal[index] += eqn[i] * eqn[j];
            index++;
        }
    }
}

__device__ void mat_inverse_gaussian(float* mat, float* inv, const int m) {
    for (size_t i = 0; i < m; i++)
    {
        inv[i * m + i] = 1.0f;
    }
    float pivot, factor;
    for (size_t i = 0; i < m; i++)
    {
        pivot = mat[i * m + i];
        for (size_t j = 0; j < m; j++)
        {
            mat[i * m + j] /= pivot;
            inv[i * m + j] /= pivot;
        }
        for (size_t j = 0; j < m; j++)
        {
            if (i == j)
            {
                continue;
            }
            factor = mat[j * m + i];
            for (size_t k = 0; k < m; k++)
            {
                mat[j * m + k] -= factor * mat[i * m + k];
                inv[j * m + k] -= factor * mat[i * m + k];
            }
        }
    }
}

/*-----------------Device Neighbor Searcher---------------------*/
__device__ long long start_end_find(long long* keys, const unsigned int size, const long long hash) {
    for (size_t i = 0; i < size; i++)
    {
        if (keys[i] == hash)
        {
            return i;
        }
    }
    return -1;
}

__device__ void calc_xyz_idx(
    float* pos, int* xyzIdx
) {
    xyzIdx[0] = 0;
    xyzIdx[1] = 0;
    xyzIdx[2] = 0;
    for (size_t i = 0; i < 3; i++)
    {
        xyzIdx[i] = int((pos[i] - BOUNDING[i * 2]) / HASH_CELL_SIZE[0]);
    }
}

__device__ long long calc_cell_hash(int* xyzIdx) {
    if (xyzIdx[0] < 0 || xyzIdx[0] >= XYZ_CELL_NUM[0] ||
		xyzIdx[1] < 0 || xyzIdx[1] >= XYZ_CELL_NUM[1] ||
		xyzIdx[2] < 0 || xyzIdx[2] >= XYZ_CELL_NUM[2])
		return -1;
	return (long long)xyzIdx[2] * (long long)XYZ_CELL_NUM[0] * (long long)XYZ_CELL_NUM[1] + 
		(long long)xyzIdx[1] * (long long)XYZ_CELL_NUM[0] + (long long)xyzIdx[0];
}

__device__ void get_in_cell_list(
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    const long long hash, int* p_list, unsigned short& p_size
) {
    long long start_index = start_end_find(start_list_keys, START_END_LIST_SIZE[0], hash);
    long long end_index = start_end_find(end_list_keys, START_END_LIST_SIZE[1], hash);
    if (start_index < 0 || end_index < 0)
    {
        return;
    }
    for (size_t countIdx = start_list_values[start_index]; countIdx < end_list_values[end_index]; countIdx++)
    {
        p_list[p_size++] = index_list[countIdx];
    }
}

/*-----------------Device Evaluator---------------------*/
__device__ bool check_splash(bool* splashs, const int pIdx)
{
    if (splashs[pIdx])
    {
        return true;    
    }
    return false;
}

__device__ float general_kernel(float d2, float h2, float h) {
    return (d2 >= h2) ? 0.0f : pow(h2 - d2, 3) * (315 / (64 * pow(h, 9))) * INV_PI;
}

__device__ void anisotropic_interpolate(
    float* G, float* diff, float& scalar
) {
    float gs_mult_diff[3] = {0};
    mat_mult_vec(G, diff, 3, gs_mult_diff);
    scalar += RADIUS[2] * determine3(G) * general_kernel(squared_norm(gs_mult_diff, 3), INFLUNCE[1], INFLUNCE[0]);
}

__device__ void single_eval_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients,
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    float* pos, float& v
) {
    float scalar = 0.0f;
    int neighbors[1024] = {0};
    unsigned short n_size = 0;
    int xyzIdx[3] = {0};
    int tempXyzIdx[3] = {0};
    float diff[3] = {0};
    long long hash = 0;
    calc_xyz_idx(pos, xyzIdx);
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                tempXyzIdx[0] = xyzIdx[0] + x;
                tempXyzIdx[1] = xyzIdx[1] + y;
                tempXyzIdx[2] = xyzIdx[2] + z;
                hash = calc_cell_hash(tempXyzIdx);
                if (hash < 0)
                {
                    continue;
                }
                get_in_cell_list(index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, hash, neighbors, n_size);
            }
        }
    }
    if (n_size > 0)
    {
        for (size_t n = 0; n < n_size; n++)
        {
            diff[0] = pos[0] - particles[neighbors[n] * 3 + 0];
            diff[1] = pos[1] - particles[neighbors[n] * 3 + 1];
            diff[2] = pos[2] - particles[neighbors[n] * 3 + 2];
            anisotropic_interpolate(Gs + neighbors[n] * 9, diff, scalar);
        }
    }
    v = ISO_VALUE[0] - ((scalar - MIN_SCALAR) / MAX_SCALAR[0] * 255);
}

__device__ void grid_eval_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    bool& signchange, float* sample_points, float* sample_grads
) {
    bool origin_sign = false;
    float p[3] = {0};
    int neighbors[1024] = {0};
    unsigned short n_size = 0;
    int xyzIdx[3] = {0};
    int tempXyzIdx[3] = {0};
    long long hash = 0;
    float diff[3] = {0};
    float scalar = 0;
    for (size_t i = 0; i < ((SAMPLE+1)*(SAMPLE+1)*(SAMPLE+1)); i++)
    {
        scalar = 0.0f;
        p[0] = sample_points[i * 4 + 0];
        p[1] = sample_points[i * 4 + 1];
        p[2] = sample_points[i * 4 + 2];
        calc_xyz_idx(p, xyzIdx);
        n_size = 0;
        for (tempXyzIdx[0] = xyzIdx[0]-1; tempXyzIdx[0] <= xyzIdx[0]+1; tempXyzIdx[0]++)
        {
            for (tempXyzIdx[1] = xyzIdx[1]-1; tempXyzIdx[1] <= xyzIdx[1]+1; tempXyzIdx[1]++)
            {
                for (tempXyzIdx[2] = xyzIdx[2]-1; tempXyzIdx[2] <= xyzIdx[2]+1; tempXyzIdx[2]++)
                {
                    hash = calc_cell_hash(tempXyzIdx);
                    if (hash < 0)
                    {
                        continue;
                    }
                    get_in_cell_list(index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, hash, neighbors, n_size);
                }
            }
        }
        if (n_size > 0)
        {
            for (size_t n = 0; n < n_size; n++)
            {
                diff[0] = p[0] - particles[neighbors[n] * 3 + 0];
                diff[1] = p[1] - particles[neighbors[n] * 3 + 1];
                diff[2] = p[2] - particles[neighbors[n] * 3 + 2];
                // scalar += 
                anisotropic_interpolate(Gs + neighbors[n] * 9, diff, scalar);
            }
        }
        scalar = ISO_VALUE[0] - ((scalar - MIN_SCALAR) / MAX_SCALAR[0] * 255);
        sample_points[i * 4 + 3] = scalar;
        origin_sign = (sample_points[3] >= 0);
        if (!signchange)
        {
            signchange = origin_sign ^ (scalar >= 0);
        }
    }
    int index, last_idx, next_idx;
    float gradient[3] = {0};
    for (int z = 0; z <= SAMPLE; z++)
    {
        for (int y = 0; y <= SAMPLE; y++)
        {
            for (int x = 0; x <= SAMPLE; x++)
            {
                index = (z * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + x);
                gradient[0] = 0;
                gradient[1] = 0;
                gradient[2] = 0;
                next_idx = (z * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + (x + 1));
                last_idx = (z * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + (x - 1));
                if (x == 0)
                {
                    gradient[0] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3])/ HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else if (x == SAMPLE)
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (HALF_CELL_BORDER_STEP_SIZE[3] * 2);
                }

                next_idx = (z * ((SAMPLE+1) * (SAMPLE+1)) + (y + 1) * (SAMPLE+1) + x);
                last_idx = (z * ((SAMPLE+1) * (SAMPLE+1)) + (y - 1) * (SAMPLE+1) + x);
                if (y == 0)
                {
                    gradient[1] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else if (y == SAMPLE)
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (HALF_CELL_BORDER_STEP_SIZE[3] * 2);
                }

                next_idx = ((z + 1) * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + x);
                last_idx = ((z - 1) * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + x);
                if (z == 0)
                {
                    gradient[2] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else if (z == SAMPLE)
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / HALF_CELL_BORDER_STEP_SIZE[3];
                }
                else
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (HALF_CELL_BORDER_STEP_SIZE[3] * 2);
                }
                sample_grads[index * 3 + 0] = gradient[0];
                sample_grads[index * 3 + 1] = gradient[1];
                sample_grads[index * 3 + 2] = gradient[2];
            }
        }
    }
}

/*-----------------Device Top Layer Func---------------------*/
__device__ void check_empty_and_calc_curv_const_r(
    float* particles, bool* splashs, float* particles_gradients, 
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    float* minV, float* maxV, 
    bool& empty, float& curv
) {
    float norms[3] = {0};
    int area = 0;
    int minXyzIdx[3] = {0};
    int maxXyzIdx[3] = {0};
    long long hash;
    int p_list[1024];
    unsigned short p_size = 0;
    int xyzIdx[3] = {0};
    bool all_splash = true;
    unsigned int all_in_box = 0;
    calc_xyz_idx(minV, minXyzIdx);
    calc_xyz_idx(maxV, maxXyzIdx);
    for (xyzIdx[0] = (minXyzIdx[0] - 1); xyzIdx[0] <= (maxXyzIdx[0] + 1); xyzIdx[0]++)
    {
        for (xyzIdx[1] = (minXyzIdx[1] - 1); xyzIdx[1] <= (maxXyzIdx[1] + 1); xyzIdx[1]++)
        {
            for (xyzIdx[2] = (minXyzIdx[2] - 1); xyzIdx[2] <= (maxXyzIdx[2] + 1); xyzIdx[2]++)
            {
                p_size = 0;
                hash = calc_cell_hash(xyzIdx);
                if (hash < 0)
                    continue;
                get_in_cell_list(index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, hash, p_list, p_size);
                if (p_size == 0)
                {
                    continue;
                }
                all_in_box += p_size;
                for (size_t i = 0; i < p_size; i++)
                {
                    if (!check_splash(splashs, p_list[i]))
                    {
                        if (particles[p_list[i] * 3 + 0] > (minV[0] - INFLUNCE[0]) && 
					        particles[p_list[i] * 3 + 0] < (maxV[0] + INFLUNCE[0]) &&
					        particles[p_list[i] * 3 + 1] > (minV[1] - INFLUNCE[0]) && 
					        particles[p_list[i] * 3 + 1] < (maxV[1] + INFLUNCE[0]) &&
					        particles[p_list[i] * 3 + 2] > (minV[2] - INFLUNCE[0]) && 
					        particles[p_list[i] * 3 + 2] < (maxV[2] + INFLUNCE[0]))
                        {
                            if (particles_gradients[p_list[i] * 3 + 0] == 0 && 
                                particles_gradients[p_list[i] * 3 + 1] == 0 && 
                                particles_gradients[p_list[i] * 3 + 2] == 0)
                            {
                                continue;
                            }
                            norms[0] += particles_gradients[p_list[i] * 3 + 0];
                            norms[1] += particles_gradients[p_list[i] * 3 + 1];
                            norms[2] += particles_gradients[p_list[i] * 3 + 2];
                            area++; 
                            all_splash = false;
                        }
                    }
                }
            }
        }
    }
    if (all_in_box == 0)
    {
        empty = true;
    } else {
        empty = all_splash;
    }
    curv = (area == 0) ? 1.0f : (norm(norms, 3) / area);
}

__device__ void generate_sampling_points(
    float* minV, float* maxV,
    float* sample_points
) {
    int node_index;
    for (size_t x = 0; x <= SAMPLE; x++)
    {
        for (size_t y = 0; y <= SAMPLE; y++)
        {
            for (size_t z = 0; z <= SAMPLE; z++)
            {
                node_index = z * (SAMPLE+1) * (SAMPLE+1) + y * (SAMPLE+1) + x;
                sample_points[node_index * 4 + 0] = 
				(1 - (float(x) / SAMPLE)) * minV[0] + (float(x) / SAMPLE) * maxV[0];
				sample_points[node_index * 4 + 1] = 
				(1 - (float(y) / SAMPLE)) * minV[1] + (float(y) / SAMPLE) * maxV[1];
				sample_points[node_index * 4 + 2] = 
				(1 - (float(z) / SAMPLE)) * minV[2] + (float(z) / SAMPLE) * maxV[2];
				sample_points[node_index * 4 + 3] = 0;
            }
        }
    }
}

__device__ void node_sampling_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    float& curv, bool& signchange, float* sample_points, float* sample_grads
) {
    signchange = false;
    grid_eval_const_r(
        particles, Gs, splashs, particles_gradients, 
        index_list, start_list_keys, start_list_values, end_list_keys, end_list_values,
        signchange, sample_points, sample_grads
    );
    float norms[3] = {0};
    float area = 0.0f;
    float tempNorm[3] = {0};
    float field_curv = 0.0f;
    for (size_t i = 0; i < ((SAMPLE+1)*(SAMPLE+1)*(SAMPLE+1)); i++)
    {
        tempNorm[0] = sample_grads[i * 3 + 0];
        tempNorm[1] = sample_grads[i * 3 + 1];
        tempNorm[2] = sample_grads[i * 3 + 2];
        norms[0] += tempNorm[0];
        norms[1] += tempNorm[1];
        norms[2] += tempNorm[2];
        area += norm(tempNorm, 3);
    }
    field_curv = (area == 0) ? 1.0f : (norm(norms, 3) / area);
    curv = (field_curv > curv) ? curv : field_curv;
}

__device__ void calc_error(float* p, float* points, float* grads, float& err)
{
    err = 0;
    float tempv[3] = {0};
	for (size_t i = 0; i < 64; i++)
	{
        tempv[0] = p[0] - points[i * 4 + 0];
        tempv[1] = p[1] - points[i * 4 + 1];
        tempv[2] = p[2] - points[i * 4 + 2];
		err += squared(p[3] - dot(grads + i * 3, tempv, 3)) / (1 + squared_norm(grads + i * 3, 3));
	}
}

__device__ void feature_calc(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    float* minV, float* maxV, float* center,
    float* sample_points, float* sample_grads, 
    float* node)
{
    float borderMinV[3] = {
        center[0] - HALF_CELL_BORDER_STEP_SIZE[0] + HALF_CELL_BORDER_STEP_SIZE[2], 
        center[1] - HALF_CELL_BORDER_STEP_SIZE[0] + HALF_CELL_BORDER_STEP_SIZE[2], 
        center[2] - HALF_CELL_BORDER_STEP_SIZE[0] + HALF_CELL_BORDER_STEP_SIZE[2]};
    float borderMaxV[3] = {
        center[0] + HALF_CELL_BORDER_STEP_SIZE[0] - HALF_CELL_BORDER_STEP_SIZE[2], 
        center[1] + HALF_CELL_BORDER_STEP_SIZE[0] - HALF_CELL_BORDER_STEP_SIZE[2], 
        center[2] + HALF_CELL_BORDER_STEP_SIZE[0] - HALF_CELL_BORDER_STEP_SIZE[2]};
    float qef_normal[15] = {0};
    float p[3] = {0};
    float pl[5] = {0};
    int node_index, index;
    for (size_t x = 0; x <= SAMPLE; x++)
    {
        for (size_t y = 0; y <= SAMPLE; y++)
        {
            for (size_t z = 0; z <= SAMPLE; z++)
            {
                node_index = z * ((SAMPLE+1) * (SAMPLE+1)) + y * (SAMPLE+1) + x;
                p[0] = sample_points[node_index * 4 + 0];
                p[1] = sample_points[node_index * 4 + 1];
                p[2] = sample_points[node_index * 4 + 2];
                pl[0] = sample_grads[node_index * 3 + 0];
				pl[1] = sample_grads[node_index * 3 + 1];
				pl[2] = sample_grads[node_index * 3 + 2];
				pl[3] = -1;
				pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + sample_points[node_index * 4 + 3];
                qef_combine(qef_normal, pl);
            }
        }
    }
    float A[16] = {0};
    float B[4] = {0};
    float* corners[2] = {borderMinV, borderMaxV};
    float AC[49] = {0};
    float BC[7] = {0};
    size_t i, j;
    int dir, side, dp, dpp;
    for (i = 0; i < 4; i++)
    {
        index = ((11 - i) * i) / 2;
        for (j = i; j < 4; j++)
        {
            A[i * 4 + j] = qef_normal[index + j - i];
            A[j * 4 + i] = A[i * 4 + j];
        }
        B[i] = -qef_normal[index + 4 - i];
    }
    bool is_out = true;
    float err = 1e30f, e;
    float pc[4] = {0};
    // float pcg[3] = {0};
    float rvalue[7] = {0};
    float inv[49] = {0};
    for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
    {
        if (cell_dim == 3)
        {
            mat_inverse_gaussian(A, inv, 4);
            for (i = 0; i < 4; i++)
            {
                rvalue[i] = 0;
                for (j = 0; j < 4; j++)
                {
                    rvalue[i] += inv[j * 4 + i] * B[j];
                }
            }
            pc[0] = rvalue[0];
            pc[1] = rvalue[1];
            pc[2] = rvalue[2];
            pc[3] = rvalue[3];
            if (pc[0] >= borderMinV[0] && pc[0] <= borderMaxV[0] &&
				pc[1] >= borderMinV[1] && pc[1] <= borderMaxV[1] &&
				pc[2] >= borderMinV[2] && pc[2] <= borderMaxV[2])
            {
                is_out = false;
				calc_error(pc, sample_points, sample_grads, err);
				node[0] = pc[0];
				node[1] = pc[1];
				node[2] = pc[2];
                node[3] = pc[3];
            }
        } else if (cell_dim == 2) {
			for (int face = 0; face < 6; face++)
            {
                dir = face / 2;
				side = face % 2;
                for (i = 0; i < 5; i++)
                {
                    for (j = 0; j < 5; j++)
                    {
                        AC[i * 5 + j] = (i < 4 && j < 4) ? A[i * 5 + j] : 0;
                    }
                    BC[i] = (i < 4) ? B[i] : 0;
                }
                AC[20 + dir] = AC[dir * 5 + 4] = 1;
                BC[4] = corners[side][dir];
                
                mat_inverse_gaussian(AC, inv, 5);
                for (i = 0; i < 5; i++)
                {
                    rvalue[i] = 0;
                    for (j = 0; j < 5; j++)
                    {
                        rvalue[i] += inv[j * 5 + i] * BC[j];
                    }
                }
                pc[0] = rvalue[0];
                pc[1] = rvalue[1];
                pc[2] = rvalue[2];
                pc[3] = rvalue[3];

                dp = (dir + 1) % 3;
                dpp = (dir + 2) % 3;
                if (pc[dp] >= borderMinV[dp] && pc[dp] <= borderMaxV[dp] &&
					pc[dpp] >= borderMinV[dpp] && pc[dpp] <= borderMaxV[dpp])
                {
                    is_out = false;
					calc_error(pc, sample_points, sample_grads, e);
					if (e < err)
					{
						err = e;
						node[0] = pc[0];
                        node[1] = pc[1];
                        node[2] = pc[2];
                        node[3] = pc[3];
					}
                }
            }
        } else if (cell_dim == 1) {
            for (int edge = 0; edge < 12; edge++) {
                dir = edge / 4;
				side = edge % 4;
                for (i = 0; i < 6; i++)
				{
					for (j = 0; j < 6; j++)
					{
						AC[i * 6 + j] = (i < 4 && j < 4 ? A[i * 4 + j] : 0);
					}
					BC[i] = (i < 4 ? B[i] : 0);
				}
                dp = (dir + 1) % 3;
				dpp = (dir + 2) % 3;
                AC[24 + dp] = AC[dp * 6 + 4] = 1;
				AC[30 + dpp] = AC[dpp * 6 + 5] = 1;
				BC[4] = corners[side & 1][dp];
				BC[5] = corners[side >> 1][dpp];

                mat_inverse_gaussian(AC, inv, 6);
                for (i = 0; i < 6; i++)
                {
                    rvalue[i] = 0;
                    for (j = 0; j < 6; j++)
                    {
                        rvalue[i] += inv[j * 6 + i] * BC[j];
                    }
                }
                pc[0] = rvalue[0];
                pc[1] = rvalue[1];
                pc[2] = rvalue[2];
                pc[3] = rvalue[3];
				if (pc[dir] >= borderMinV[dir] && pc[dir] <= borderMaxV[dir]) {
                    is_out = false;
					calc_error(pc, sample_points, sample_grads, e);
                    if (e < err)
					{
						err = e;
						node[0] = pc[0];
                        node[1] = pc[1];
                        node[2] = pc[2];
                        node[3] = pc[3];
					}
                }
            }
        } else if (cell_dim == 0) {
			for (int vertex = 0; vertex < 8; vertex++)
            {
                for (i = 0; i < 7; i++)
				{
					for (j = 0; j < 7; j++)
					{
						AC[i * 7 + j] = (i < 4 && j < 4 ? A[i * 4 + j] : 0);
					}
					BC[i] = (i < 4 ? B[i] : 0);
				}
                for (i = 0; i < 3; i++)
                {
                    AC[(4 + i) * 4 + i] = AC[i * 4 + 4 + i] = 1;
                    BC[4 + i] = corners[(vertex >> i) & 1][i];
                }

                mat_inverse_gaussian(AC, inv, 7);
                for (i = 0; i < 7; i++)
                {
                    rvalue[i] = 0;
                    for (j = 0; j < 7; j++)
                    {
                        rvalue[i] += inv[j * 7 + i] * BC[j];
                    }
                }
                pc[0] = rvalue[0];
                pc[1] = rvalue[1];
                pc[2] = rvalue[2];
                pc[3] = rvalue[3];
                calc_error(pc, sample_points, sample_grads, e);
                if (e < err)
				{
					err = e;
					node[0] = pc[0];
                    node[1] = pc[1];
                    node[2] = pc[2];
                    node[3] = pc[3];
				}
            }
        }
    }
    single_eval_const_r(
        particles, Gs, splashs, particles_gradients,
        index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
        node, node[3]
    );
}

/*
CUDA_NODE_CALC_CONST_R:
    initial evaluator:
        particles: always means xmeans in anisotropic interpolation
        Gs: transform matrix in anisotropic interpolation
        splashs
        particles_gradients
    initial hash_grid
    TNodes:
        types: nodes' type
        depths: nodes' depth
        centers: nodes' center
        half_length: nodes' half length
        tnodes_num: the number of tnodes
        default_oversample: the default oversample
    output:
        types
        nodes: tnodes' feature point
*/
__global__ void 
cuda_node_calc_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values,
    char* types, float* centers, float* nodes
) {
    int blockId = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
    int index = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index < TNODE_NUM[0])
    {
        bool empty = false;
        bool signchange = false;
        float sample_points[(SAMPLE+1) * (SAMPLE+1) * (SAMPLE+1) * 4] = {0};
        float sample_grads[(SAMPLE+1) * (SAMPLE+1) * (SAMPLE+1) * 3] = {0};
        float curv = 0.0f;
        float minV[3] = {centers[index * 3 + 0] - HALF_CELL_BORDER_STEP_SIZE[0], centers[index * 3 + 1] - HALF_CELL_BORDER_STEP_SIZE[0], centers[index * 3 + 2] - HALF_CELL_BORDER_STEP_SIZE[0]};
        float maxV[3] = {centers[index * 3 + 0] + HALF_CELL_BORDER_STEP_SIZE[0], centers[index * 3 + 1] + HALF_CELL_BORDER_STEP_SIZE[0], centers[index * 3 + 2] + HALF_CELL_BORDER_STEP_SIZE[0]};
        check_empty_and_calc_curv_const_r(
            particles, splashs, particles_gradients, 
            index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            minV, maxV, 
            empty, curv
        );
        if (empty)
        {
            types[index] = 0;
            single_eval_const_r(
                particles, Gs, splashs, particles_gradients, 
                index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
                centers + index * 3, nodes[index * 4 + 3]
            );
            return;
        }
        generate_sampling_points(minV, maxV, sample_points);
        node_sampling_const_r(
            particles, Gs, splashs, particles_gradients, 
            index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            curv, signchange, sample_points, sample_grads
        );
        if ((HALF_CELL_BORDER_STEP_SIZE[1] - RADIUS[0]) < TOLERANCE)
        {
            types[index] = 2;
            // feature_calc(
            //     particles, Gs, splashs, particles_gradients, 
            //     index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            //     minV, maxV, centers + index * 3, 
            //     sample_points, sample_grads, 
            //     nodes + index * 4);
            single_eval_const_r(
                particles, Gs, splashs, particles_gradients,
                index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
                centers + index * 3, nodes[index * 4 + 3]
            );
            return;
        }
        if (signchange && (curv < FLATNESS))
        {
            types[index] = 1;
            return;
        } else {
            types[index] = 2;
            // feature_calc(
            //     particles, Gs, splashs, particles_gradients,
            //     index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            //     minV, maxV, centers + index * 3, 
            //     sample_points, sample_grads, 
            //     nodes + index * 4);
            single_eval_const_r(
                particles, Gs, splashs, particles_gradients,
                index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
                centers + index * 3, nodes[index * 4 + 3]
            );
            return;
        }
    }
}

void cuda_node_calc_initialize_const_r(
    int GlobalParticlesNum, int DepthMin, float R, float InfFactor, float IsoValue, float MaxScalar,
    Evaluator* evaluator, HashGrid* hashgrid
) {
    cudaSetDevice(0);
    size_t start_end_list_size[2] = {hashgrid->StartList.size(), hashgrid->EndList.size()};
    float hash_cell_size[1] = {hashgrid->CellSize};
    // int particles_num[1] = {GlobalParticlesNum};
    int min_depth[1] = {DepthMin};
    float radius[3] = {R, R*R, R*R*R};
    // int neighbor_factor[1] = {NeighborFactor};
    float influnce[3] = {R * InfFactor, pow(R*InfFactor, 2), pow(R*InfFactor,3)};
    float iso_value[1] = {IsoValue};
    float max_scalar[1] = {MaxScalar};
    float* xmeans_cpu = new float[GlobalParticlesNum * 3];
	float* Gs_cpu = new float[GlobalParticlesNum * 9];
	float* particles_grads_cpu = new float[GlobalParticlesNum * 3];
	for (size_t i = 0; i < GlobalParticlesNum; i++)
	{
		xmeans_cpu[i * 3 + 0] = evaluator->GlobalxMeans[i][0];
		xmeans_cpu[i * 3 + 1] = evaluator->GlobalxMeans[i][1];
		xmeans_cpu[i * 3 + 2] = evaluator->GlobalxMeans[i][2];
		Gs_cpu[i * 9 + 0] = evaluator->GlobalGs[i].data()[0];
		Gs_cpu[i * 9 + 1] = evaluator->GlobalGs[i].data()[1];
		Gs_cpu[i * 9 + 2] = evaluator->GlobalGs[i].data()[2];
		Gs_cpu[i * 9 + 3] = evaluator->GlobalGs[i].data()[3];
		Gs_cpu[i * 9 + 4] = evaluator->GlobalGs[i].data()[4];
		Gs_cpu[i * 9 + 5] = evaluator->GlobalGs[i].data()[5];
		Gs_cpu[i * 9 + 6] = evaluator->GlobalGs[i].data()[6];
		Gs_cpu[i * 9 + 7] = evaluator->GlobalGs[i].data()[7];
		Gs_cpu[i * 9 + 8] = evaluator->GlobalGs[i].data()[8];
		particles_grads_cpu[i * 3 + 0] = evaluator->PariclesNormals[i][0];
		particles_grads_cpu[i * 3 + 1] = evaluator->PariclesNormals[i][1];
		particles_grads_cpu[i * 3 + 2] = evaluator->PariclesNormals[i][2];
	}
    long long* start_list_keys_cpu = new long long[hashgrid->StartList.size()];
	int* start_list_values_cpu = new int[hashgrid->StartList.size()];
	long long* end_list_keys_cpu = new long long[hashgrid->EndList.size()];
	int* end_list_values_cpu = new int[hashgrid->EndList.size()];
    int i = 0;
    for (auto it: hashgrid->StartList) {
        start_list_keys_cpu[i] = it.first;
        start_list_values_cpu[i] = it.second;
        i++;
    }
    i = 0;
    for (auto it: hashgrid->EndList) {
        end_list_keys_cpu[i] = it.first;
        end_list_values_cpu[i] = it.second;
        i++;
    }

    cudaMemcpyToSymbol(BOUNDING, hashgrid->Bounding, 6 * sizeof(float));
	cudaMemcpyToSymbol(XYZ_CELL_NUM, hashgrid->XYZCellNum, 3 * sizeof(unsigned int));
	cudaMemcpyToSymbol(START_END_LIST_SIZE, start_end_list_size, 2 * sizeof(size_t));
	// cudaMemcpyToSymbol(PARTICLES_NUM, particles_num, 1 * sizeof(int));
	cudaMemcpyToSymbol(MIN_DEPTH, min_depth, 1 * sizeof(int));

	cudaMemcpyToSymbol(RADIUS, radius, 3 * sizeof(float));
	// cudaMemcpyToSymbol(NEIGHBOR_FACTOR, neighbor_factor, 1 * sizeof(float));
	cudaMemcpyToSymbol(INFLUNCE, influnce, 3 * sizeof(float));
	cudaMemcpyToSymbol(ISO_VALUE, iso_value, 1 * sizeof(float));
	cudaMemcpyToSymbol(MAX_SCALAR, max_scalar, 1 * sizeof(float));
	cudaMemcpyToSymbol(HASH_CELL_SIZE, hash_cell_size, 1 * sizeof(float));

    cudaMalloc((void**)& particles_gpu, GlobalParticlesNum * 3 * sizeof(float));
	cudaMalloc((void**)& Gs_gpu, GlobalParticlesNum * 9 * sizeof(float));
	cudaMalloc((void**)& splashs_gpu, GlobalParticlesNum * sizeof(bool));
	cudaMalloc((void**)& particles_gradients_gpu, GlobalParticlesNum * 3 * sizeof(float));
	cudaMalloc((void**)& index_list_gpu, hashgrid->IndexList.size() * sizeof(int));
	cudaMalloc((void**)& start_list_keys_gpu, hashgrid->StartList.size() * sizeof(long long));
	cudaMalloc((void**)& start_list_values_gpu, hashgrid->StartList.size() * sizeof(int));
	cudaMalloc((void**)& end_list_keys_gpu, hashgrid->StartList.size() * sizeof(long long));
	cudaMalloc((void**)& end_list_values_gpu, hashgrid->StartList.size() * sizeof(int));

	cudaMemcpy(particles_gpu, xmeans_cpu, GlobalParticlesNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Gs_gpu, Gs_cpu, GlobalParticlesNum * 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(splashs_gpu, evaluator->GlobalSplash.data(), GlobalParticlesNum * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(particles_gradients_gpu, particles_grads_cpu, GlobalParticlesNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(index_list_gpu, hashgrid->IndexList.data(), hashgrid->IndexList.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(start_list_keys_gpu, start_list_keys_cpu, hashgrid->StartList.size() * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(start_list_values_gpu, start_list_values_cpu, hashgrid->StartList.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(end_list_keys_gpu, end_list_keys_cpu, hashgrid->EndList.size() * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(end_list_values_gpu, end_list_values_cpu, hashgrid->EndList.size() * sizeof(int), cudaMemcpyHostToDevice);

    delete[] xmeans_cpu;
	delete[] Gs_cpu;
	delete[] particles_grads_cpu;
	delete[] start_list_keys_cpu;
	delete[] start_list_values_cpu;
	delete[] end_list_keys_cpu;
	delete[] end_list_values_cpu;
}

extern "C" void cuda_node_calc_release_const_r() {
    cudaFree(particles_gpu);
	cudaFree(Gs_gpu);
	cudaFree(splashs_gpu);
	cudaFree(particles_gradients_gpu);
	cudaFree(index_list_gpu);
	cudaFree(start_list_keys_gpu);
	cudaFree(start_list_values_gpu);
	cudaFree(end_list_keys_gpu);
	cudaFree(end_list_values_gpu);
}

void cuda_node_calc_const_r_kernel(
    int QueueFlag, float half_length, char* types_cpu, float* centers_cpu, float* nodes_cpu
) {
    double in_cuda_time, phase1, phase2;
    // int threadsPerBlock = 1024, blocksPerGrid;
    int threadsPerBlock_x = 1, threadsPerBlock_y = 1, threadsPerBlock_z = 1, blocksPerGrid = 65536 / REGS_USED_BY_KERNEL;
    if (QueueFlag <= 1024 * blocksPerGrid) {
        threadsPerBlock_x = 1024;
    } else if (QueueFlag <= 1024 * 64 * blocksPerGrid) {
        threadsPerBlock_x = 1024;
        threadsPerBlock_z = 64;
    } else if (QueueFlag <= 1024 * 1024 * blocksPerGrid) {
        threadsPerBlock_x = 1024;
        threadsPerBlock_y = 1024;
    } else {
        threadsPerBlock_x = 1024;
        threadsPerBlock_y = 1024;
        threadsPerBlock_z = 64;
    }
    blocksPerGrid = (QueueFlag + (threadsPerBlock_x * threadsPerBlock_y * threadsPerBlock_z - 1)) / (threadsPerBlock_x * threadsPerBlock_y * threadsPerBlock_z);
    // blocksPerGrid = (QueueFlag + threadsPerBlock_x - 1) / threadsPerBlock_x;
	// memory on host
    int tnode_num_cpu[1] = {QueueFlag};
    float half_cell_border_step[4] = {half_length, half_length * 2, half_length / 8, half_length * 2 / SAMPLE};
    // memory on device
    in_cuda_time = get_time();
	cudaMemcpyToSymbol(TNODE_NUM, tnode_num_cpu, sizeof(int));
	cudaMemcpyToSymbol(HALF_CELL_BORDER_STEP_SIZE, half_cell_border_step, 4 * sizeof(float));

	cudaMalloc((void**)& types_gpu, QueueFlag * sizeof(char));
	cudaMalloc((void**)& centers_gpu, QueueFlag * 3 * sizeof(float));
	cudaMalloc((void**)& nodes_gpu, QueueFlag * 4 * sizeof(float));

	cudaMemcpy(types_gpu, types_cpu, QueueFlag * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(centers_gpu, centers_cpu, QueueFlag * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(nodes_gpu, nodes_cpu, QueueFlag * 4 * sizeof(float), cudaMemcpyHostToDevice);
    phase1 = get_time();
    cuda_node_calc_const_r<<<dim3(threadsPerBlock_x, threadsPerBlock_y, threadsPerBlock_z), blocksPerGrid>>>(
        particles_gpu, Gs_gpu, splashs_gpu, particles_gradients_gpu, 
        index_list_gpu, start_list_keys_gpu, start_list_values_gpu, end_list_keys_gpu, end_list_values_gpu,
        types_gpu, centers_gpu, nodes_gpu
    );
    cudaMemcpy(types_cpu, types_gpu, QueueFlag * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(nodes_cpu, nodes_gpu, QueueFlag * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    phase2 = get_time();
    cudaFree(types_gpu);
    cudaFree(centers_gpu);
    cudaFree(nodes_gpu);
    printf("In Cuda Phase 1 time = %f, Phase 2 time = %f, Phase 3 time = %f;\n", phase1 - in_cuda_time, phase2 - phase1, get_time() - phase2);
}