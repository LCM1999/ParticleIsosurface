#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "math.h"

#define TOLERANCE 1e-5f
#define FLATNESS 0.99f
#define INV_PI 0.31830988618379067153776752674503


__constant__ float BOUNDING[6];
__constant__ unsigned int XYZ_CELL_NUM[3];
__constant__ size_t START_END_LIST_SIZE[2];
__constant__ int PARTICLES_NUM_MIN_DEPTH[2];
__constant__ float RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[6];


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

__device__ void mat_inverse(float* mat, float* inv, const int m) {
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
    const float CELL_SIZE, 
    float* pos, int* xyzIdx
) {
    xyzIdx[0] = 0;
    xyzIdx[1] = 0;
    xyzIdx[2] = 0;
    for (size_t i = 0; i < 3; i++)
    {
        xyzIdx[i] = int((pos[i] - BOUNDING[i * 2]) / CELL_SIZE);
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
    const long long hash, int* p_list, unsigned char& p_size
) {
    size_t countIdx;
    long long startIdx = start_end_find(start_list_keys, START_END_LIST_SIZE[0], hash), endIdx = start_end_find(end_list_keys, START_END_LIST_SIZE[1], hash);
    if (startIdx == -1 || endIdx == -1)
    {
        return;
    }
    for (countIdx = startIdx; countIdx < endIdx; countIdx++)
    {
        p_list[p_size++] = countIdx;
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
    float p_dist = (d2 >= h2) ? 0.0f : pow(h2 - d2, 3);
    float sigma = (315 / (64 * pow(h, 9))) * INV_PI;
    return p_dist * sigma;
}

__device__ float anisotropic_interpolate(
    float* G, const float radius, const float INF_FACTOR,
    float* diff
) {
    float influnce = radius * INF_FACTOR;
    float gs_mult_diff[3] = {0};
    mat_mult_vec(G, diff, 3, gs_mult_diff);
    float k_value = general_kernel(squared_norm(gs_mult_diff, 3), squared(influnce), influnce);
    return pow(radius, 3) * determine3(G) * k_value;
}

__device__ void grid_eval_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, const int PARTICLES_NUM, const float RADIUS, const float INF_FACTOR, const float ISO_VALUE, const float MIN_SCALAR, const float MAX_SCALAR,
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    const float CELL_SIZE, 
    bool& signchange, float* sample_points, float* sample_grads
) {
    bool origin_sign = false;
    float p[4] = {0};
    float scalar = 0.0f;
    int neighbors[512] = {0};
    unsigned char n_size = 0;
    int xyzIdx[3] = {0};
    int tempXyzIdx[3] = {0};
    long long hash = 0;
    float diff[3] = {0};
    for (size_t i = 0; i < 64; i++)
    {
        p[0] = sample_points[i * 4 + 0];
        p[1] = sample_points[i * 4 + 1];
        p[2] = sample_points[i * 4 + 2];
        p[3] = sample_points[i * 4 + 3];
        calc_xyz_idx(CELL_SIZE, p, xyzIdx);
        n_size = 0;
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
        if (n_size != 0)
        {
            for (size_t n = 0; n < n_size; n++)
            {
                diff[0] = p[0] - particles[neighbors[n] * 3 + 0];
                diff[1] = p[1] - particles[neighbors[n] * 3 + 1];
                diff[0] = p[0] - particles[neighbors[n] * 3 + 0];
                scalar += anisotropic_interpolate(Gs + neighbors[n] * 9, RADIUS, INF_FACTOR, diff);
            }
        }
        scalar = ISO_VALUE - ((scalar - MIN_SCALAR) / MAX_SCALAR * 255);
        sample_points[i * 4 + 3] = scalar;
        origin_sign = (sample_points[3] >= 0);
        if (!signchange)
        {
            signchange = origin_sign ^ (scalar >= 0);
        }
    }
    int index, last_idx, next_idx;
    float gradient[3] = {0};
    for (int z = 0; z < 4; z++)
    {
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                index = (z * 16 + y * 4 + x);
                gradient[0] = 0;
                gradient[1] = 0;
                gradient[2] = 0;
                next_idx = (z * 16 + y * 4 + (x + 1));
                last_idx = (z * 16 + y * 4 + (x - 1));
                if (x == 0)
                {
                    gradient[0] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (x == 3)
                {
                    gradient[0] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[0] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
                }

                next_idx = (z * 16 + (y + 1) * 4 + x);
                last_idx = (z * 16 + (y - 1) * 4 + x);
                if (y == 0)
                {
                    gradient[1] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (y == 3)
                {
                    gradient[1] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[1] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
                }

                next_idx = ((z + 1) * 16 + y * 4 + x);
                last_idx = ((z - 1) * 16 + y * 4 + x);
                if (z == 0)
                {
                    gradient[2] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (z == 3)
                {
                    gradient[2] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[2] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
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
    float* particles, bool* splashs, float* particles_gradients, const float RADIUS, const float INFLUNCE,
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    const float CELL_SIZE,
    float* minV, float* maxV, 
    bool& empty, float& curv
) {
    float norms[3] = {0.0f};
    float area = 0.0f;
    int minXyzIdx[3] = {0};
    int maxXyzIdx[3] = {0};
    long long hash;
    int p_list[512];
    unsigned char p_size = 0;
    int xyzIdx[3];
    bool all_splash = true;
    unsigned int all_in_box = 0;
    calc_xyz_idx(CELL_SIZE, minV, minXyzIdx);
    calc_xyz_idx(CELL_SIZE, maxV, maxXyzIdx);
    for (int x = (minXyzIdx[0] - 1); x < (maxXyzIdx[0] + 1); x++)
    {
        for (int y = (minXyzIdx[1] - 1); y < (maxXyzIdx[1] + 1); y++)
        {
            for (int z = (minXyzIdx[2] - 1); z < (maxXyzIdx[2] + 1); z++)
            {
                xyzIdx[0] = x;
                xyzIdx[1] = y;
                xyzIdx[2] = z;
                hash = calc_cell_hash(xyzIdx);
                if (hash < 0)
                    continue;
                get_in_cell_list(index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, hash, p_list, p_size);
                all_in_box += p_size;
                if (p_size == 0)
                {
                    continue;
                }
                for (size_t i = 0; i < p_size; i++)
                {
                    if (!check_splash(splashs, p_list[i]))
                    {
                        if (particles[p_list[i] * 3 + 0] > (minV[0] - INFLUNCE) && 
					        particles[p_list[i] * 3 + 0] < (maxV[0] + INFLUNCE) &&
					        particles[p_list[i] * 3 + 1] > (minV[1] - INFLUNCE) && 
					        particles[p_list[i] * 3 + 1] < (maxV[1] + INFLUNCE) &&
					        particles[p_list[i] * 3 + 2] > (minV[2] - INFLUNCE) && 
					        particles[p_list[i] * 3 + 2] < (maxV[2] + INFLUNCE))
                        {
                            float tempNorm[3] = {
                                particles_gradients[p_list[i] * 3 + 0], 
                                particles_gradients[p_list[i] * 3 + 1], 
                                particles_gradients[p_list[i] * 3 + 2]
                            };
                            if (tempNorm[0] == 0 && tempNorm[1] == 0 && tempNorm[2] == 0)
                            {
                                continue;
                            }
                            norms[0] += tempNorm[0];
                            norms[1] += tempNorm[1];
                            norms[2] += tempNorm[2];
                            area += norm(tempNorm, 3);
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
    curv = (area == 0) ? 0.0f : (norm(norms, 3) / area);
}

__device__ void generate_sampling_points(
    float* minV, float* maxV,
    float* sample_points
) {
    int node_index;
    for (size_t x = 0; x < 4; x++)
    {
        for (size_t y = 0; y < 4; y++)
        {
            for (size_t z = 0; z < 4; z++)
            {
                node_index = z * 16 + y * 4 + x;
                sample_points[node_index * 4 + 0] = 
				(1 - x / 3.0f) * minV[0] + (x / 3.0f) * maxV[0];
				sample_points[node_index * 4 + 1] = 
				(1 - y / 3.0f) * minV[1] + (y / 3.0f) * maxV[1];
				sample_points[node_index * 4 + 2] = 
				(1 - z / 3.0f) * minV[2] + (z / 3.0f) * maxV[2];
				sample_points[node_index * 4 + 3] = 0;
            }
        }
    }
}

__device__ void node_sampling_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, const int PARTICLES_NUM, const float RADIUS, const float INF_FACTOR, const float ISO_VALUE, const float MIN_SCALAR, const float MAX_SCALAR,
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values, 
    const float CELL_SIZE, 
    float& curv, bool& signchange, float* sample_points, float* sample_grads
) {
    signchange = false;
    grid_eval_const_r(
        particles, Gs, splashs, particles_gradients, PARTICLES_NUM, RADIUS, INF_FACTOR, ISO_VALUE, MIN_SCALAR, MAX_SCALAR,
        hash_list, index_list, start_list_keys, start_list_values, end_list_keys, end_list_values,
        CELL_SIZE, 
        signchange, sample_points, sample_grads
    );
    float norms[3] = {0};
    float area = 0.0f;
    float tempNorm[3] = {0};
    float field_curv = 0.0f;
    for (size_t i = 0; i < 64; i++)
    {
        tempNorm[0] = sample_grads[i * 3 + 0];
        tempNorm[1] = sample_grads[i * 3 + 1];
        tempNorm[2] = sample_grads[i * 3 + 2];
        // normlize(tempNorm, 3);
        norms[0] += tempNorm[0];
        norms[1] += tempNorm[1];
        norms[2] += tempNorm[2];
        area += norm(tempNorm, 3);
    }
    field_curv = (area == 0) ? 0.0 : (norm(norms, 3) / area);
    if (field_curv == 0.0f)
	{
		return;
	} else if (curv == 0.0f) {
		curv = field_curv;
	} else {
        curv = (field_curv > curv) ? curv : field_curv;
    }
}

__device__ float calc_error(float* p, float* points, float* grads)
{
    float err = 0;
    float tempv[3] = {0};
	for (size_t i = 0; i < 64; i++)
	{
        tempv[0] = p[0] - points[i * 4 + 0];
        tempv[1] = p[1] - points[i * 4 + 1];
        tempv[2] = p[2] - points[i * 4 + 2];
		err += squared(p[3] - dot(grads + i * 3, tempv, 3)) / (1 + squared_norm(grads + i * 3, 3));
	}
	return err;
}

__device__ void feature_calc(
    float* minV, float* maxV, const float cell_size, const float half_length, float* center,
    float* sample_points, float* sample_grads, 
    float* node)
{
    const float border = cell_size / 16.0;
    float borderMinV[3] = {center[0] - half_length + border, center[1] - half_length + border, center[2] - half_length + border};
    float borderMaxV[3] = {center[0] + half_length - border, center[1] + half_length - border, center[2] + half_length - border};
    float qef_normal[15] = {0};
    float p[3] = {0};
    float pl[5] = {0};
    int node_index;
    for (size_t x = 0; x < 4; x++)
    {
        for (size_t y = 0; y < 4; y++)
        {
            for (size_t z = 0; z < 4; z++)
            {
                node_index = z * 16 + y * 4 + x;
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
    for (size_t i = 0; i < 4; i++)
    {
        int index = ((11 - i) * i) / 2;
        for (size_t j = i; j < 4; j++)
        {
            A[i * 4 + j] = qef_normal[index + j - i];
            A[j * 4 + i] = A[i * 4 + j];
        }
        B[i] = -qef_normal[index + 4 - i];
    }
    bool is_out = true;
    float err = 1e30f;
    float pc[4] = {0};
    // float pcg[3] = {0};
    for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
    {
        if (cell_dim == 3)
        {
            float rvalue[4] = {0};
            float inv[16] = {0};
            mat_inverse(A, inv, 4);
            for (size_t i = 0; i < 4; i++)
            {
                rvalue[i] = 0;
                for (size_t j = 0; j < 4; j++)
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
				err = calc_error(pc, sample_points, sample_grads);
				node[0] = pc[0];
				node[1] = pc[1];
				node[2] = pc[2];
                node[3] = pc[3];
            }
        } else if (cell_dim == 2) {
			for (int face = 0; face < 6; face++)
            {
                int dir = face / 2;
				int side = face % 2;
                float* corners[2] = {borderMinV, borderMaxV};
                float AC[25] = {0};
                float BC[5] = {0};
                for (size_t i = 0; i < 5; i++)
                {
                    for (size_t j = 0; j < 5; j++)
                    {
                        AC[i * 5 + j] = (i < 4 && j < 4) ? A[i * 5 + j] : 0;
                    }
                    BC[i] = (i < 4) ? B[i] : 0;
                }
                AC[20 + dir] = AC[dir * 5 + 4] = 1;
                BC[4] = corners[side][dir];
                
                float rvalue[5] = {0};
                float inv[25] = {0};
                mat_inverse(AC, inv, 5);
                for (size_t i = 0; i < 5; i++)
                {
                    rvalue[i] = 0;
                    for (size_t j = 0; j < 5; j++)
                    {
                        rvalue[i] += inv[j * 5 + i] * BC[j];
                    }
                }
                pc[0] = rvalue[0];
                pc[1] = rvalue[1];
                pc[2] = rvalue[2];
                pc[3] = rvalue[3];

                int dp = (dir + 1) % 3;
                int dpp = (dir + 2) % 3;
                if (pc[dp] >= borderMinV[dp] && pc[dp] <= borderMaxV[dp] &&
					pc[dpp] >= borderMinV[dpp] && pc[dpp] <= borderMaxV[dpp])
                {
                    is_out = false;
					float e = calc_error(pc, sample_points, sample_grads);
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
                int dir = edge / 4;
				int side = edge % 4;
                float* corners[2] = {borderMinV, borderMaxV};
                float AC[36] = {0};
                float BC[6] = {0};
                for (int i = 0; i < 6; i++)
				{
					for (int j = 0; j < 6; j++)
					{
						AC[i * 6 + j] = (i < 4 && j < 4 ? A[i * 4 + j] : 0);
					}
					BC[i] = (i < 4 ? B[i] : 0);
				}
                int dp = (dir + 1) % 3;
				int dpp = (dir + 2) % 3;
                AC[24 + dp] = AC[dp * 6 + 4] = 1;
				AC[30 + dpp] = AC[dpp * 6 + 5] = 1;
				BC[4] = corners[side & 1][dp];
				BC[5] = corners[side >> 1][dpp];

                float rvalue[6] = {0};
                float inv[36] = {0};
                mat_inverse(AC, inv, 6);
                for (size_t i = 0; i < 6; i++)
                {
                    rvalue[i] = 0;
                    for (size_t j = 0; j < 6; j++)
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
					float e = calc_error(pc, sample_points, sample_grads);
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
                float* corners[2] = {borderMinV, borderMaxV};
                float AC[49] = {0};
                float BC[7] = {0};
                for (int i = 0; i < 7; i++)
				{
					for (int j = 0; j < 7; j++)
					{
						AC[i * 7 + j] = (i < 4 && j < 4 ? A[i * 4 + j] : 0);
					}
					BC[i] = (i < 4 ? B[i] : 0);
				}
                for (int i = 0; i < 3; i++)
                {
                    AC[(4 + i) * 4 + i] = AC[i * 4 + 4 + i] = 1;
                    BC[4 + i] = corners[(vertex >> i) & 1][i];
                }

                float rvalue[7] = {0};
                float inv[49] = {0};
                mat_inverse(AC, inv, 7);
                for (size_t i = 0; i < 7; i++)
                {
                    rvalue[i] = 0;
                    for (size_t j = 0; j < 7; j++)
                    {
                        rvalue[i] += inv[j * 7 + i] * BC[j];
                    }
                }
                node[0] = pc[0];
                node[1] = pc[1];
                node[2] = pc[2];
                node[3] = pc[3];
            }
        }
    }
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
__global__ void cuda_node_calc_const_r(
    float* particles, float* Gs, bool* splashs, float* particles_gradients, 
    long long* hash_list, int* index_list, long long* start_list_keys, int* start_list_values, long long* end_list_keys, int* end_list_values,
    char* types, char* depths, float* centers, float* half_lengthes, int* tnode_num,
    float* nodes
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int PARTICLES_NUM = PARTICLES_NUM_MIN_DEPTH[0];
    const float RADIUS = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[0];
    const float INF_FACTOR = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[1];
    const float INFLUNCE = RADIUS * INF_FACTOR;
    const float ISO_VALUE = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[2];
    const float MIN_SCALAR = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[3];
    const float MAX_SCALAR = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[4];
    const int DEPTH_MIN = PARTICLES_NUM_MIN_DEPTH[1];
    const float CELL_SIZE = RADIUS_INF_FACTOR_ISO_MIN_MAX_SCALAR_CELL_SIZE[5];
    const int TNODES_NUM = tnode_num[0];
    if (index < TNODES_NUM)
    {
        bool empty = false;
        bool signchange = false;
        float sample_points[256] = {0};
        float sample_grads[192] = {0};
        float curv = 0.0f;
        float cell_size = half_lengthes[index] * 2;
        float minV[3] = {centers[index * 3 + 0] - half_lengthes[index], centers[index * 3 + 1] - half_lengthes[index], centers[index * 3 + 2] - half_lengthes[index]};
        float maxV[3] = {centers[index * 3 + 0] + half_lengthes[index], centers[index * 3 + 1] + half_lengthes[index], centers[index * 3 + 2] + half_lengthes[index]};
        check_empty_and_calc_curv_const_r(
            particles, splashs, particles_gradients, RADIUS, INFLUNCE,
            hash_list, index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            CELL_SIZE, 
            minV, maxV, 
            empty, curv
        );
        if (empty)
        {
            types[index] = 0;
            nodes[index * 4 + 0] = centers[index * 3 + 0];
            nodes[index * 4 + 0] = centers[index * 3 + 0];
            nodes[index * 4 + 0] = centers[index * 3 + 0];
            return;
        }
        if (depths[index] < DEPTH_MIN)
        {
            types[index] = 1;
            return;
        }
        generate_sampling_points(minV, maxV, sample_points);
        node_sampling_const_r(
            particles, Gs, splashs, particles_gradients, PARTICLES_NUM, RADIUS, INF_FACTOR, ISO_VALUE, MIN_SCALAR, MAX_SCALAR,
            hash_list, index_list, start_list_keys, start_list_values, end_list_keys, end_list_values, 
            CELL_SIZE, 
            curv, signchange, sample_points, sample_grads
        );
        if ((CELL_SIZE - RADIUS) < TOLERANCE)
        {
            types[index] = 2;
            feature_calc(minV, maxV, cell_size, half_lengthes[index], centers + index * 3, sample_points, sample_grads, nodes + index * 3);
        }
        if (signchange && (curv < FLATNESS))
        {
            types[index] = 2;
        }
    }
}