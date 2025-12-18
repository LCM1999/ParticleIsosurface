#include "surface_reconstructor.h"
// #include "hash_grid.h"
#include "multi_level_searcher_gpu.cuh"
#include "evaluator.h"
#include "evaluatorGPU.cuh"
#include "iso_method_ours.h"
#include "global.h"
#include "visitorextract.h"
#include "traverse.h"
#include "timer.h"
#include "iso.cuh"
#include <var.h>
#include <octree_func.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace cal;
using namespace cstoneOctree;


__global__ void calculateParticlesNormalsKernel(HashGridGPU** d_searchers
												, int d_searchers_size
												, EvaluatorGPU* d_evaluator)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= d_evaluator->d_GlobalParticlesNum) return;
	cstoneOctree::Vec3f pos = d_evaluator->d_GlobalxMeans[tid];
	cstoneOctree::Vec3f normal = cstoneOctree::Vec3f(0.0f);
	for (int hgi = 0; hgi < d_searchers_size; hgi++)
	{
		HashGridGPU* cur_searcher = d_searchers[hgi];
        cstoneOctree::Vec3i xyzIdx;
        int64_t neighbor_hash = -1;
        cur_searcher->CalcXYZIdx(pos, xyzIdx);
        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    neighbor_hash = cur_searcher->CalcCellHash((xyzIdx + cstoneOctree::Vec3i(x, y, z)));
                    if (neighbor_hash < 0) {continue;}
                    int countIndex, startIndex, endIndex;
                    if ((cur_searcher->d_StartList[neighbor_hash] >= 0) && (cur_searcher->d_EndList[neighbor_hash] >= 0))
                    {
                        startIndex = cur_searcher->d_StartList[neighbor_hash];
                        endIndex = cur_searcher->d_EndList[neighbor_hash];
                    }
                    else
                    {
                        continue;
                    }
                    for (int countIndex = startIndex; countIndex < endIndex; countIndex++)
                    {
                        int pId = cur_searcher->d_PIndexes[cur_searcher->d_IndexList[countIndex]];
                        if (d_evaluator->CheckSplash(pId))
                        {
                            continue;
                        }
                        cstoneOctree::Vec3f diff = pos - d_evaluator->d_GlobalxMeans[pId];
                        normal += d_evaluator->AnisotropicInterpolateGrad(pId, diff);
                    }
                }
            }
        }
	}
	d_evaluator->d_PariclesNormals[tid] = normal;
}

__global__ void estimateTotalInfluenceParticlesKernel(uint64_t* d_iso_tree, int d_iso_tree_size,
		                                              Vec3f* d_iso_centers, Vec3f* d_iso_sizes,
													  HashGridGPU** d_searchers, int d_searchers_size,
													  int* d_estimateNeighborsNums){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= d_iso_tree_size) return;
	Vec3f center = d_iso_centers[tid];
	Vec3f box1 = center - Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	Vec3f box2 = center + Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	for(int i = 0; i < d_searchers_size; i++){
		HashGridGPU* cur_searcher = d_searchers[i];
		cur_searcher->GetInBoxEstimateGPU(box1, box2, d_estimateNeighborsNums[tid]);
	}
	
}


__global__ void calculateSplitsKernel(uint64_t* d_iso_tree, int d_iso_tree_size, unsigned* d_iso_depths,
										float* d_iso_scalars, Vec3f* d_iso_centers, Vec3f* d_iso_sizes,
										HashGridGPU** d_searchers, int d_searchers_size, int d_depth_min, int d_depth_max,
										EvaluatorGPU* d_evaluator,
										int* d_estimateNeighborsNums, int* d_estimateNeighborsNumsLayout, 
										int* d_totalInsideParticlesIdx,
										int* d_iso_nodeOps){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= d_iso_tree_size) return;
	Vec3f center = d_iso_centers[tid];
	Vec3f box1 = center - Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	Vec3f box2 = center + Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	int d_particlesBeginIdx = d_estimateNeighborsNumsLayout[tid];
	
	int tmp_numNeighbors = 0;
	for(int i = 0; i < d_searchers_size; i++){
		HashGridGPU* cur_searcher = d_searchers[i];
		cur_searcher->GetInBoxParticlesGPU(box1, box2, tmp_numNeighbors, d_totalInsideParticlesIdx + d_particlesBeginIdx);
	}
	float minRadius = FLT_MAX;
	bool empty = true;
	float curv = d_evaluator->EvalInNodeCurv(box1, box2, tmp_numNeighbors, d_totalInsideParticlesIdx + d_particlesBeginIdx, minRadius, empty);
	if (empty) {
		d_iso_scalars[tid] = d_evaluator->d_ISO_VALUE;
		d_iso_nodeOps[tid] = 1;
		return;
	}
	bool isbig = (d_iso_depths[tid] < d_depth_min);
	if (isbig)
	{
		d_iso_nodeOps[tid] = 8;
		return;
	}
	float nodeSamplePoints[27 * 3];
	float nodeSampleScalars[27] = {0};
	bool signchange = false;
	d_evaluator->EvalInNode(box1, box2, nodeSamplePoints, nodeSampleScalars, tmp_numNeighbors, d_totalInsideParticlesIdx + d_particlesBeginIdx, signchange, curv);
	if (d_iso_sizes[tid].x * 2 - minRadius < 1e-8) {
		d_iso_nodeOps[tid] = 1;
		d_iso_scalars[tid] = nodeSampleScalars[13];
		return;
	}
	if (isbig || (signchange && curv < 0.995))
	{
		d_iso_nodeOps[tid] = 8;
		return;
	} else {
		d_iso_nodeOps[tid] = 1;
		d_iso_scalars[tid] = nodeSampleScalars[13];
		return;
	}
}

void SurfReconstructor::RunGPU(float iso_factor, float smooth_factor){
	timer t;
	useCPU = false;
	printf("-= Box =-\n");
	loadRootBox(*std::max_element(_GlobalRadiuses.begin(), _GlobalRadiuses.end()));

	int particles_size = _GlobalParticles.size();
	printf("-= Build Neighbor Searcher =-\n");
    _searcherGPU = std::make_shared<MultiLevelSearcherGPU>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
	printf("   Build Neighbor Searcher Time = %f \n", t.elapsed());
	t.reset();

    printf("-= Initialize Evaluator =-\n");
	_evaluator = std::make_shared<Evaluator>(_searcherGPU, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
	_evaluator->setSmoothFactor(smooth_factor);
	_evaluator->setIsoFactor(iso_factor);
	_evaluator->compute_Gs_xMeans();
	printf("   Initialize Evaluator Time = %f \n", t.elapsed());
	t.reset();
	printf("-= Resize Box =-\n");
	resizeRootBoxVarR();
	printf("   MAX_DEPTH = %d, MIN_DEPTH = %d\n", _DEPTH_MAX, _DEPTH_MIN);

	_evaluator->CalculateMaxScalarVarR();
    printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
	_evaluator->RecommendIsoValueVarR();
    printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());
	// if (CALC_P_NORMAL)
	// {
	// 	_evaluator->CalcParticlesNormal();
	// 	printf("   Calculate Particals Normal Time = %f\n", t.elapsed());
	// 	t.reset();
	// }

	// for each resolution particles level, calculate each estimate neighbors number for each octree node
	int HashGridGPUs_size = _searcherGPU->h_searchers.size();
	HashGridGPU** d_searcher;	
	cudaMalloc(&d_searcher, sizeof(HashGridGPU*) * HashGridGPUs_size);
	cudaMemcpy(d_searcher, _searcherGPU->h_searchers.data(), sizeof(HashGridGPU*) * HashGridGPUs_size, cudaMemcpyHostToDevice);

	EvaluatorGPU evaluatorGPU(*_evaluator);
	EvaluatorGPU* d_evaluator;
	cudaMalloc(&d_evaluator, sizeof(EvaluatorGPU));
	cudaMemcpy(d_evaluator, &evaluatorGPU, sizeof(EvaluatorGPU), cudaMemcpyHostToDevice);
	std::cout << "hash searchers and evaluator upload device done" << std::endl;
	// ----------- generating iso surface octree ---------
    // -------- assign data and sort the coordinate with morton code-------
	thrust::host_vector<float> iso_scalars;
	std::vector<uint64_t> iso_tree;
	int initial_iso_tree_size = std::pow(8, std::min(_DEPTH_MIN, 6)) + 1;
	iso_tree.resize(initial_iso_tree_size);
	uint64_t max_value = uint64_t(1) << 63;
	uint64_t step = max_value / (initial_iso_tree_size - 1);
	for (size_t i = 0; i < initial_iso_tree_size; ++i) {
        iso_tree[i] = static_cast<uint64_t>(i * step);
    }
	iso_tree[initial_iso_tree_size - 1] = max_value;

	std::cout << "iso tree initialization done" << std::endl;
	
	int iso_count = 0;
	
	cstoneOctree::Box box(_BoundingBox[0], _BoundingBox[1], _BoundingBox[2], 
						  _BoundingBox[3], _BoundingBox[4], _BoundingBox[5]);
	cstoneOctree::Box* d_box;
	cudaMalloc(&d_box, sizeof(cstoneOctree::Box));
	cudaMemcpy(d_box, &box, sizeof(cstoneOctree::Box), cudaMemcpyHostToDevice);

	std::cout << "box upload device done" << std::endl;
	DeviceConfig cudaConfig(particles_size);
	{
		t.reset();
		calculateParticlesNormalsKernel<<<cudaConfig.blocks, cudaConfig.threads>>>(d_searcher, HashGridGPUs_size, d_evaluator);
		cudaDeviceSynchronize();
		printf("   Calculate Particals Normal Time = %f\n", t.elapsed());
		t.reset();
	}
	cudaDeviceSynchronize();

	thrust::device_vector<float> d_scalars(iso_tree.size(), 0.0f);
	thrust::device_vector<uint64_t> d_iso_tree(iso_tree);
	int d_iso_tree_size = d_iso_tree.size() - 1;
    while(1){
		d_iso_tree_size = d_iso_tree.size() - 1;
		d_scalars.resize(d_iso_tree_size);
        // --- iso octree's info calculation ---
		uint64_t* d_iso_treePtr = thrust::raw_pointer_cast(d_iso_tree.data());
		// std::cout << "d_iso_tree_size = " << d_iso_tree_size << std::endl;
        thrust::device_vector<uint64_t> d_prefixes(d_iso_tree_size);
        uint64_t* d_prefixesPtr = thrust::raw_pointer_cast(d_prefixes.data());

        // calculate prefixes for each node
        DeviceConfig isoTreeConfig(d_iso_tree_size);
        cstoneOctree::calculatePrefixesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_prefixesPtr, d_iso_tree_size);
        cudaDeviceSynchronize();
        // std::cout << "prefixes calculation done" << std::endl;
		// d_scalars = thrust::device_vector<float> (d_iso_tree_size, 0.0f);
        thrust::device_vector<Vec3f> d_iso_centers(d_iso_tree_size);
        thrust::device_vector<Vec3f> d_iso_sizes(d_iso_tree_size);
		
        thrust::device_vector<unsigned> d_iso_depths(d_iso_tree_size);
        Vec3f* d_iso_centersPtr = thrust::raw_pointer_cast(d_iso_centers.data());
        Vec3f* d_iso_sizesPtr = thrust::raw_pointer_cast(d_iso_sizes.data());
        unsigned* d_iso_depthsPtr = thrust::raw_pointer_cast(d_iso_depths.data());
        calculateLeavesCentersAndSizesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size, d_iso_centersPtr, d_iso_sizesPtr, d_iso_depthsPtr, d_box);
        cudaDeviceSynchronize();
        // std::cout << "leaves centers and sizes calculation done" << std::endl;

        // thrust::host_vector<int> iso_nodeOps(d_iso_tree.size(), 0); // Store the split decision for each node (the split decision is based on whether the node has isosurface)
        thrust::device_vector<int> d_iso_nodeOps(d_iso_tree.size(), 0);
		thrust::device_vector<uint64_t> d_iso_nodeOpsLayout(d_iso_tree.size());
		// std::vector<uint64_t> iso_nodeOpsLayout(iso_nodeOps.size());
        // ----------------------- begin split calculation ---------------------------
		int newInternalNodes = 0;
		uint64_t iso_allOpsSum = 0;
		// estimate total number of searched particles through each octree node
		// thrust::host_vector<int> estimateNeighborsNums(d_iso_tree_size, 0);
		thrust::device_vector<int> d_estimateNeighborsNums(d_iso_tree_size, 0);
		int* d_estimateNeighborsNumPtr = thrust::raw_pointer_cast(d_estimateNeighborsNums.data());
		// calculate the estimate neighbors number for each octree node
		estimateTotalInfluenceParticlesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size,
																								d_iso_centersPtr, d_iso_sizesPtr,
																								d_searcher, HashGridGPUs_size,
																								d_estimateNeighborsNumPtr);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
		}
		cudaDeviceSynchronize();
		// std::cout << "estimate neighbors number calculation done" << std::endl;
		// estimateNeighborsNums = d_estimateNeighborsNums;
		// int total_estimateNeighborsNum = std::accumulate(estimateNeighborsNums.begin(), estimateNeighborsNums.end(), 0);
		thrust::device_vector<int> d_estimateNeighborsNumsLayout(d_iso_tree_size);
		thrust::exclusive_scan(d_estimateNeighborsNums.begin(), d_estimateNeighborsNums.end(), d_estimateNeighborsNumsLayout.begin(), 0);
		int* d_estimateNeighborsNumsLayoutPtr = thrust::raw_pointer_cast(d_estimateNeighborsNumsLayout.data());
		int total_estimateNeighborsNum = thrust::reduce(d_estimateNeighborsNums.begin(), d_estimateNeighborsNums.end(), 0, thrust::plus<int>());
		thrust::device_vector<int> d_totalInsideParticlesIdx(total_estimateNeighborsNum, -1);
		calculateSplitsKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size, d_iso_depthsPtr, 
																				   thrust::raw_pointer_cast(d_scalars.data()), 
																				d_iso_centersPtr, d_iso_sizesPtr,
																				d_searcher, HashGridGPUs_size, _DEPTH_MIN, _DEPTH_MAX,
																				d_evaluator,
																				d_estimateNeighborsNumPtr, d_estimateNeighborsNumsLayoutPtr,
																				thrust::raw_pointer_cast(d_totalInsideParticlesIdx.data()),
																				thrust::raw_pointer_cast(d_iso_nodeOps.data()));
		cudaDeviceSynchronize();

		// iso_nodeOps = d_iso_nodeOps;
		thrust::exclusive_scan(d_iso_nodeOps.begin(), d_iso_nodeOps.end(), d_iso_nodeOpsLayout.begin(), 0);
		iso_allOpsSum = thrust::reduce(d_iso_nodeOps.begin(), d_iso_nodeOps.end(), 0);
		if(iso_allOpsSum == d_iso_tree_size) break;
		// std::exclusive_scan(iso_nodeOps.begin(), iso_nodeOps.end(), iso_nodeOpsLayout.begin(), 0);
		uint64_t lastNodeOffset = d_iso_nodeOpsLayout.back();
		// 获取最后一个元素的 op 值 (count)
		int lastNodeOp = d_iso_nodeOps.back();
		uint64_t newTreeNodesNum = lastNodeOffset + lastNodeOp;
		// std::vector<uint64_t> new_iso_tree(newTreeNodesNum + 1);  // updated tree array
		thrust::device_vector<uint64_t> d_new_iso_tree(newTreeNodesNum + 1);

		// updateTreeArrayCPU(iso_nodeOpsLayout, iso_tree, new_iso_tree);
		updateTreeArrayGPUKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(thrust::raw_pointer_cast(d_iso_nodeOpsLayout.data()),
																						 d_iso_treePtr,
																						 thrust::raw_pointer_cast(d_new_iso_tree.data()),
																						 d_iso_tree_size);
		cudaDeviceSynchronize();
		// std::cout << "tree update done" << std::endl;
		thrust::copy(d_iso_tree.end() - 1, d_iso_tree.end(), d_new_iso_tree.end() - 1);
		d_iso_tree.swap(d_new_iso_tree);
		// std::copy_n(iso_tree.data() + iso_tree.size() - 1, 1, new_iso_tree.data() + new_iso_tree.size() - 1);
		// std::swap(new_iso_tree, iso_tree);
		iso_count++;

    }

	printf("Octree generation time = %f\n", t.elapsed());

	// thrust::device_vector<uint64_t> d_iso_tree(iso_tree);
	// d_iso_tree_size = d_iso_tree.size() - 1;
	// d_scalars.resize(d_iso_tree_size);
	DeviceConfig isoTreeConfig(d_iso_tree_size);
	thrust::device_vector<Vec3i> d_iso_lowers(d_iso_tree_size);
	thrust::device_vector<unsigned> d_iso_levels(d_iso_tree_size);
	calculateLeavesLowersAndLevelsKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(thrust::raw_pointer_cast(d_iso_tree.data()), d_iso_tree_size,
																						  thrust::raw_pointer_cast(d_iso_lowers.data()),
																						  thrust::raw_pointer_cast(d_iso_levels.data())
																						);
	// d_scalars = iso_scalars;
	iso::generateIsoDirectGPU(d_iso_tree
		, d_iso_lowers, d_iso_levels, d_scalars, d_iso_tree_size, 
		d_searcher, HashGridGPUs_size, d_evaluator, 
		0.0, (_searcherGPU->minRadius / 2),
		d_box, &box,
		_OurMesh
	);
	if (GEN_SPLASH)
	{
		printf("-= Generate Splash =-\n");
		std::vector<cstoneOctree::Vec3f> splash_pos;
		std::vector<float> splash_radiuses;
		for (int pIdx = 0; pIdx < getGlobalParticlesNum(); pIdx++)
		{
			if (_evaluator->CheckSplash(pIdx))
			{
				splash_pos.push_back(_GlobalParticles[pIdx]);
				if (!IS_CONST_RADIUS)
				{
					splash_radiuses.push_back(_GlobalRadiuses[pIdx]);
				}
			}
		}
		if (IS_CONST_RADIUS)
		{
			_OurMesh->AppendSplash_ConstR(splash_pos, _RADIUS);
		} else {
			_OurMesh->AppendSplash_VarR(splash_pos, splash_radiuses);
		}
	}
	// std::cout << "Time generating polygons;" << std::endl;
	printf("-=  generate time = %f  =-\n", t.elapsed());
}