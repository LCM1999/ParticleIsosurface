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

__global__ void estimateTotalInfluenceParticlesKernel(uint64_t* d_iso_tree, int d_iso_tree_size,
		                                              Vec3f* d_iso_centers, Vec3f* d_iso_sizes,
													  HashGridGPU** d_searchers, int d_searchers_size,
													  int* d_estimateNeighborsNums){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= d_iso_tree_size) return;
	Vec3f center = d_iso_centers[tid];
	Vec3f box1 = center - Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	Vec3f box2 = center + Vec3f(d_iso_sizes[tid].x, d_iso_sizes[tid].y, d_iso_sizes[tid].z);
	// int* curr_estimateNeighborsNumsBegin = d_estimateNeighborsNums + tid * d_searchers_size;
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
	// int curr_d_estimateNeighborsNumsBeginIdx = tid * d_searchers_size;
	// int curr_d_estimateNeighborsNumsEndIdx = tid * d_searchers_size + d_searchers_size;
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
	d_evaluator->EvalInNode(box1, box2, nodeSamplePoints, nodeSampleScalars, tmp_numNeighbors, d_totalInsideParticlesIdx + d_particlesBeginIdx, signchange);
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

	printf("-= Box =-\n");
	loadRootBox();

	int particles_size = _GlobalParticles.size();
	printf("-= Build Neighbor Searcher =-\n");
	// if (IS_CONST_RADIUS)
	// {
    	// _hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 4.0f);
	// } else {
	useCPU = false;
    _searcherGPU = std::make_shared<MultiLevelSearcherGPU>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
	// }
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
	if (IS_CONST_RADIUS)
	{
		resizeRootBoxConstR();
	} else {
		resizeRootBoxVarR();
	}
	printf("   MAX_DEPTH = %d, MIN_DEPTH = %d\n", _DEPTH_MAX, _DEPTH_MIN);

	_evaluator->CalculateMaxScalarVarR();
    printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
	_evaluator->RecommendIsoValueVarR();
    printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());
	if (CALC_P_NORMAL)
	{
		_evaluator->CalcParticlesNormal();
		printf("   Calculate Particals Normal Time = %f\n", t.elapsed());
		t.reset();
	}

	EvaluatorGPU evaluatorGPU(*_evaluator);
	EvaluatorGPU* d_evaluator;
	cudaMalloc(&d_evaluator, sizeof(EvaluatorGPU));
	cudaMemcpy(d_evaluator, &evaluatorGPU, sizeof(EvaluatorGPU), cudaMemcpyHostToDevice);

	// ----------- generating iso surface octree ---------
    // -------- assign data and sort the coordinate with morton code-------
    std::vector<float> scalars; // used to store each leaf nodes' scalar value on dual vertices. important for isosurface generation

	std::vector<uint64_t> iso_tree;
	iso_tree.resize(1 + 1);
	cal::fill_data_cpu(iso_tree.data(), 1, 0);
	cal::fill_data_cpu(iso_tree.data() + 1, 1, uint64_t(1) << 63);
	int iso_count = 0;
	
    thrust::device_vector<uint64_t> d_iso_tree(iso_tree);
    uint64_t* d_iso_treePtr = thrust::raw_pointer_cast(d_iso_tree.data());
	
	cstoneOctree::Box box(_BoundingBox[0], _BoundingBox[1], _BoundingBox[2], 
						  _BoundingBox[3], _BoundingBox[4], _BoundingBox[5]);
	cstoneOctree::Box* d_box;
	cudaMalloc(&d_box, sizeof(cstoneOctree::Box));
	cudaMemcpy(d_box, &box, sizeof(cstoneOctree::Box), cudaMemcpyHostToDevice);

	thrust::host_vector<uint64_t> mortonCodes(particles_size, 0);
	thrust::device_vector<cstoneOctree::Vec3f> d_GlobalParticles = _GlobalParticles;
	thrust::device_vector<uint64_t> d_mortonCodes = mortonCodes;

	uint64_t* d_mortonCodesPtr = thrust::raw_pointer_cast(d_mortonCodes.data());
	cstoneOctree::Vec3f* d_GlobalParticlesPtr = thrust::raw_pointer_cast(d_GlobalParticles.data());
	DeviceConfig cudaConfig(particles_size);
	calMortonCodeGPUKenrel<<<cudaConfig.blocks, cudaConfig.threads>>>(d_GlobalParticlesPtr, d_mortonCodesPtr, d_box, particles_size);
	cudaDeviceSynchronize();
    std::cout << "works" << std::endl;
    while(1){
        // --- iso octree's info calculation ---
        int d_iso_tree_size = d_iso_tree.size() - 1;
        thrust::device_vector<uint64_t> d_prefixes(d_iso_tree_size);
        uint64_t* d_prefixesPtr = thrust::raw_pointer_cast(d_prefixes.data());

        // calculate prefixes for each node
        DeviceConfig isoTreeConfig(d_iso_tree_size);
        cstoneOctree::calculatePrefixesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_prefixesPtr, d_iso_tree_size);
        cudaDeviceSynchronize();
        std::cout << "prefixes calculation done" << std::endl;
		thrust::device_vector<float> d_scalars(d_iso_tree_size, 0.0f);
        thrust::device_vector<Vec3f> d_iso_centers(d_iso_tree_size);
        thrust::device_vector<Vec3f> d_iso_sizes(d_iso_tree_size);
		
        thrust::device_vector<unsigned> d_iso_depths(d_iso_tree_size);
        Vec3f* d_iso_centersPtr = thrust::raw_pointer_cast(d_iso_centers.data());
        Vec3f* d_iso_sizesPtr = thrust::raw_pointer_cast(d_iso_sizes.data());
        unsigned* d_iso_depthsPtr = thrust::raw_pointer_cast(d_iso_depths.data());
        calculateLeavesCentersAndSizesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size, d_iso_centersPtr, d_iso_sizesPtr, d_iso_depthsPtr, d_box);
        cudaDeviceSynchronize();
        std::cout << "leaves centers and sizes calculation done" << std::endl;

        thrust::host_vector<int> iso_nodeOps(d_iso_tree_size, 0); // Store the split decision for each node (the split decision is based on whether the node has isosurface)
        thrust::device_vector<int> d_iso_nodeOps(iso_nodeOps);
        // ----------------------- begin split calculation ---------------------------
		// estimate total number of searched particles through each octree node
		std::cout << "the searcher's size is " << _searcherGPU->searchers.size() << std::endl;
		thrust::host_vector<int> estimateNeighborsNums(d_iso_tree_size, 0);
		thrust::device_vector<int> d_estimateNeighborsNums(estimateNeighborsNums);
		int* d_estimateNeighborsNumPtr = thrust::raw_pointer_cast(d_estimateNeighborsNums.data());
		
		// for each resolution particles level, calculate each estimate neighbors number for each octree node
		int HashGridGPUs_size = _searcherGPU->searchers.size();
		HashGridGPU** d_searcher;	
		cudaMalloc(&d_searcher, sizeof(HashGridGPU*) * HashGridGPUs_size);
		cudaMemcpy(d_searcher, _searcherGPU->searchers.data(), sizeof(HashGridGPU*) * HashGridGPUs_size, cudaMemcpyHostToDevice);
		// calculate the estimate neighbors number for each octree node
		estimateTotalInfluenceParticlesKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size,
																								d_iso_centersPtr, d_iso_sizesPtr,
																								d_searcher, HashGridGPUs_size,
																								d_estimateNeighborsNumPtr);
		cudaDeviceSynchronize();
		std::cout << "estimate neighbors number calculation done" << std::endl;
		estimateNeighborsNums = d_estimateNeighborsNums;

		for(int i = 0; i < estimateNeighborsNums.size(); i++){
			int val = estimateNeighborsNums[i];
			std::cout << "estimateNeighborsNums[" << i << "] = " << val << std::endl;
		}
		int total_estimateNeighborsNum = std::accumulate(estimateNeighborsNums.begin(), estimateNeighborsNums.end(), 0);
		thrust::device_vector<int> d_totalInsideParticlesIdx(total_estimateNeighborsNum, -1);
		thrust::host_vector<int> estimateNeighborsNumsLayout(estimateNeighborsNums.size(), 0);
		thrust::exclusive_scan(estimateNeighborsNums.begin(), estimateNeighborsNums.end(), estimateNeighborsNumsLayout.begin(), 0);
		thrust::device_vector<int> d_estimateNeighborsNumsLayout(estimateNeighborsNumsLayout);
		int* d_estimateNeighborsNumsLayoutPtr = thrust::raw_pointer_cast(d_estimateNeighborsNumsLayout.data());
		calculateSplitsKernel<<<isoTreeConfig.blocks, isoTreeConfig.threads>>>(d_iso_treePtr, d_iso_tree_size, d_iso_depthsPtr, 
																			   	thrust::raw_pointer_cast(d_scalars.data()), 
																				d_iso_centersPtr, d_iso_sizesPtr,
																				d_searcher, HashGridGPUs_size, _DEPTH_MIN, _DEPTH_MAX,
																				d_evaluator,
																				d_estimateNeighborsNumPtr, d_estimateNeighborsNumsLayoutPtr,
																				thrust::raw_pointer_cast(d_totalInsideParticlesIdx.data()),
																				thrust::raw_pointer_cast(d_iso_nodeOps.data()));
		cudaDeviceSynchronize();
		thrust::host_vector<int> totalInsidesParticlesIdx = d_totalInsideParticlesIdx;
		thrust::host_vector<float> iso_scalars = d_scalars;
		iso_nodeOps = d_iso_nodeOps;
		uint64_t iso_allOpsSum = std::accumulate(iso_nodeOps.begin(), iso_nodeOps.end(), 0);
		std::cout << "iso_allOpsSum = " << iso_allOpsSum << std::endl;
		std::vector<uint64_t> iso_nodeOpsLayout(iso_nodeOps.size());
		int newInternalNodes = 0;
		for(int i = 0; i < iso_nodeOps.size() - 1; i++){
			int val = iso_nodeOps[i] - 1;
			if(val != 0 && val != 7){
				std::cout << "error in iso_nodeOps, " << i << "th value is " << val << std::endl;
			}
			if (iso_nodeOps[i] == 8)
			{
				newInternalNodes += 8;
			}
		}
		std::cout << newInternalNodes << std::endl;
		std::exclusive_scan(iso_nodeOps.begin(), iso_nodeOps.end(), iso_nodeOpsLayout.begin(), 0);
		uint64_t newTreeNodesNum;	
		newTreeNodesNum = iso_nodeOpsLayout[iso_tree.size() - 1];
		std::vector<uint64_t> new_iso_tree(newTreeNodesNum + 1);  // updated tree array

		updateTreeArrayCPU(iso_nodeOpsLayout, iso_tree, new_iso_tree);

		std::copy_n(iso_tree.data() + iso_tree.size() - 1, 1, new_iso_tree.data() + new_iso_tree.size() - 1);
		std::swap(new_iso_tree, iso_tree);

		iso_count++;
		if(iso_allOpsSum == iso_nodeOps.size() - 1) break;
		break;
    }
	// while(1){

	// 	std::vector<Vec3f> iso_centers(iso_tree_size);
	// 	std::vector<Vec3f> iso_sizes(iso_tree_size);
	// 	std::vector<unsigned> iso_depths(iso_tree_size);
	//  calculateLeavesCentersAndSizesCPU(iso_tree, iso_centers, iso_sizes, iso_depths, box);

	// 	// ---- iso surface split decision making ----
	// 	std::vector<float> curvs(iso_tree_size);
	// 	std::vector<float> min_radiuses(iso_tree_size);	
	// 	std::vector<unsigned char> emptys(iso_tree_size, 0);
	// 	scalars = std::vector<float>(iso_tree_size, 0.0); // Stores each tree nodes' scalar value on dual vertices
	// 	std::vector<uint64_t> iso_nodeOps(iso_tree.size(), 0); // Store the split decision for each node (the split decision is based on whether the node has isosurface)
	// 				// bool
	// 	// #pragma omp parallel for
	// 	for(int i = 0; i < iso_tree_size; i++) {
	// 		// begin beforeSampleEval
	// 		Vec3f center = iso_centers[i];
	// 		Vec3f box1 = center - Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
	// 		Vec3f box2 = center + Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
	// 		int estimateNeighborsNum = 0;
	// 		int trueNeighborsNum = 0;
	// 		std::vector<int> insideParticlesIdx;
	// 		if (IS_CONST_RADIUS)
	// 		{
	// 			_hashgrid->GetInBoxEstimate(box1, box2, estimateNeighborsNum);
	// 			insideParticlesIdx.resize(estimateNeighborsNum);
	// 			_hashgrid->GetInBoxParticles(box1, box2, trueNeighborsNum, estimateNeighborsNum, insideParticlesIdx.data());
	// 		} else {
	// 			_searcher->GetInBoxEstimate(box1, box2, estimateNeighborsNum);
	// 			insideParticlesIdx.resize(estimateNeighborsNum);
	// 			_searcher->GetInBoxParticles(box1, box2, trueNeighborsNum, estimateNeighborsNum, insideParticlesIdx.data());
	// 		}
	// 		// check empty and calculate curvature implentation below
	// 		cstoneOctree::Vec3f norms(0, 0, 0);
	// 		float area = 0.0f;
	// 		min_radiuses[i] = IS_CONST_RADIUS ? _GlobalRadiuses[i] : FLT_MAX;
	// 		emptys[i] = trueNeighborsNum == 0;
	// 		if (!emptys[i])
	// 		{
	// 			bool allSplash = true;
	// 			for (int j = 0; j < trueNeighborsNum; j++) {
	// 				int in = insideParticlesIdx[j];
	// 				if (!_evaluator->CheckSplash(in))
	// 				{
	// 					if (_GlobalParticles[in].x > (box1.x - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
	// 						_GlobalParticles[in].x < (box2.x + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
	// 						_GlobalParticles[in].y > (box1.y - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
	// 						_GlobalParticles[in].y < (box2.y + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
	// 						_GlobalParticles[in].z > (box1.z - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
	// 						_GlobalParticles[in].z < (box2.z + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())))
	// 					{
	// 						if (CALC_P_NORMAL)
	// 						{
	// 							cstoneOctree::Vec3f tempNorm = _evaluator->PariclesNormals[in];
	// 							norms += tempNorm;
	// 							area += tempNorm.norm();
	// 						}
	// 						if (!IS_CONST_RADIUS)
	// 						{
	// 							if (min_radiuses[i] > _GlobalRadiuses[in])
	// 							{
	// 								min_radiuses[i] = _GlobalRadiuses[in];
	// 							}
	// 						}
	// 						allSplash = false;
	// 					}
	// 				}
	// 			}
	// 			emptys[i] = allSplash;
	// 		}
	// 		curvs[i] = (area == 0) ? 1.0 : (norms.norm() / area);
	// 		if (emptys[i]) {
	// 			scalars[i] = _evaluator->getIsoValue();
	// 			// nodes_type[i] = 0;
	// 			iso_nodeOps[i] = 1;
	// 			continue;
	// 		}
	// 		bool isbig = (iso_depths[i] < _DEPTH_MIN);
	// 		if (isbig)
	// 		{
	// 			iso_nodeOps[i] = 8;
	// 			continue;
	// 		}
	// 		bool signchange = false;
	// 		std::vector<float> nodeSamplePoints(pow(2 + 1, 3) * 3);
	// 		std::vector<float> nodeSampleScalars(pow(2 + 1, 3), 0);
	// 		std::vector<float> nodeSampleGrads(pow(2+1, 3) * 3);
	// 		for (float z = 0; z <= 2; z++)
	// 		{
	// 			for (float y = 0; y <= 2; y++)
	// 			{
	// 				for (float x = 0; x <= 2; x++)
	// 				{
	// 					nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 0] = 
	// 					(1 - x / 2) * box1[0] + (x / 2) * box2[0];
	// 					nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 1] = 
	// 					(1 - y / 2) * box1[1] + (y / 2) * box2[1];
	// 					nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 2] = 
	// 					(1 - z / 2) * box1[2] + (z / 2) * box2[2];
	// 				}
	// 			}
	// 		}
	// 		// grid sampling
	// 		bool origin_sign;
	// 		float cellSize = iso_sizes[i][0] * 2;
	// 		float step = cellSize / 2;
	// 		for (int j = 0; j < pow(2+1, 3); j++)
	// 		{
	// 			Vec3f diff;
	// 			Vec3f samplePoint(nodeSamplePoints[j * 3 + 0], nodeSamplePoints[j * 3 + 1], nodeSamplePoints[j * 3 + 2]);
	// 			for (int k = 0; k < trueNeighborsNum; k++)
	// 			{
	// 				int pIdx = insideParticlesIdx[k];
	// 				if (_evaluator->CheckSplash(pIdx))
	// 				{
	// 					continue;
	// 				}
	// 				diff = samplePoint - _evaluator->GlobalxMeans[pIdx];
	// 				nodeSampleScalars[j] += _evaluator->AnisotropicInterpolate(pIdx, diff);
	// 				// if (USE_ANI)
	// 				// {
	// 				// }
	// 				// else {
	// 				// 	diff = pos - (*GlobalPoses)[pIdx];
	// 				// 	scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
	// 				// }
	// 			}
	// 			nodeSampleScalars[j] = _evaluator->getIsoValue() - nodeSampleScalars[j];
	// 			origin_sign = (nodeSampleScalars[0] >= 0);
	// 			if (!signchange)
	// 			{
	// 				signchange = origin_sign ^ (nodeSampleScalars[j] >= 0);
	// 			}
	// 		}
	// 		// after process
	// 		if ((cellSize - min_radiuses[i]) < 1e-8)
	// 		{
	// 			// it's a leaf
	// 			// nodes_type[i] = LEAF;
	// 			iso_nodeOps[i] = 1;
	// 			scalars[i] = nodeSampleScalars[13];
	// 			continue;
	// 		}
	// 		// check curvature
	// 		if (isbig || (signchange && curvs[i] < 0.995))//
	// 		{
	// 			// tnode->type = INTERNAL;
	// 			iso_nodeOps[i] = 8;
	// 			continue;
	// 		}
	// 		else
	// 		{
	// 			// tnode->type = LEAF;
	// 			iso_nodeOps[i] = 1;
	// 			scalars[i] = nodeSampleScalars[13];
	// 			// _evaluator->SingleEval(tnode->center, tnode->nodeScalar);
	// 			continue;
	// 		}
	// 	}
	// 	uint64_t iso_allOpsSum = std::accumulate(iso_nodeOps.begin(), iso_nodeOps.end(), 0);
	// 	// exclusive_csan ops to get new octree indices
	// 	std::vector<uint64_t> iso_nodeOpsLayout(iso_nodeOps.size());
	// 	int newInternalNodes = 0;
	// 	for(int i = 0; i < iso_nodeOps.size() - 1; i++){
	// 		int val = iso_nodeOps[i] - 1;
	// 		if(val != 0 && val != 7){
	// 			std::cout << "error in iso_nodeOps, " << i << "th value is " << val << std::endl;
	// 		}
	// 		if (iso_nodeOps[i] == 8)
	// 		{
	// 			newInternalNodes += 8;
	// 		}
	// 	}
	// 	std::cout << newInternalNodes << std::endl;
	// 	std::exclusive_scan(iso_nodeOps.begin(), iso_nodeOps.end(), iso_nodeOpsLayout.begin(), 0);
	// 	uint64_t newTreeNodesNum;	
	// 	newTreeNodesNum = iso_nodeOpsLayout[iso_tree.size() - 1];
	// 	std::vector<uint64_t> new_iso_tree(newTreeNodesNum + 1);  // updated tree array

	// 	updateTreeArrayCPU(iso_nodeOpsLayout, iso_tree, new_iso_tree);

	// 	std::copy_n(iso_tree.data() + iso_tree.size() - 1, 1, new_iso_tree.data() + new_iso_tree.size() - 1);
	// 	std::swap(new_iso_tree, iso_tree);

	// 	iso_count++;
	// 	if(iso_allOpsSum == iso_nodeOps.size() - 1) break;
	// } // end octree generation loop

	// std::vector<Vec3i> iso_lowers(iso_tree.size() - 1);
	// std::vector<unsigned> iso_levels(iso_tree.size() - 1);
	// calculateLeavesLowersAndLevelsCPU(iso_tree, iso_lowers, iso_levels, box);

	// iso::generateIso(iso_tree, iso_lowers, iso_levels, scalars, 0.0, _OurMesh);
	// if (GEN_SPLASH)
	// {
	// 	printf("-= Generate Splash =-\n");
	// 	std::vector<cstoneOctree::Vec3f> splash_pos;
	// 	std::vector<float> splash_radiuses;
	// 	for (int pIdx = 0; pIdx < getGlobalParticlesNum(); pIdx++)
	// 	{
	// 		if (_evaluator->CheckSplash(pIdx))
	// 		{
	// 			splash_pos.push_back(_GlobalParticles[pIdx]);
	// 			if (!IS_CONST_RADIUS)
	// 			{
	// 				splash_radiuses.push_back(_GlobalRadiuses[pIdx]);
	// 			}
	// 		}
	// 	}
	// 	if (IS_CONST_RADIUS)
	// 	{
	// 		_OurMesh->AppendSplash_ConstR(splash_pos, _RADIUS);
	// 	} else {
	// 		_OurMesh->AppendSplash_VarR(splash_pos, splash_radiuses);
	// 	}
	// }
	// // printf("Time generating polygons = %f\n", t_gen_mesh.elapsed());
	// std::cout << "Time generating polygons;" << std::endl;
}