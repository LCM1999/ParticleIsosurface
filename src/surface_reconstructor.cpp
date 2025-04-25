#include <queue>
#include <omp.h>

#include "surface_reconstructor.h"
// #include "hash_grid.h"
// #include "multi_level_searcher.h"
// #include "evaluator.h"
#include "iso_method_ours.h"
#include "global.h"
#include "visitorextract.h"
#include "traverse.h"
#include "timer.h"
#include "iso.cuh"
#include <var.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// #include <morton.h>
using namespace cal;
using namespace cstoneOctree;

SurfReconstructor::SurfReconstructor(
	std::vector<cstoneOctree::Vec3f>& particles,
	std::vector<float>& radiuses, Mesh* mesh, 
	float radius)
{
	_GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalRadiuses = radiuses;
	_RADIUS = radius;

	WaitingStack.clear();

	queue_flag = 0;

	_OurMesh = mesh;
}

void SurfReconstructor::loadRootBox()
{
	_BoundingBox[0] = _BoundingBox[2] = _BoundingBox[4] = FLT_MAX;
	_BoundingBox[1] = _BoundingBox[3] = _BoundingBox[5] = -FLT_MAX;
	for (const cstoneOctree::Vec3f& p: _GlobalParticles)
	{
		if (p.x < _BoundingBox[0]) _BoundingBox[0] = p.x;
		if (p.x > _BoundingBox[1]) _BoundingBox[1] = p.x;
		if (p.y < _BoundingBox[2]) _BoundingBox[2] = p.y;
		if (p.y > _BoundingBox[3]) _BoundingBox[3] = p.y;
		if (p.z < _BoundingBox[4]) _BoundingBox[4] = p.z;
		if (p.z > _BoundingBox[5]) _BoundingBox[5] = p.z;
	}
	if (_BoundingBox[0] == _BoundingBox[1] ||
		_BoundingBox[2] == _BoundingBox[3] ||
		_BoundingBox[4] == _BoundingBox[5])
	{
		SINGLE_LAYER = true;
	}
}

void SurfReconstructor::resizeRootBoxConstR()
{
    float maxLen, resizeLen;
	float r = _RADIUS;
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / r))));
	resizeLen = pow(2, _DEPTH_MAX) * r;
	while (resizeLen - maxLen < (_evaluator->getNeighborFactor() * _RADIUS * 2))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * r;
	}
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		float center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}
	
	_DEPTH_MIN = (_DEPTH_MAX - (SINGLE_LAYER ? 1 : 2));	
}

void SurfReconstructor::resizeRootBoxVarR()
{
	float maxLen, resizeLen;
	float minR = useCPU ? _searcherCPU->getMinRadius() : _searcherGPU->getMinRadius(), 
		  maxR = useCPU ? _searcherCPU->getMaxRadius() : _searcherGPU->getMaxRadius(),
		  avgR = useCPU ? _searcherCPU->getAvgRadius() : _searcherGPU->getAvgRadius();
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / minR))));
	resizeLen = pow(2, _DEPTH_MAX) * minR;
	while (resizeLen - maxLen < (_evaluator->getNeighborFactor() * maxR * 2))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * minR;
	}
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		float center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}

	_DEPTH_MIN = std::min(int(std::ceil(std::log2(std::ceil(maxLen / maxR)))) - 1, _DEPTH_MAX-2); //, _DEPTH_MAX - int(_DEPTH_MAX / 3));
}

void SurfReconstructor::checkEmptyAndCalcCurv(std::shared_ptr<TNode> tnode, unsigned char& empty, float& curv, float& min_radius)
{
	cstoneOctree::Vec3f norms(0, 0, 0);
	float area = 0.0f;
	// int impact_num = 0;
	std::vector<int> insides;
	int real = 0, estiamte = 0;
	min_radius = IS_CONST_RADIUS ? _RADIUS : FLT_MAX;
	const cstoneOctree::Vec3f
	box1 = tnode->center - cstoneOctree::Vec3f(tnode->half_length, tnode->half_length, tnode->half_length),
	box2 = tnode->center + cstoneOctree::Vec3f(tnode->half_length, tnode->half_length, tnode->half_length);
	if (useCPU)
	{
		_searcherCPU->GetInBoxParticles(box1, box2, insides);
	} else {
		_searcherGPU->GetInBoxEstimate(box1, box2, estiamte);
		insides.resize(estiamte);
		_searcherGPU->GetInBoxParticles(box1, box2, real, estiamte, insides.data());
	}
	empty = insides.empty();
	if (!empty)
	{
		bool all_splash = true;
		for (const int& in: insides)
		{
			if (!_evaluator->CheckSplash(in))
			{
				if (_GlobalParticles[in].x > (box1.x - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].x < (box2.x + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
					_GlobalParticles[in].y > (box1.y - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].y < (box2.y + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
					_GlobalParticles[in].z > (box1.z - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].z < (box2.z + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())))
				{
					if (CALC_P_NORMAL)
					{
						cstoneOctree::Vec3f tempNorm = _evaluator->PariclesNormals[in];
						// if (tempNorm == Eigen::Vector3f(0, 0, 0))	{continue;}
						norms += tempNorm;
						area += tempNorm.norm();
					}

					// if (tnode->depth < _DEPTH_MIN)
					// {
					// 	empty = false;
					// 	curv = 0.0f;
					// 	if (!IS_CONST_RADIUS) 
					// 	{
					// 		min_radius = _searcher->getMinRadius();
					// 	}	
					// 	return;
					// }

					if (!IS_CONST_RADIUS)
					{
						if (min_radius > _GlobalRadiuses[in])
						{
							min_radius = _GlobalRadiuses[in];
						}
					}
					// impact_num++;
					all_splash = false;
				}
			}
		}
		empty = all_splash;
	}
	// if (impact_num < 25)
	// {
		// curv = 0;
	// } else {
		curv = (area == 0) ? 1.0 : (norms.norm() / area);
	// }
}

void SurfReconstructor::beforeSampleEval(std::shared_ptr<TNode> tnode, float& curv, float& min_radius, unsigned char& empty)
{

	checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
	if (empty)
	{
		// _evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3]);
		tnode->nodeScalar = _evaluator->getIsoValue();
		tnode->type = EMPTY;
		return;
	}
}

void SurfReconstructor::afterSampleEval(
	std::shared_ptr<TNode> tnode, float& curv, float& min_radius, 
	float* sample_points, float* sample_grads)
{
	bool isbig = (tnode->depth < _DEPTH_MIN);
	bool signchange = false;
	float cellsize = 2 * tnode->half_length;

	if (!isbig)
	{
		tnode->GenerateSampling(sample_points);
		tnode->NodeSampling(curv, signchange, cellsize, sample_points, sample_grads);
	}
	
	// judge this node need calculate iso-surface
	// check max/min sizes of cells
	if ((cellsize - min_radius) < 1e-8)
	{
		// it's a leaf
		tnode->type = LEAF;
		_evaluator->SingleEval(tnode->center, tnode->nodeScalar);
		// tnode->NodeCalcNode(sample_points, sample_grads, cellsize);
		return;
	}

	// check curvature
	if (isbig || (signchange && curv < 0.995))//
	{
		tnode->type = INTERNAL;
	}
	else
	{
		tnode->type = LEAF;
		// tnode->NodeCalcNode(sample_points, sample_grads, cellsize);
		_evaluator->SingleEval(tnode->center, tnode->nodeScalar);
	}
}

void SurfReconstructor::genIsoOurs()
{
    timer t;

	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	if (_STATE == 0)
	{
		printf("-= Calculating Tree Structure =-\n");
		_OurRoot = std::make_shared<TNode>(this, 0);
		_OurRoot->center = cstoneOctree::Vec3f(_RootCenter[0], _RootCenter[1], _RootCenter[2]);
		//_OurRoot->node << _RootCenter[0], _RootCenter[1], _RootCenter[2], 0.0;
		_OurRoot->nodeScalar = 0.0;
		_OurRoot->half_length = _RootHalfLength;
	} else if (_STATE == 1) {
		printf("-= Generate Surface =-\n");
		timer t_gen_mesh;
		_OurMesh->tris.reserve(1000000);
		VisitorExtract v(this, _OurMesh);
		TraversalData td(_OurRoot);
		traverse_node<trav_vert>(v, td);
		v.calc_vertices();
		v.generate_mesh();
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
		printf("Time generating polygons = %f\n", t_gen_mesh.elapsed());
		return;
	}
	// int depth = 0;
	// float half = _OurRoot->half_length;
	float* sample_points = nullptr;
	float* sample_grads = nullptr;
	std::vector<float> cuvrs;
	std::vector<float> min_raiduses;
	std::vector<unsigned char> emptys;
	WaitingStack.push_back(&_OurRoot);
	ProcessArray.resize(inProcessSize);
	int count = 0;
	int final_leaf_count = 0;
	while (!WaitingStack.empty())
	{
		std::cout << WaitingStack.size() << std::endl;
		for (queue_flag = 0; queue_flag < inProcessSize && !WaitingStack.empty(); queue_flag++)
		{
			ProcessArray[queue_flag] = WaitingStack.back();
			WaitingStack.pop_back();
		}
		cuvrs.clear();
		min_raiduses.clear();
		emptys.clear();
		cuvrs.resize(queue_flag);
		min_raiduses.resize(queue_flag);
		emptys.resize(queue_flag);
		{
			{
				for (size_t i = 0; i < queue_flag; i++)
				{
						beforeSampleEval(*ProcessArray[i], cuvrs[i], min_raiduses[i], emptys[i]);
				}
			}
		}
		if (sample_points != nullptr)
		{
			delete[] sample_points;
			sample_points = nullptr;
		}
		if (sample_grads != nullptr)
		{
			delete[] sample_grads;
			sample_grads = nullptr;
		}
		sample_points = new float[int(pow(getOverSampleQEF()+1, 3)) * 4 * queue_flag];
		sample_grads = new float[int(pow(getOverSampleQEF()+1, 3)) * 3 * queue_flag];
		//TODO: Sampling
		{
			{
				for (size_t i = 0; i < queue_flag; i++)
				{
					if (!emptys[i])
					{
							afterSampleEval(
									*ProcessArray[i], cuvrs[i], min_raiduses[i], 
									sample_points + int(i * pow(getOverSampleQEF()+1, 3) * 4), 
									sample_grads + int(i * pow(getOverSampleQEF()+1, 3) * 3));
					}
				}
			}
		}
		for (size_t i = 0; i < queue_flag; i++)
		{
			if ((*ProcessArray[i])->type == INTERNAL) {
				for (Index t = 0; t.v < 8; t++)
				{
					(*ProcessArray[i])->children[t.v] = std::make_shared<TNode>(this, *ProcessArray[i], t);
					WaitingStack.push_back(&((*ProcessArray[i])->children[t.v]));
				}
			}
			else{
				final_leaf_count++;
			}
		}
		// depth++;
		// half/=2;
		count++;
	}
	std::cout << "Final leaf count: " << final_leaf_count << std::endl;
	ProcessArray.clear();
	delete[] sample_points;
	delete[] sample_grads;
	printf("Time generating tree = %f\n", t.elapsed());	
	_STATE++;
}

/**
 * @brief Execute the surface extraction in cornerstone octree method on CPU
 * 
 * @param iso_factor 
 * @param smooth_factor 
 */
// void SurfReconstructor::RunCPU(float iso_factor, float smooth_factor){

// 	_OurMesh->reset();

// 	printf("-= Run =-\n");
// 	timer t;

// 	loadRootBox();


// 	// ===========1.1 Re-assign particles to Vec3f type============
// 	_particles.resize(_GlobalParticles.size());
// 	for(int i = 0; i < _GlobalParticles.size(); i++){
// 		_particles[i] = Vec3f(_GlobalParticles[i].x, _GlobalParticles[i].y, _GlobalParticles[i].z);
// 	}
// 	_GlobalParticles.clear();
// 	int particle_size = _particles.size();

// 	// float min_num = cal::min(_BoundingBox[0], cal::min(_BoundingBox[2], _BoundingBox[4]));
// 	// float max_num = cal::max(_BoundingBox[1], cal::max(_BoundingBox[3], _BoundingBox[5]));
// 	// std::cout << "The box min num: " << min_num << std::endl;
// 	// std::cout << "The box max num: " << max_num << std::endl;
// 	// Box box(min_num, max_num);
// 	// --- Freeze the box only for this dam project test --- 
// 	cstoneOctree::Box box(1.0526, 3.6126, -0.780035, 1.77996, -1.01442, 1.54558);
	
// 	// ============1.2 Calculate the morton code for each particle and sort the outputs===========
// 	_mortonCodes.resize(particle_size);
// 	calMortonCodeCPU(_particles, _mortonCodes, box);

// 	std::vector<size_t> indices(particle_size);
// 	for (size_t i = 0; i < indices.size(); ++i) {
// 		indices[i] = i;
// 	}
// 	std::sort(indices.begin(), indices.end(),
//               [&](size_t i, size_t j) { return _mortonCodes[i] < _mortonCodes[j]; });
// 	std::vector<Vec3f> particleSorted(particle_size);
// 	std::vector<float> rSorted(particle_size);
// 	std::vector<uint64_t> mortonCodesSorted(_mortonCodes.size());
// 	for (size_t i = 0; i < indices.size(); ++i) {
//         particleSorted[i] = _particles[indices[i]];
// 		if(IS_CONST_RADIUS){
// 			rSorted[i] = _RADIUS;
// 		}
// 		else{
// 			rSorted[i] = _GlobalRadiuses[indices[i]];
// 		}
// 		mortonCodesSorted[i] = _mortonCodes[indices[i]];
//     }
// 	_particles = particleSorted;
// 	_GlobalRadiuses = rSorted;
// 	_mortonCodes = mortonCodesSorted;	
// 	particleSorted.clear();
// 	rSorted.clear();
// 	mortonCodesSorted.clear();

// 	_GlobalParticles = std::vector<cstoneOctree::Vec3f>(_particles.size());
// 	for(int i = 0; i < _particles.size(); i++){
// 		_GlobalParticles[i] = cstoneOctree::Vec3f(_particles[i].x, _particles[i].y, _particles[i].z);
// 	}

// 	_tree.resize(1 + 1);
// 	cal::fill_data_cpu(_tree.data(), 1, 0);
// 	cal::fill_data_cpu(_tree.data() + 1, 1, uint64_t(1) << 63);

// 	int bucket_size = 8;
// 	_counts.resize(1);
// 	cal::fill_data_cpu(_counts.data(), 1, particle_size);
// 	int max_count = std::numeric_limits<int>::max();
// 	int count = 0;
	
// 	// ========================2 octree karray(leaf) construction loop start======================
// 	while(1){
// 		std::vector<uint64_t> nodeOps(_tree.size()); // Store the split decision for each node (dynamically during makeSplitDecision function). 
// 													// Start from (one root node + 1) size for scan.
// 		cal::fill_data_cpu(nodeOps.data(), nodeOps.size(), 0);
// 		makeSplitsDecisionsCPU(_tree, _counts, bucket_size, nodeOps);

// 		std::cout << "tree size in loop " << count << ": " << _tree.size() << std::endl;

// 		uint64_t allOpsSum = std::accumulate(nodeOps.begin(), nodeOps.end(), 0);

// 		// exclusive_csan ops to get new octree indices
// 		std::vector<uint64_t> nodeOpsLayout(nodeOps.size());
// 		std::exclusive_scan(nodeOps.begin(), nodeOps.end(), nodeOpsLayout.begin(), 0);
// 		uint64_t newTreeNodesNum;	
// 		newTreeNodesNum = nodeOpsLayout[_tree.size() - 1];
// 		std::vector<uint64_t> newTree(newTreeNodesNum + 1);  // updated tree array

// 		updateTreeArrayCPU(nodeOpsLayout, _tree, newTree);

// 		std::copy_n(_tree.data() + _tree.size() - 1, 1, newTree.data() + newTree.size() - 1);
// 		std::swap(newTree, _tree);

// 		// update node counts
// 		_counts.resize(_tree.size() - 1);
// 		std::vector<uint64_t> coverNodes(2);	
// 		findCoverNodesCPU(_tree, _mortonCodes, coverNodes);	

// 		// set the out of bound nodes (absolutely does not contain coordinates) to minCount(0).
// 		cal::fill_data_cpu(_counts.data(), coverNodes[0], 0);
// 		cal::fill_data_cpu(_counts.data() + coverNodes[1], _tree.size() - coverNodes[1] - 1, 0);

// 		updateNodeCountsCPU(_tree, _mortonCodes, _counts, coverNodes);

// 		count++;
// 		if(allOpsSum == nodeOps.size() - 1) break;
// 	}
// 	// ========================2 octree karray(leaf) construction loop end======================

// 	// ========================3 octree internal nodes construction start======================
// 	// construct octree internal nodes, which is equal to buildOctreeCpu() in cornerstone octree source code
// 	int numLeafNodes = _tree.size() - 1; // tree size minus 1 because the range end variable is added before.
//     int numInternalNodes = (numLeafNodes - 1) / 7; // number of nodes which is not leaf nodes
//     int numNodes         = numLeafNodes + numInternalNodes; // total number of nodes (leaf nodes + internal nodes)
// 	std::vector<uint64_t> prefixes;  // used to delete the righthand side 0 bits for each morton code
//     std::vector<TreeNodeIndex> internalToLeaf;
//     std::vector<TreeNodeIndex> leafToInternal;
//     std::vector<TreeNodeIndex> childOffsets;
// 	prefixes.resize(numNodes);
// 	internalToLeaf.resize(numNodes);
//     leafToInternal.resize(numNodes);
//     childOffsets.resize(numNodes + 1);
//     std::vector<TreeNodeIndex> parents;
//     std::vector<int> levelRange;
//     parents.resize(std::max(1, (numNodes - 1) / 8));
//     levelRange.resize(21 + 2);	

	
// 	// combine internal and leaf tree parts into a single array with the nodeKey prefixes
// 	createUnsortedLayoutCPU(_tree, numInternalNodes, numLeafNodes, prefixes, internalToLeaf);	
	

// 	cal::sort_by_key_cpu(prefixes.data(), prefixes.data() + prefixes.size(), internalToLeaf.data());

// 	invertOrderCPU(internalToLeaf, leafToInternal, numNodes, numInternalNodes);
// 	// Calculate node range for each tree level
// 	getLevelRangeCPU(prefixes, numNodes, levelRange);
// 	cal::fill_data_cpu(childOffsets.data(), numNodes + 1, 0);

// 	linkOctreeCPU(prefixes,
// 				numInternalNodes,
// 				leafToInternal,
// 				levelRange,
// 				childOffsets,
// 				parents);
	
// 	// ========================3 octree internal nodes construction end======================

// 	// calculate centers and sizes for each node
// 	std::cout << "end" << std::endl;	
// 	std::vector<Vec3f> centers(numNodes);
// 	std::vector<Vec3f> sizes(numNodes);
// 	calculateNodeCentersAndSizesCPU(prefixes, centers, sizes, box);

// 	// ========================4 neighbor search start======================
	
//     std::vector<int> layout(numLeafNodes + 1); // index of first particle for each leaf node
//     std::exclusive_scan(_counts.begin(), _counts.end(), layout.begin(), 0);
	
// 	int ngmax = 16; 
// 	_box = box;
// 	OctreeNs octreeNs(prefixes.data(), childOffsets.data(), internalToLeaf.data(), levelRange.data(), layout.data(), centers.data(), sizes.data());
// 	_octreeNs = octreeNs;

// 	_evaluator = std::make_shared<Evaluator>(&_particles, &_GlobalRadiuses, _octreeNs, _box, ngmax);
// 	_evaluator->setSmoothFactor(smooth_factor);
// 	_evaluator->setIsoFactor(iso_factor);
// 	//_evaluator->compute_Gs_xMeansCPU();

// 	printf("   Initialize Evaluator Time = %f \n", t.elapsed());
// 	t.reset();	

// 	printf("-= Resize Box =-\n");
// 	// shrinkBox();
// 	if (IS_CONST_RADIUS)
// 	{
// 		resizeRootBoxConstR();
// 	} else {
// 		resizeRootBoxVarR();
// 	}
// 	printf("   MAX_DEPTH = %d, MIN_DEPTH = %d\n", _DEPTH_MAX, _DEPTH_MIN);

// 	IS_CONST_RADIUS ? _evaluator->CalculateMaxScalarConstR() : _evaluator->CalculateMaxScalarVarR();
//     printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
// 	IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR() : _evaluator->RecommendIsoValueVarR();
//     printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());

// 	if (CALC_P_NORMAL)
// 	{
// 		_evaluator->CalcParticlesNormal();
// 		printf("   Calculate Particals Normal Time = %f\n", t.elapsed());
// 		t.reset();
// 	}

// 	// ----------- generating iso surface octree ---------
// 	std::vector<uint64_t> iso_tree;
// 	// iso_tree.assign(_tree.begin(), _tree.end());
// 	iso_tree.resize(1 + 1);
// 	cal::fill_data_cpu(iso_tree.data(), 1, 0);
// 	cal::fill_data_cpu(iso_tree.data() + 1, 1, uint64_t(1) << 63);

// 	int iso_count = 0;
// 	std::vector<float> scalars; // used to store each leaf nodes' scalar value on dual vertices. important for isosurface generation
	
// 	while(1){
// 		// ---- iso octree's info calculation ----
// 		int iso_tree_size = iso_tree.size() - 1;
// 		std::cout << "tree size in loop " << iso_count << ": " << iso_tree_size << std::endl;
// 		std::vector<uint64_t> iso_prefixes(iso_tree_size);

// 		// calculate prefixes for each node
// 		// for(int i = 0; i < iso_tree_size; i++){
// 		// 	uint64_t curr_key = iso_tree[i];
// 		// 	if(!isPowerOf8(iso_tree[i + 1] - curr_key)) printf("The index %d is not the power of 8\n", i);
//     	// 	unsigned curr_level = treeLevel(iso_tree[i + 1] - curr_key);
//     	// 	iso_prefixes[i] = encodePlaceholderBit(curr_key, 3 * curr_level);	
// 		// }
// 		std::vector<Vec3f> iso_centers(iso_tree_size);
// 		std::vector<Vec3f> iso_sizes(iso_tree_size);
// 		// calculateLeavesCentersAndSizesCPU(iso_tree, iso_centers, iso_sizes, box);

// 		// ---- iso surface split decision making ----
// 		std::vector<float> sample_points(int(std::pow(getOverSampleQEF() + 1, 3) * 4 * iso_tree_size));
// 		std::vector<float> sample_grads(int(std::pow(getOverSampleQEF() + 1, 3) * 4 * iso_tree_size));
// 		std::vector<float> curvs(iso_tree_size);
// 		std::vector<float> min_radiuses(iso_tree_size);	
// 		std::vector<unsigned char> emptys(iso_tree_size);
// 		// std::vector<float> iso_values(iso_tree_size);
// 		std::vector<int> nodes_type(iso_tree_size, 1);
// 		scalars = std::vector<float>(iso_tree_size, 0.0); // Stores each tree nodes' scalar value on dual vertices
// 		std::vector<uint64_t> iso_nodeOps(iso_tree.size(), 0); // Store the split decision for each node (the split decision is based on whether the node has isosurface)
// 					// bool
// 		// #pragma omp parallel for
// 		for(int i = 0; i < iso_tree_size; i++) {
// 			// begin beforeSampleEval
// 			Vec3f center = iso_centers[i];
// 			Vec3f box1 = center - Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
// 			Vec3f box2 = center + Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
// 			std::vector<int> insideParticlesIdx;
// 			// check empty and calculate curvature implentation below
// 			cstoneOctree::Vec3f norms(0, 0, 0);
// 			float area = 0.0f;
// 			std::vector<int> insides;
// 			min_radiuses[i] = IS_CONST_RADIUS ? _GlobalRadiuses[i] : FLT_MAX;
// 			int mortonCodesSize = _mortonCodes.size();
// 			uint64_t nodeStartVal = iso_tree[i];
// 			uint64_t nodeEndVal = iso_tree[i + 1];
// 			// here, a better method could be estimate neighbors size by counting leaves which covered by box.
// 			// however, i dont know how to get leaf info in searching tree
// 			// so, TODO()
// 			auto rangeStart = cal::lower_bound(_mortonCodes.data(), _mortonCodes.data() + mortonCodesSize, nodeStartVal);
// 			auto rangeEnd = cal::lower_bound(_mortonCodes.data(), _mortonCodes.data() + mortonCodesSize, nodeEndVal);
// 			int index_start = std::distance(_mortonCodes.data(), rangeStart);
// 			int index_end = std::distance(_mortonCodes.data(), rangeEnd);
// 			for(int index = index_start; index < index_end; index++){
// 				insideParticlesIdx.push_back(index);
// 			}
// 			emptys[i] = insideParticlesIdx.empty();
// 			curvs[i] = (area == 0) ? 1.0 : (norms.norm() / area);
// 			if(emptys[i]){
// 				scalars[i] = _evaluator->getIsoValue();
// 				nodes_type[i] = 0;
// 			}
// 		}
// 		//
		
// 		//
// 		uint64_t iso_allOpsSum = std::accumulate(iso_nodeOps.begin(), iso_nodeOps.end(), 0);
// 		// exclusive_csan ops to get new octree indices
// 		std::vector<uint64_t> iso_nodeOpsLayout(iso_nodeOps.size());
// 		for(int i = 0; i < iso_nodeOps.size(); i++){
// 			int val = iso_nodeOps[i] - 1;
// 			if(val != 0 && val != 7){
// 				std::cout << "error in iso_nodeOps, " << i << "th value is " << val << std::endl;
// 			}
// 		}
// 		std::exclusive_scan(iso_nodeOps.begin(), iso_nodeOps.end(), iso_nodeOpsLayout.begin(), 0);
// 		uint64_t newTreeNodesNum;	
// 		newTreeNodesNum = iso_nodeOpsLayout[iso_tree.size() - 1];
// 		std::vector<uint64_t> new_iso_tree(newTreeNodesNum + 1);  // updated tree array

// 		updateTreeArrayCPU(iso_nodeOpsLayout, iso_tree, new_iso_tree);

// 		std::copy_n(iso_tree.data() + iso_tree.size() - 1, 1, new_iso_tree.data() + new_iso_tree.size() - 1);
// 		std::swap(new_iso_tree, iso_tree);

// 		iso_count++;
// 		if(iso_allOpsSum == iso_nodeOps.size() - 1) break;
// 	} // end octree generation loop

// 	// //  ---  link iso octree (tree node and internal node) ---
// 	// int iso_numLeafNodes = iso_tree.size() - 1; // tree size minus 1 because the range end variable is added before.
//     // int iso_numInternalNodes = (iso_numLeafNodes - 1) / 7; // number of nodes which is not leaf nodes
//     // int iso_numNodes         = iso_numLeafNodes + iso_numInternalNodes; // total number of nodes (leaf nodes + internal nodes)
// 	// std::vector<uint64_t> iso_prefixes;  // used to delete the righthand side 0 bits for each morton code
//     // std::vector<TreeNodeIndex> iso_internalToLeaf;
//     // std::vector<TreeNodeIndex> iso_leafToInternal;
//     // std::vector<TreeNodeIndex> iso_childOffsets;
// 	// iso_prefixes.resize(iso_numNodes);
// 	// iso_internalToLeaf.resize(iso_numNodes);
//     // iso_leafToInternal.resize(iso_numNodes);
//     // iso_childOffsets.resize(iso_numNodes + 1);
//     // std::vector<TreeNodeIndex> iso_parents;
//     // std::vector<int> iso_levelRange;
//     // iso_parents.resize(std::max(1, (iso_numNodes - 1) / 8));
//     // iso_levelRange.resize(21 + 2);

// 	// createUnsortedLayoutCPU(iso_tree, iso_numInternalNodes, iso_numLeafNodes, iso_prefixes, iso_internalToLeaf);	
// 	// cal::sort_by_key_cpu(iso_prefixes.data(), iso_prefixes.data() + iso_prefixes.size(), iso_internalToLeaf.data());

// 	// invertOrderCPU(iso_internalToLeaf, iso_leafToInternal, iso_numNodes, iso_numInternalNodes);
// 	// // Calculate node range for each tree level
// 	// getLevelRangeCPU(iso_prefixes, iso_numNodes, iso_levelRange);
// 	// cal::fill_data_cpu(iso_childOffsets.data(), iso_numNodes + 1, 0);

// 	// linkOctreeCPU(iso_prefixes,
// 	// 			iso_numInternalNodes,
// 	// 			iso_leafToInternal,
// 	// 			iso_levelRange,
// 	// 			iso_childOffsets,
// 	// 			iso_parents);
	
// 	// std::vector<Vec3f> iso_centers(iso_numNodes);
// 	// std::vector<Vec3f> iso_sizes(iso_numNodes);
// 	// calculateNodeCentersAndSizesCPU(iso_prefixes, iso_centers, iso_sizes, box);
// 	// std::cout << "iso centers[0]: " << iso_centers[0].x << ", " << iso_centers[0].y << ", " << iso_centers[0].z << std::endl;
// 	// std::cout << "iso size[0]: " << iso_sizes[0].x << ", " << iso_sizes[0].y << ", " << iso_sizes[0].z << std::endl;

//     // std::vector<int> iso_layout(iso_numLeafNodes + 1); // particle layout. This is not used in iso octree
// 	// IsoOctreeNs iso_octreeNs(iso_prefixes.data(), iso_childOffsets.data(), iso_leafToInternal.data(), iso_internalToLeaf.data(), iso_levelRange.data(), scalars.data(), iso_centers.data(), iso_sizes.data());

// 	// extract the isosurface value
	
// }

void SurfReconstructor::RunCPU2(float iso_factor, float smooth_factor)
{
	timer t;

	std::cout << "-= Box =-" << std::endl;
	loadRootBox();

	std::cout << "-= Build Neighbor Searcher =-" << std::endl;
	// if (IS_CONST_RADIUS)
	// {
	// IS_CONST_RADIUS = true;
    // 	HashGrid _hashgrid(&_GlobalParticles, _BoundingBox, _GlobalRadiuses[0], 4.0f);
	// 	IS_CONST_RADIUS = false;
		// } else {
		// useCPU = true;
		// _searcherCPU = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
		useCPU = false;
		_searcherGPU = std::make_shared<MultiLevelSearcherGPU>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
	// }
	std::cout << "   Build Neighbor Searcher Time = " << t.elapsed() << std::endl;
	t.reset();

    printf("-= Initialize Evaluator =-\n");
	if (useCPU)
	{
		_evaluator = std::make_shared<Evaluator>(_searcherCPU, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
	} else {
		_evaluator = std::make_shared<Evaluator>(_searcherGPU, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
	}
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

	IS_CONST_RADIUS ? _evaluator->CalculateMaxScalarConstR() : _evaluator->CalculateMaxScalarVarR();
    printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
	IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR() : _evaluator->RecommendIsoValueVarR();
    printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());
	if (CALC_P_NORMAL)
	{
		_evaluator->CalcParticlesNormal();
		printf("   Calculate Particals Normal Time = %f\n", t.elapsed());
		t.reset();
	}

	// ----------- generating iso surface octree ---------
	std::vector<uint64_t> iso_tree;
	// iso_tree.assign(_tree.begin(), _tree.end());
	iso_tree.resize(1 + 1);
	cal::fill_data_cpu(iso_tree.data(), 1, 0);
	cal::fill_data_cpu(iso_tree.data() + 1, 1, uint64_t(1) << 63);

	int iso_count = 0;
	std::vector<float> scalars; // used to store each leaf nodes' scalar value on dual vertices. important for isosurface generation

	cstoneOctree::Box box(_BoundingBox[0], _BoundingBox[1], _BoundingBox[2], 
						  _BoundingBox[3], _BoundingBox[4], _BoundingBox[5]);
	
	while(1){
		// ---- iso octree's info calculation ----
		int iso_tree_size = iso_tree.size() - 1;
		std::cout << "tree size in loop " << iso_count << ": " << iso_tree_size << std::endl;
		std::vector<uint64_t> iso_prefixes(iso_tree_size);

		// calculate prefixes for each node
		// for(int i = 0; i < iso_tree_size; i++){
		// 	uint64_t curr_key = iso_tree[i];
		// 	if(!isPowerOf8(iso_tree[i + 1] - curr_key)) printf("The index %d is not the power of 8\n", i);
		// 	unsigned curr_level = treeLevel(iso_tree[i + 1] - curr_key);
		// 	iso_prefixes[i] = encodePlaceholderBit(curr_key, 3 * curr_level);	
		// }
		std::vector<Vec3f> iso_centers(iso_tree_size);
		std::vector<Vec3f> iso_sizes(iso_tree_size);
		std::vector<unsigned> iso_depths(iso_tree_size);
		calculateLeavesCentersAndSizesCPU(iso_tree, iso_centers, iso_sizes, iso_depths, box);

		// ---- iso surface split decision making ----
		// std::vector<float> sample_points(int(std::pow(getOverSampleQEF() + 1, 3) * 4 * iso_tree_size));
		// std::vector<float> sample_grads(int(std::pow(getOverSampleQEF() + 1, 3) * 4 * iso_tree_size));
		std::vector<float> curvs(iso_tree_size);
		std::vector<float> min_radiuses(iso_tree_size);	
		std::vector<unsigned char> emptys(iso_tree_size, 0);
		// std::vector<float> iso_values(iso_tree_size);
		// std::vector<int> nodes_type(iso_tree_size, 1);
		scalars = std::vector<float>(iso_tree_size, 0.0); // Stores each tree nodes' scalar value on dual vertices
		std::vector<uint64_t> iso_nodeOps(iso_tree.size(), 0); // Store the split decision for each node (the split decision is based on whether the node has isosurface)
					// bool
		#pragma omp parallel for
		for(int i = 0; i < iso_tree_size; i++) {
			// begin beforeSampleEval
			Vec3f center = iso_centers[i];
			Vec3f box1 = center - Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
			Vec3f box2 = center + Vec3f(iso_sizes[i].x, iso_sizes[i].y, iso_sizes[i].z);
			int estimateNeighborsNum = 0;
			int trueNeighborsNum = 0;
			std::vector<int> insideParticlesIdx;
			// if (IS_CONST_RADIUS)
			// {
			// 	_hashgrid->GetInBoxEstimate(box1, box2, estimateNeighborsNum);
			// 	insideParticlesIdx.resize(estimateNeighborsNum);
			// 	_hashgrid->GetInBoxParticles(box1, box2, trueNeighborsNum, estimateNeighborsNum, insideParticlesIdx.data());
			// } else {
			if (useCPU)
			{
				_searcherCPU->GetInBoxParticles(box1, box2, insideParticlesIdx);
				trueNeighborsNum = insideParticlesIdx.size();
			} else {
				_searcherGPU->GetInBoxEstimate(box1, box2, estimateNeighborsNum);
				insideParticlesIdx.resize(estimateNeighborsNum);
				_searcherGPU->GetInBoxParticles(box1, box2, trueNeighborsNum, estimateNeighborsNum, insideParticlesIdx.data());
			}
			// }
			// check empty and calculate curvature implentation below
			cstoneOctree::Vec3f norms(0, 0, 0);
			float area = 0.0f;
			min_radiuses[i] = FLT_MAX;
			emptys[i] = trueNeighborsNum == 0;
			if (!emptys[i])
			{
				bool allSplash = true;
				for (int j = 0; j < trueNeighborsNum; j++) {
					int in = insideParticlesIdx[j];
					if (!_evaluator->CheckSplash(in))
					{
						if (_GlobalParticles[in].x > (box1.x - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
							_GlobalParticles[in].x < (box2.x + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
							_GlobalParticles[in].y > (box1.y - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
							_GlobalParticles[in].y < (box2.y + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
							_GlobalParticles[in].z > (box1.z - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
							_GlobalParticles[in].z < (box2.z + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())))
						{
							if (CALC_P_NORMAL)
							{
								cstoneOctree::Vec3f tempNorm = _evaluator->PariclesNormals[in];
								norms += tempNorm;
								area += tempNorm.norm();
							}
							if (!IS_CONST_RADIUS)
							{
								if (min_radiuses[i] > _GlobalRadiuses[in])
								{
									min_radiuses[i] = _GlobalRadiuses[in];
								}
							}
							allSplash = false;
						}
					}
				}
				emptys[i] = allSplash;
			}
			curvs[i] = (area == 0) ? 1.0 : (norms.norm() / area);
			if (emptys[i]) {
				scalars[i] = _evaluator->getIsoValue();
				// nodes_type[i] = 0;
				iso_nodeOps[i] = 1;
				continue;
			}
			bool isbig = (iso_depths[i] < _DEPTH_MIN);
			if (isbig)
			{
				iso_nodeOps[i] = 8;
				continue;
			}
			bool signchange = false;
			std::vector<float> nodeSamplePoints(pow(2 + 1, 3) * 3);
			std::vector<float> nodeSampleScalars(pow(2 + 1, 3), 0);
			// std::vector<float> nodeSampleGrads(pow(2+1, 3) * 3);
			for (float z = 0; z <= 2; z++)
			{
				for (float y = 0; y <= 2; y++)
				{
					for (float x = 0; x <= 2; x++)
					{
						nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 0] = 
						(1 - x / 2) * box1[0] + (x / 2) * box2[0];
						nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 1] = 
						(1 - y / 2) * box1[1] + (y / 2) * box2[1];
						nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 2] = 
						(1 - z / 2) * box1[2] + (z / 2) * box2[2];
					}
				}
			}
			// grid sampling
			bool origin_sign;
			float cellSize = iso_sizes[i][0] * 2;
			float step = cellSize / 2;
			for (int j = 0; j < pow(2+1, 3); j++)
			{
				Vec3f diff;
				Vec3f samplePoint(nodeSamplePoints[j * 3 + 0], nodeSamplePoints[j * 3 + 1], nodeSamplePoints[j * 3 + 2]);
				for (int k = 0; k < trueNeighborsNum; k++)
				{
					int pIdx = insideParticlesIdx[k];
					if (_evaluator->CheckSplash(pIdx))
					{
						continue;
					}
					diff = samplePoint - _evaluator->GlobalxMeans[pIdx];
					nodeSampleScalars[j] += _evaluator->AnisotropicInterpolate(pIdx, diff);
					// if (USE_ANI)
					// {
					// }
					// else {
					// 	diff = pos - (*GlobalPoses)[pIdx];
					// 	scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
					// }
				}
				nodeSampleScalars[j] = _evaluator->getIsoValue() - nodeSampleScalars[j];
				origin_sign = (nodeSampleScalars[0] >= 0);
				if (!signchange)
				{
					signchange = origin_sign ^ (nodeSampleScalars[j] >= 0);
				}
				// int index, next_idx, last_idx;
				// for (int z = 0; z <= 2; z++)
				// {
				// 	for (int y = 0; y <= 2; y++)
				// 	{
				// 		for (int x = 0; x <= 2; x++)
				// 		{
				// 			index = (z * (2 + 1) * (2 + 1) + y * (2 + 1) + x);
				// 			Vec3f gradient(0.0f, 0.0f, 0.0f);
				// 			next_idx = (z * (2 + 1) * (2 + 1) + y * (2 + 1) + (x + 1));
				// 			last_idx = (z * (2 + 1) * (2 + 1) + y * (2 + 1) + (x - 1));
				// 			if (x == 0)
				// 			{
				// 				gradient[0] = (nodeSampleScalars[index] - nodeSampleScalars[next_idx]) / step;
				// 			}
				// 			else if (x == 2)
				// 			{
				// 				gradient[0] = (nodeSampleScalars[last_idx] - nodeSampleScalars[index]) / step;
				// 			}
				// 			else
				// 			{
				// 				gradient[0] = (nodeSampleScalars[last_idx] - nodeSampleScalars[next_idx]) / (step * 2);
				// 			}
				// 			next_idx = (z * (2 + 1) * (2 + 1) + (y + 1) * (2 + 1) + x);
				// 			last_idx = (z * (2 + 1) * (2 + 1) + (y - 1) * (2 + 1) + x);
				// 			if (y == 0)
				// 			{
				// 				gradient[1] = (nodeSampleScalars[index] - nodeSampleScalars[next_idx]) / step;
				// 			}
				// 			else if (y == 2)
				// 			{
				// 				gradient[1] = (nodeSampleScalars[last_idx] - nodeSampleScalars[index]) / step;
				// 			}
				// 			else
				// 			{
				// 				gradient[1] = (nodeSampleScalars[last_idx] - nodeSampleScalars[next_idx]) / (step * 2);
				// 			}
				// 			next_idx = ((z + 1) * (2 + 1) * (2 + 1) + y * (2 + 1) + x);
				// 			last_idx = ((z - 1) * (2 + 1) * (2 + 1) + y * (2 + 1) + x);
				// 			if (z == 0)
				// 			{
				// 				gradient[2] = (nodeSampleScalars[index] - nodeSampleScalars[next_idx]) / step;
				// 			}
				// 			else if (z == 2)
				// 			{
				// 				gradient[2] = (nodeSampleScalars[last_idx] - nodeSampleScalars[index]) / step;
				// 			}
				// 			else
				// 			{
				// 				gradient[2] = (nodeSampleScalars[last_idx] - nodeSampleScalars[next_idx]) / (step * 2);
				// 			}
				// 			gradient.normalize();
				// 			nodeSampleGrads[index * 3 + 0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
				// 			nodeSampleGrads[index * 3 + 1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
				// 			nodeSampleGrads[index * 3 + 2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
				// 		}
				// 	}
				// }
			}
			// after process
			if ((cellSize - min_radiuses[i]) < 1e-8)
			{
				// it's a leaf
				// nodes_type[i] = LEAF;
				iso_nodeOps[i] = 1;
				scalars[i] = nodeSampleScalars[13];
				continue;
			}
			// check curvature
			if (isbig || (signchange && curvs[i] < 0.995))//
			{
				// tnode->type = INTERNAL;
				iso_nodeOps[i] = 8;
				continue;
			}
			else
			{
				// tnode->type = LEAF;
				iso_nodeOps[i] = 1;
				scalars[i] = nodeSampleScalars[13];
				// _evaluator->SingleEval(tnode->center, tnode->nodeScalar);
				continue;
			}
		}
		uint64_t iso_allOpsSum = std::accumulate(iso_nodeOps.begin(), iso_nodeOps.end(), 0);
		// exclusive_csan ops to get new octree indices
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
	} // end octree generation loop

	std::vector<Vec3i> iso_lowers(iso_tree.size() - 1);
	std::vector<unsigned> iso_levels(iso_tree.size() - 1);
	calculateLeavesLowersAndLevelsCPU(iso_tree, iso_lowers, iso_levels, box);

	iso::generateIso(iso_tree, iso_lowers, iso_levels, scalars, 0.0, 
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
	// printf("Time generating polygons = %f\n", t_gen_mesh.elapsed());
	std::cout << "Time generating polygons;" << std::endl;
}

void SurfReconstructor::Run(float iso_factor, float smooth_factor)
{
	printf("-= Run =-\n");
	_OurRoot = nullptr;
    _OurMesh->reset();
    _STATE = 0;

	timer t;

	printf("-= Box =-\n");
	loadRootBox();

	printf("-= Build Neighbor Searcher =-\n");
	// if (IS_CONST_RADIUS)
	// {
    // 	_hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 4.0f);
	// } else {
		_searcherCPU = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
	// }
	printf("   Build Neighbor Searcher Time = %f \n", t.elapsed());
	t.reset();

    printf("-= Initialize Evaluator =-\n");
	_evaluator = std::make_shared<Evaluator>(_searcherCPU, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
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

	IS_CONST_RADIUS ? _evaluator->CalculateMaxScalarConstR() : _evaluator->CalculateMaxScalarVarR();
    printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
	IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR() : _evaluator->RecommendIsoValueVarR();
    printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());

	if (CALC_P_NORMAL)
	{
		_evaluator->CalcParticlesNormal();
		printf("   Calculate Particals Normal Time = %f\n",t.elapsed());
		t.reset();
	}

	// printMem();
	genIsoOurs();
	// printMem();
	genIsoOurs();

	printf("-=  Total time= %f  =-\n", t.elapsed());
}
