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

void SurfReconstructor::loadRootBox(float r)
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
	for (size_t i = 0; i < 3; i++)
	{
		_BoundingBox[i * 2] -= r;
		_BoundingBox[i * 2 + 1] += r;
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
		  maxR = useCPU ? _searcherCPU->getMaxRadius() : _searcherGPU->getMaxRadius();
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
	// _DEPTH_MIN = (_DEPTH_MAX - (SINGLE_LAYER ? 1 : 2));
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
		#pragma omp parallel for
				for (int i = 0; i < queue_flag; i++)
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
		#pragma omp parallel for
				for (int i = 0; i < queue_flag; i++)
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

void SurfReconstructor::RunCPU2(float iso_factor, float smooth_factor)
{
	timer t;

	std::cout << "-= Box =-" << std::endl;
	loadRootBox(*std::max_element(_GlobalRadiuses.begin(), _GlobalRadiuses.end()));

	std::cout << "-= Build Neighbor Searcher =-" << std::endl;
	// if (IS_CONST_RADIUS)
	// {
	// IS_CONST_RADIUS = true;
    // 	HashGrid _hashgrid(&_GlobalParticles, _BoundingBox, _GlobalRadiuses[0], 4.0f);
	// 	IS_CONST_RADIUS = false;
		// } else {
		useCPU = true;
		_searcherCPU = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
		// useCPU = false;
		// _searcherGPU = std::make_shared<MultiLevelSearcherGPU>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
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
		// #pragma omp parallel for
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
				}
				nodeSampleScalars[j] = _evaluator->getIsoValue() - nodeSampleScalars[j];
				origin_sign = (nodeSampleScalars[0] >= 0);
				if (!signchange)
				{
					signchange = origin_sign ^ (nodeSampleScalars[j] >= 0);
				}
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
	calculateLeavesLowersAndLevelsCPU(iso_tree, iso_lowers, iso_levels);

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
	loadRootBox(*std::max_element(_GlobalRadiuses.begin(), _GlobalRadiuses.end()));

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
