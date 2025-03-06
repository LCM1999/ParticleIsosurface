#include <queue>
#include <omp.h>

#include "surface_reconstructor.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include "evaluator.h"
#include "iso_method_ours.h"
#include "global.h"
#include "visitorextract.h"
#include "traverse.h"
#include "timer.h"
#include <var.h>

SurfReconstructor::SurfReconstructor(
	std::vector<Eigen::Vector3f>& particles, 
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

inline void SurfReconstructor::loadRootBox()
{
	_BoundingBox[0] = _BoundingBox[2] = _BoundingBox[4] = FLT_MAX;
	_BoundingBox[1] = _BoundingBox[3] = _BoundingBox[5] = -FLT_MAX;
	for (const Eigen::Vector3f& p: _GlobalParticles)
	{
		if (p.x() < _BoundingBox[0]) _BoundingBox[0] = p.x();
		if (p.x() > _BoundingBox[1]) _BoundingBox[1] = p.x();
		if (p.y() < _BoundingBox[2]) _BoundingBox[2] = p.y();
		if (p.y() > _BoundingBox[3]) _BoundingBox[3] = p.y();
		if (p.z() < _BoundingBox[4]) _BoundingBox[4] = p.z();
		if (p.z() > _BoundingBox[5]) _BoundingBox[5] = p.z();
	}
	if (_BoundingBox[0] == _BoundingBox[1] ||
		_BoundingBox[2] == _BoundingBox[3] ||
		_BoundingBox[4] == _BoundingBox[5])
	{
		SINGLE_LAYER = true;
	}
}

void SurfReconstructor::shrinkBox()
{
	std::vector<int> ids;
	ids.resize(_GlobalParticlesNum);
	for (size_t i = 0; i < _GlobalParticlesNum; i++)
	{
		ids[i] = i;
	}
	_BoundingBox[0] = _GlobalParticles[(*std::min_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id1)) {	return false;	}
		if (getEvaluator()->CheckSplash(id2)) {	return true;	}
		return getGlobalParticles()->at(id1).x() < getGlobalParticles()->at(id2).x();
		}))].x();
	_BoundingBox[1] = _GlobalParticles[(*std::max_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id2)) {	return false;	}
		if (getEvaluator()->CheckSplash(id1)) {	return true;	}
		return getGlobalParticles()->at(id1).x() < getGlobalParticles()->at(id2).x();
		}))].x();
	_BoundingBox[2] = _GlobalParticles[(*std::min_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id1)) {	return false;	}
		if (getEvaluator()->CheckSplash(id2)) {	return true;	}
		return getGlobalParticles()->at(id1).y() < getGlobalParticles()->at(id2).y();
		}))].y();
	_BoundingBox[3] = _GlobalParticles[(*std::max_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id2)) {	return false;	}
		if (getEvaluator()->CheckSplash(id1)) {	return true;	}
		return getGlobalParticles()->at(id1).y() < getGlobalParticles()->at(id2).y();
		}))].y();
	_BoundingBox[4] = _GlobalParticles[(*std::min_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id1)) {	return false;	}
		if (getEvaluator()->CheckSplash(id2)) {	return true;	}
		return getGlobalParticles()->at(id1).z() < getGlobalParticles()->at(id2).z();
		}))].z();
	_BoundingBox[5] = _GlobalParticles[(*std::max_element(ids.begin(), ids.end(), 
	[&] (const int& id1, const int& id2) {
		if (getEvaluator()->CheckSplash(id2)) {	return false;	}
		if (getEvaluator()->CheckSplash(id1)) {	return true;	}
		return getGlobalParticles()->at(id1).z() < getGlobalParticles()->at(id2).z();
		}))].z();
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
	float minR = _searcher->getMinRadius(), maxR = _searcher->getMaxRadius(), avgR = _searcher->getAvgRadius();
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
	Eigen::Vector3f norms(0, 0, 0);
	float area = 0.0f;
	// int impact_num = 0;
	std::vector<int> insides;
	min_radius = IS_CONST_RADIUS ? _RADIUS : FLT_MAX;
	const Eigen::Vector3f 
	box1 = tnode->center - Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length),
	box2 = tnode->center + Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length);
	if (IS_CONST_RADIUS)
	{
		_hashgrid->GetInBoxParticles(box1, box2, insides);
	} else {
		_searcher->GetInBoxParticles(box1, box2, insides);
	}
	empty = insides.empty();
	if (!empty)
	{
		bool all_splash = true;
		for (const int& in: insides)
		{
			if (!_evaluator->CheckSplash(in))
			{
				if (_GlobalParticles[in].x() > (box1.x() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].x() < (box2.x() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
					_GlobalParticles[in].y() > (box1.y() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].y() < (box2.y() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) &&
					_GlobalParticles[in].z() > (box1.z() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())) && 
					_GlobalParticles[in].z() < (box2.z() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses[in]) * _evaluator->getSmoothFactor())))
				{
					if (CALC_P_NORMAL)
					{
						Eigen::Vector3f tempNorm = _evaluator->PariclesNormals[in];
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
		tnode->node[3] = _evaluator->getIsoValue();
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
		_evaluator->SingleEval(tnode->node.head(3), tnode->node[3]);
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
		_evaluator->SingleEval(tnode->node.head(3), tnode->node[3]);
	}
}

void SurfReconstructor::genIsoOurs()
{
    float t_start = get_time();

	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	if (_STATE == 0)
	{
		printf("-= Calculating Tree Structure =-\n");
		_OurRoot = std::make_shared<TNode>(this, 0);
		_OurRoot->center << _RootCenter[0], _RootCenter[1], _RootCenter[2];
		_OurRoot->node << _RootCenter[0], _RootCenter[1], _RootCenter[2], 0.0;
		_OurRoot->half_length = _RootHalfLength;
	} else if (_STATE == 1) {
		printf("-= Generate Surface =-\n");
		float t_gen_mesh = get_time();
		_OurMesh->tris.reserve(1000000);
		VisitorExtract v(this, _OurMesh);
		TraversalData td(_OurRoot);
		traverse_node<trav_vert>(v, td);
		v.calc_vertices();
		v.generate_mesh();
		if (GEN_SPLASH)
		{
			printf("-= Generate Splash =-\n");
			std::vector<Eigen::Vector3f> splash_pos;
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
		float t_alldone = get_time();
		printf("Time generating polygons = %f\n", t_alldone - t_gen_mesh);
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
		#pragma omp parallel
		{
			#pragma omp single
			{
				for (size_t i = 0; i < queue_flag; i++)
				{
					#pragma omp task
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
		#pragma omp parallel
		{
			#pragma omp single
			{
				for (size_t i = 0; i < queue_flag; i++)
				{
					if (!emptys[i])
					{
						#pragma omp task
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
		}
		// depth++;
		// half/=2;
	}
	ProcessArray.clear();
	delete[] sample_points;
	delete[] sample_grads;
	float t_finish = get_time();
	printf("Time generating tree = %f\n", t_finish - t_start);	
	_STATE++;
}

void SurfReconstructor::Run(float iso_factor, float smooth_factor)
{
	printf("-= Run =-\n");
	_OurRoot = nullptr;
    _OurMesh->reset();
    _STATE = 0;

    float time_all_start = get_time();
	float temp_time, last_temp_time;

	printf("-= Box =-\n");
	loadRootBox();

	temp_time = get_time();

	printf("-= Build Neighbor Searcher =-\n");
	if (IS_CONST_RADIUS)
	{
    	_hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 4.0f);
	} else {
		_searcher = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
	}
    last_temp_time = temp_time;
    temp_time = get_time();
	printf("   Build Neighbor Searcher Time = %f \n", temp_time - last_temp_time);

    printf("-= Initialize Evaluator =-\n");
	_evaluator = std::make_shared<Evaluator>(_hashgrid, _searcher, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
	_evaluator->setSmoothFactor(smooth_factor);
	_evaluator->setIsoFactor(iso_factor);
	_evaluator->compute_Gs_xMeans();
	// if (USE_POLY6)
	// {
	// 	if (IS_CONST_RADIUS)
	// 	{
	// 		_hashgrid.reset();
	// 		_hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 2.0f);
	// 		_evaluator->_hashgrid = _hashgrid;
	// 	} else {
	// 		_searcher.reset();
	// 		_searcher = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 2.0f);
	// 		_evaluator->_searcher = _searcher;
	// 	}
	// }
	last_temp_time = temp_time;
	temp_time = get_time();
	printf("   Initialize Evaluator Time = %f \n", temp_time - last_temp_time);
	
	printf("-= Resize Box =-\n");
	// shrinkBox();
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
	// _evaluator->setIsoValue(0.4033877702884984e+30);
	temp_time = get_time();
	if (CALC_P_NORMAL)
	{
		_evaluator->CalcParticlesNormal();
		last_temp_time = temp_time;
		temp_time = get_time();
		printf("   Calculate Particals Normal Time = %f\n", temp_time - last_temp_time);
	}

	// printMem();
	genIsoOurs();
	// printMem();
	genIsoOurs();

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);
}


