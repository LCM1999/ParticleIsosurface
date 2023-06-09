#include <queue>
#include <omp.h>
//#include <metis.h>

#include "surface_reconstructor.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include "evaluator.h"
#include "iso_method_ours.h"
#include "global.h"
#include "visitorextract.h"
#include "traverse.h"

extern "C" void cuda_node_calc_initialize_const_r(
    int GlobalParticlesNum, int DepthMin, float R, float InfFactor, float IsoValue, float MaxScalar,
    Evaluator* evaluator, HashGrid* hashgrid
);

extern "C" void cuda_node_calc_release_const_r();

extern "C" void cuda_node_calc_const_r_kernel(
    int QueueFlag, float half_length, char* types_cpu, float* centers_cpu, float* nodes_cpu
);

SurfReconstructor::SurfReconstructor(std::vector<Eigen::Vector3f>& particles, 
std::vector<float>* radiuses, Mesh* mesh, 
float radius, float flatness, float inf_factor)
{
	_GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalRadiuses = radiuses;
	_RADIUS = radius;
	_NEIGHBOR_FACTOR = inf_factor;

	WaitingStack.clear();

	queue_flag = 0;

	_OurMesh = mesh;
}

inline void SurfReconstructor::loadRootBox()
{
	_BoundingBox[0] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.x() < b.x(); })).x();
	_BoundingBox[1] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.x() < b.x(); })).x();
	_BoundingBox[2] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.y() < b.y(); })).y();
	_BoundingBox[3] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.y() < b.y(); })).y();
	_BoundingBox[4] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.z() < b.z(); })).z();
	_BoundingBox[5] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	[&] (Eigen::Vector3f& a, Eigen::Vector3f& b) { return a.z() < b.z(); })).z();
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
	while (resizeLen - maxLen < (_NEIGHBOR_FACTOR * _RADIUS))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * r;
	}
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}
	
	_DEPTH_MIN = (_DEPTH_MAX - 2);	//int(_DEPTH_MAX / 3));
}

void SurfReconstructor::resizeRootBoxVarR()
{
	double maxLen, resizeLen;
	float minR = _searcher->getMinRadius(), maxR = _searcher->getMaxRadius(), avgR = _searcher->getAvgRadius();
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / minR))));
	resizeLen = pow(2, _DEPTH_MAX) * minR;
	while (resizeLen - maxLen < (_NEIGHBOR_FACTOR * maxR))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * avgR;
	}
	// resizeLen *= 1.005;
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}

	_DEPTH_MIN = int(ceil(log2(ceil(maxLen / maxR)))) - 2; //, _DEPTH_MAX - int(_DEPTH_MAX / 3));
}


void SurfReconstructor::checkEmptyAndCalcCurv(TNode* tnode, bool& empty, float& curv, float& min_radius)
{
	Eigen::Vector3f norms(0, 0, 0);
	int area = 0;
	std::vector<int> insides;
	// double node_vol = std::max(pow(tnode->half_length * 2, 3), pow(_hashgrid->CellSize, 3));
	// double p_vol = 0.0f;
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
				if (_GlobalParticles[in].x() > (box1.x() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)) && 
					_GlobalParticles[in].x() < (box2.x() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)) &&
					_GlobalParticles[in].y() > (box1.y() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)) && 
					_GlobalParticles[in].y() < (box2.y() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)) &&
					_GlobalParticles[in].z() > (box1.z() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)) && 
					_GlobalParticles[in].z() < (box2.z() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * _SMOOTH_FACTOR)))
				{
					Eigen::Vector3f tempNorm = _evaluator->PariclesNormals[in];
					if (tempNorm == Eigen::Vector3f(0, 0, 0))	{continue;}
					norms += tempNorm;
					area++;
					if (!IS_CONST_RADIUS)
					{
						if (min_radius > _GlobalRadiuses->at(in))
						{
							min_radius = _GlobalRadiuses->at(in);
						}
					}
					all_splash = false;
				}
			}
		}
		empty = all_splash;
	}
	curv = (area == 0) ? 1.0 : (norms.norm() / area);
	// printf("%f, %d, %f\n", norms.norm(), area, curv);
}

void SurfReconstructor::beforeSampleEval(TNode* tnode, float& curv, float& min_radius, bool& empty)
{
	switch (tnode->type)
	{
	case EMPTY:
	case LEAF:
	case INTERNAL:
		return;
	case UNCERTAIN:
	{
		// evaluate QEF samples
		// bool full = false;
		checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
		if (empty)
		{
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3]);
			tnode->type = EMPTY;
			return;
		}
	}
	break;
	default:
		break;
	}
}

void SurfReconstructor::afterSampleEval(
	TNode* tnode, float& curv, float& min_radius,
	float* sample_points, float* sample_grads)
{
	// double generate_time = 0, sampling_time = 0, calc_time = 0;
	bool isbig = (tnode->depth < _DEPTH_MIN);
	bool signchange = false;
	// tnode->vertAll(curv, signchange, qef_error, min_radius);
	float cellsize = 2 * tnode->half_length;

	if (!isbig)
	{
		tnode->GenerateSampling(sample_points);
		tnode->NodeSampling(curv, signchange, cellsize, sample_points, sample_grads);
	}
	
	// judge this node need calculate iso-surface

	// check max/min sizes of cells
	if ((cellsize - min_radius) < _TOLERANCE)
	{
		// it's a leaf
		tnode->type = LEAF;
		// tnode->NodeCalcNode(sample_points, sample_grads);
		_evaluator->SingleEval(tnode->node.head(3), tnode->node[3]);
		return;
	}
	// check curvature
	if (isbig || (signchange && curv < _FLATNESS))
	{
		tnode->type = INTERNAL;
	}
	else
	{
		tnode->type = LEAF;
		// tnode->NodeCalcNode(sample_points, sample_grads);
		_evaluator->SingleEval(tnode->node.head(3), tnode->node[3]);
	}
}

void SurfReconstructor::genIsoOurs()
{
    double t_start = get_time();

	TNode* root;
	Mesh* m = _OurMesh;
	TNode* loaded_tree = nullptr;

	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	if (_STATE == 0)
	{
		printf("-= Calculating Tree Structure =-\n");
		root = new TNode(this);
		root->center << _RootCenter[0], _RootCenter[1], _RootCenter[2];
		root->node << _RootCenter[0], _RootCenter[1], _RootCenter[2], 0.0;
		root->half_length = _RootHalfLength;
		_OurRoot = root;
	}
	// else if (_STATE == 1)
	// {
	// 	printf("-= Our Method =-\n");
	// 	root = _OurRoot;
	// }
	else if (_STATE == 1)
	{
		printf("-= Generate Surface =-\n");
		double t_gen_mesh = get_time();
		m->tris.reserve(1000000);
		VisitorExtract v(this, m);
		TraversalData td(_OurRoot);
		traverse_node<trav_vert>(v, td);
		std::vector<Eigen::Vector3f> splash_pos;
		std::vector<float> splash_radiuses;
		for (int pIdx = 0; pIdx < getGlobalParticlesNum(); pIdx++)
		{
			if (_evaluator->CheckSplash(pIdx))
			{
				splash_pos.push_back(_GlobalParticles[pIdx]);
				if (!IS_CONST_RADIUS)
				{
				 	splash_radiuses.push_back(_GlobalRadiuses->at(pIdx));
				}
			}
		}
		if (IS_CONST_RADIUS)
		{
			m->AppendSplash_ConstR(splash_pos, _RADIUS);
		} else {
			m->AppendSplash_VarR(splash_pos, splash_radiuses);
		}
		
		//}
		double t_alldone = get_time();
		printf("Time generating polygons = %f\n", t_alldone - t_gen_mesh);
		return;
	}
	int depth = 0;
	float half = root->half_length;
	double layer_time, phase1, phase2, phase3;
	float* sample_points;
	float* sample_grads;
	float* cuvrs;
	float* min_raiduses;
	bool* emptys;
	char* types_cpu;
	float* centers_cpu;
	float* nodes_cpu;
	WaitingStack.push_back(root);
	while (!WaitingStack.empty()) {
		printf("%d \n", WaitingStack.size());
		for (queue_flag = 0; queue_flag < inProcessSize && !WaitingStack.empty(); queue_flag++)
		{
			ProcessArray[queue_flag] = WaitingStack.back();
			WaitingStack.pop_back();
		}
		if (USE_CUDA && IS_CONST_RADIUS && depth >= _DEPTH_MIN)
		{
			types_cpu = new char[queue_flag];
			centers_cpu = new float[queue_flag * 3];
			nodes_cpu = new float[queue_flag * 4];
			for (size_t i = 0; i < queue_flag; i++)
			{
				types_cpu[i] = ProcessArray[i]->type;
				centers_cpu[i * 3 + 0] = ProcessArray[i]->center[0];
				centers_cpu[i * 3 + 1] = ProcessArray[i]->center[1];
				centers_cpu[i * 3 + 2] = ProcessArray[i]->center[2];
				nodes_cpu[i * 4 + 0] = ProcessArray[i]->center[0];
				nodes_cpu[i * 4 + 1] = ProcessArray[i]->center[1];
				nodes_cpu[i * 4 + 2] = ProcessArray[i]->center[2];
				nodes_cpu[i * 4 + 3] = ProcessArray[i]->center[3];
			}
			cuda_node_calc_const_r_kernel (
				queue_flag, half, types_cpu, centers_cpu, nodes_cpu);
			for (size_t i = 0; i < queue_flag; i++)
			{
				ProcessArray[i]->type = types_cpu[i];
				if (ProcessArray[i]->type == EMPTY || ProcessArray[i]->type == LEAF)
				{
					ProcessArray[i]->node[0] = nodes_cpu[i * 4 + 0];
					ProcessArray[i]->node[1] = nodes_cpu[i * 4 + 1];
					ProcessArray[i]->node[2] = nodes_cpu[i * 4 + 2];
					ProcessArray[i]->node[3] = nodes_cpu[i * 4 + 3];
				} else if (ProcessArray[i]->type == INTERNAL) {
					for (Index t = 0; t.v < 8; t++)
					{
						ProcessArray[i]->children[t.v] = new TNode(this, ProcessArray[i], t);
						WaitingStack.push_back(ProcessArray[i]->children[t.v]);
					}
				} else {
					printf("Unexcepted node type.");
					exit(1);
				}
			}
		} else {
			cuvrs = new float[queue_flag];
			min_raiduses = new float[queue_flag];
			emptys = new bool[queue_flag];
			#pragma omp parallel //for schedule(dynamic, OMP_THREADS_NUM) 
			{
				#pragma omp single
				{
					for (size_t i = 0; i < queue_flag; i++)
					{
						#pragma omp task
							beforeSampleEval(ProcessArray[i], cuvrs[i], min_raiduses[i], emptys[i]);
					}
				}
			}
			sample_points = new float[pow(getOverSampleQEF()+1, 3) * 4 * queue_flag];
			sample_grads = new float[pow(getOverSampleQEF()+1, 3) * 3 * queue_flag];
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
									ProcessArray[i], cuvrs[i], min_raiduses[i], 
									sample_points + int(i * pow(getOverSampleQEF()+1, 3) * 4), 
									sample_grads + int(i * pow(getOverSampleQEF()+1, 3) * 3));
						}
					}
				}
			}
			for (size_t i = 0; i < queue_flag; i++)
			{
				if (ProcessArray[i]->type == INTERNAL) {
					for (Index t = 0; t.v < 8; t++)
					{
						ProcessArray[i]->children[t.v] = new TNode(this, ProcessArray[i], t);
						WaitingStack.push_back(ProcessArray[i]->children[t.v]);
					}
				}
			}
		}
		depth++;
		half/=2;
	}
	// delete[] sample_points;
	// delete[] sample_grads;
	// delete[] cuvrs;
	// delete[] min_raiduses;
	// delete[] emptys;
	// delete[] oversamples;
	// delete[] types_cpu;
	// delete[] centers_cpu;
	// delete[] nodes_cpu;
	/*
	if (!USE_CUDA)
	{
		float* sample_points;
		float* sample_grads;
		float* cuvrs;
		float* min_raiduses;
		bool* emptys;
		int* oversamples;
		int all_sampling = 0;
		WaitingStack.push_back(root);
		while (!WaitingStack.empty())
		{
			all_sampling = 0;
			printf("%d \n", WaitingStack.size());
			for (queue_flag = 0; queue_flag < inProcessSize && !WaitingStack.empty(); queue_flag++)
			{
				ProcessArray[queue_flag] = WaitingStack.back();
				WaitingStack.pop_back();
			}
			layer_time = get_time();
			cuvrs = new float[queue_flag];
			min_raiduses = new float[queue_flag];
			emptys = new bool[queue_flag];
			oversamples = new int[queue_flag]();
			phase1 = get_time();
			#pragma omp parallel //for schedule(dynamic, OMP_THREADS_NUM) 
			{
				#pragma omp single
				{
					for (size_t i = 0; i < queue_flag; i++)
					{
						#pragma omp task
							beforeSampleEval(ProcessArray[i], cuvrs[i], min_raiduses[i], emptys[i], oversamples[i]);
					}
				}
			}
			// printf("\n");
			for (size_t i = 0; i < queue_flag; i++)
			{
				all_sampling += pow(oversamples[i]+1, 3);
			}
			sample_points = new float[all_sampling * 4];
			sample_grads = new float[all_sampling * 3];
			//TODO: Sampling
			#pragma omp parallel //for schedule(dynamic, OMP_THREADS_NUM)
			{
				#pragma omp single
				{
					for (size_t i = 0; i < queue_flag; i++)
					{
						if (!emptys[i])
						{
							#pragma omp task
								afterSampleEval(ProcessArray[i], cuvrs[i], min_raiduses[i], oversamples, i, sample_points, sample_grads);
						}
					}
				}
			}
			phase2 = get_time();
			for (size_t i = 0; i < queue_flag; i++)
			{
				if (ProcessArray[i]->type == INTERNAL) {
					for (Index t = 0; t.v < 8; t++)
					{
						ProcessArray[i]->children[t.v] = new TNode(this, ProcessArray[i], t);
						WaitingStack.push_back(ProcessArray[i]->children[t.v]);
					}
				}
			}
			printf("Layer time = %f; Phase 1 time = %f; Phase 2 time = %f; Phase 3 time = %f;\n", get_time() - layer_time, phase1 - layer_time, phase2 - phase1, get_time() - phase2);
		}
		delete[] sample_points;
		delete[] sample_grads;
		delete[] cuvrs;
		delete[] min_raiduses;
		delete[] emptys;
		delete[] oversamples;
	} else {
		if (IS_CONST_RADIUS)
		{
			char* types_cpu;
			char* depths_cpu;
			float* centers_cpu;
			float* half_lengthes_cpu;
			float* nodes_cpu;
			WaitingStack.push_back(root);
			while (!WaitingStack.empty())
			{
				printf("%d \n", WaitingStack.size());
				for (queue_flag = 0; !WaitingStack.empty(); queue_flag++)
				{
					ProcessArray[queue_flag] = WaitingStack.back();
					WaitingStack.pop_back();
				}
				layer_time = get_time();
				types_cpu = new char[queue_flag];
				depths_cpu = new char[queue_flag];
				centers_cpu = new float[queue_flag * 3];
				half_lengthes_cpu = new float[queue_flag];
				nodes_cpu = new float[queue_flag * 4];
				for (size_t i = 0; i < queue_flag; i++)
				{
					types_cpu[i] = ProcessArray[i]->type;
					depths_cpu[i] = ProcessArray[i]->depth;
					centers_cpu[i * 3 + 0] = ProcessArray[i]->center[0];
					centers_cpu[i * 3 + 1] = ProcessArray[i]->center[1];
					centers_cpu[i * 3 + 2] = ProcessArray[i]->center[2];
					half_lengthes_cpu[i] = ProcessArray[i]->half_length;
					nodes_cpu[i * 4 + 0] = ProcessArray[i]->center[0];
					nodes_cpu[i * 4 + 1] = ProcessArray[i]->center[1];
					nodes_cpu[i * 4 + 2] = ProcessArray[i]->center[2];
					nodes_cpu[i * 4 + 3] = ProcessArray[i]->center[3];
				}
				phase1 = get_time();
				cuda_node_calc_const_r_kernel (
					queue_flag, types_cpu, depths_cpu, centers_cpu, half_lengthes_cpu, nodes_cpu);
				phase2 = get_time();
				for (size_t i = 0; i < queue_flag; i++)
				{
					ProcessArray[i]->type = types_cpu[i];
					if (ProcessArray[i]->type == EMPTY || ProcessArray[i]->type == LEAF)
					{
						ProcessArray[i]->node[0] = nodes_cpu[i * 4 + 0];
						ProcessArray[i]->node[1] = nodes_cpu[i * 4 + 1];
						ProcessArray[i]->node[2] = nodes_cpu[i * 4 + 2];
						ProcessArray[i]->node[3] = nodes_cpu[i * 4 + 3];
					} else if (ProcessArray[i]->type == INTERNAL) {
						for (Index t = 0; t.v < 8; t++)
						{
							ProcessArray[i]->children[t.v] = new TNode(this, ProcessArray[i], t);
							WaitingStack.push_back(ProcessArray[i]->children[t.v]);
						}
					} else {
						printf("Unexcepted node type.");
						exit(1);
					}
				}
				// printf("Layer time = %f; Phase 1 time = %f; Phase 2 time = %f; Phase 3 time = %f;\n", get_time() - layer_time, phase1 - layer_time, phase2 - phase1, get_time() - phase2);
			}
			delete[] types_cpu;
			delete[] depths_cpu;
			delete[] centers_cpu;
			delete[] half_lengthes_cpu;
			delete[] nodes_cpu;
		} else {
			printf("GPU acceleration for multi resolution particle data is not yet supported.");
			exit(0);
		}
	}
	*/
	double t_finish = get_time();
	printf("Time generating tree = %f;\n", t_finish - t_start);	
	_STATE++;
}

void SurfReconstructor::Run()
{
	generalModeRun();
}

void SurfReconstructor::generalModeRun()
{
	printf("-= Run =-\n");
	_OurRoot = nullptr;
    _OurMesh->reset();
    _STATE = 0;

    double time_all_start = get_time();
	double temp_time, last_temp_time;

	printf("-= Box =-\n");
	loadRootBox();

	temp_time = get_time();

	printf("-= Build Neighbor Searcher =-\n");
	if (IS_CONST_RADIUS)
	{
    	_hashgrid = new HashGrid(_GlobalParticles, _BoundingBox, _RADIUS, _NEIGHBOR_FACTOR);
	} else {
		_searcher = new MultiLevelSearcher(&_GlobalParticles, _BoundingBox, _GlobalRadiuses, _NEIGHBOR_FACTOR);
	}
    last_temp_time = temp_time;
    temp_time = get_time();
	printf("   Build Neighbor Searcher Time = %f \n", temp_time - last_temp_time);

    printf("-= Initialize Evaluator =-\n");
	_evaluator = new Evaluator(this, &_GlobalParticles, _GlobalRadiuses, _RADIUS);
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

	_MAX_SCALAR = (IS_CONST_RADIUS ? _evaluator->CalculateMaxScalarConstR() : _evaluator->CalculateMaxScalarVarR());

	_ISO_VALUE = (IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR() : _evaluator->RecommendIsoValueVarR());
    printf("   Recommend Iso Value = %f\n", _ISO_VALUE);
	temp_time = get_time();
	_evaluator->CalcParticlesNormal();
	last_temp_time = temp_time;
	temp_time = get_time();
    printf("   Calculate Particals Normal Time = %f\n", temp_time - last_temp_time);
	
	if (USE_CUDA)
	{
		if (IS_CONST_RADIUS)
		{
			cuda_node_calc_initialize_const_r(
				_GlobalParticlesNum, _DEPTH_MIN, _RADIUS, _SMOOTH_FACTOR, _ISO_VALUE, _MAX_SCALAR, 
				_evaluator, _hashgrid
			);
		}
	}
	
	genIsoOurs();
	genIsoOurs();
	//genIsoOurs();

	if (USE_CUDA)
	{
		if (IS_CONST_RADIUS)
		{
			cuda_node_calc_release_const_r();
		}
	}

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);
	if (IS_CONST_RADIUS)
	{
    	delete(_hashgrid);
	} else {
		delete(_searcher);
	}
}

