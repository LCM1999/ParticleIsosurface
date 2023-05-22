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

extern "C" void cuda_node_calc_const_r_kernel();

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
	float area = 0;
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
					area += tempNorm.norm();
					if (!IS_CONST_RADIUS)
					{
						if (min_radius > _GlobalRadiuses->at(in))
						{
							min_radius = _GlobalRadiuses->at(in);
						}
					}
					all_splash = false;
					// p_vol += pow((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)), 3);
				}
			}
		}
		empty = all_splash;
		// if ((p_vol > node_vol))
		// {
		// 	empty = true;
		// }
	}
	curv = (area == 0) ? 1.0 : (norms.norm() / area);
}
/*
void SurfReconstructor::eval(TNode* tnode)
{
	float qef_error = 0, curv = 0, min_radius;
	bool signchange = false, recur = false, next = false, empty;

	switch (tnode->type)
	{
	case EMPTY:
	case LEAF:
		return;
	case INTERNAL:
	{
		for (Index i; i < 8; i++)
		{
			#pragma omp task
				eval(tnode->children[i]);
		}
		return;
	}
	case UNCERTAIN:
	{
		// double check_time = 0, generate_time = 0, sampling_time = 0, calc_time = 0;
		// evaluate QEF samples
		// check_time = get_time();
		checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
		// check_time = get_time() - check_time;
		if (empty)
		{
			Eigen::Vector3f tempG;
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], tempG);
			tnode->type = EMPTY;
			return;
		}
		else if (tnode->depth <= _DEPTH_MIN)
		{
			Eigen::Vector3f tempG;
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], tempG);
		}
		else
		{
			tnode->vertAll(curv, signchange, qef_error, min_radius);
			// std::vector<Eigen::Vector4f> sample_points;
			// std::vector<Eigen::Vector3f> sample_grads;
			// int over_sample;
			// // generate_time = get_time();
			// tnode->GenerateSampling(sample_points, sample_grads, over_sample, min_radius);
			// // generate_time = get_time() - generate_time;
			// // sampling_time = get_time();
			// tnode->NodeSampling(curv, signchange, sample_points, sample_grads, over_sample);
			// sampling_time = get_time() - sampling_time;
			// calc_time = get_time();
			// tnode->NodeCalcNode(sample_points, sample_grads, over_sample);
			// calc_time = get_time() - calc_time;
			// printf("check:%f, generate:%f, sampling:%f, calc:%f\n", check_time, generate_time, sampling_time, calc_time);
		}
		
		// judge this node need calculate iso-surface
		float cellsize = 2 * tnode->half_length;

		// check max/min sizes of cells
		bool issmall = (cellsize - min_radius) < _TOLERANCE;// || depth >= DEPTH_MAX;
		if (issmall)
		{
			// it's a leaf
			tnode->type = LEAF;
			return;
		}
		//static float maxsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MIN);
		bool isbig = (tnode->depth <= _DEPTH_MIN);
		// check for qef error
		//bool badqef = (qef_error / cellsize) > _BAD_QEF;

		// check curvature
		bool badcurv = curv < _FLATNESS;

		recur = isbig || (signchange && badcurv);	//(badcurv || badqef)

	}
	break;
	default:
		break;
	}

	if (recur)
	{
		tnode->type = INTERNAL;
		// find points and function values in the subdivided cell

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};

		// create children
		for (int t = 0; t < 8; t++)
		{
			Index i = t;
			tnode->children[i] = new TNode(this);
			tnode->children[i]->depth = tnode->depth + 1;
			tnode->children[i]->half_length = tnode->half_length / 2;
			tnode->children[i]->center =
				tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
			tnode->children[i]->node[0] = tnode->children[i]->center[0];
			tnode->children[i]->node[1] = tnode->children[i]->center[1];
			tnode->children[i]->node[2] = tnode->children[i]->center[2];
			#pragma omp task
				eval(tnode->children[i]);
		}
	}
	else
	{
		tnode->type = LEAF;
	}
}
*/
void SurfReconstructor::beforeSampleEval(TNode* tnode, float& curv, float& min_radius, bool& empty, int& oversample)
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
		bool full = false;
		checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
		if (empty)
		{
			Eigen::Vector3f tempG;
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], tempG);
			tnode->type = EMPTY;
			return;
		}
		// else if (tnode->depth <= _DEPTH_MIN)
		// {
		// 	Eigen::Vector3f tempG;
		// 	_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], tempG);
		// }
	}
	break;
	default:
		break;
	}
	if (!empty)
	{
		oversample = getOverSampleQEF();	//, int(ceil((tnode->half_length * 2) / min_radius) + 2);
		// if (tnode->depth <= getDepthMin())
		// {
		// 	oversample = getOverSampleQEF();
		// } else {
		// 	oversample = ;
		// }
	}
}

void SurfReconstructor::afterSampleEval(
	TNode* tnode, float& curv, float& min_radius, int* oversamples, const int index,
	float* sample_points, float* sample_grads)
{
	// double generate_time = 0, sampling_time = 0, calc_time = 0;
	bool isbig = (tnode->depth < _DEPTH_MIN);
	bool signchange = false;
	// tnode->vertAll(curv, signchange, qef_error, min_radius);
	int sampling_idx = 0, oversample = oversamples[index];

	for (size_t i = 0; i < index; i++)
	{
		sampling_idx += pow(oversamples[i] + 1, 3);
	}
	if (!isbig)
	{
		// generate_time = get_time();
		tnode->GenerateSampling(sample_points, sampling_idx, oversample);
		// generate_time = get_time() - generate_time;
		// sampling_time = get_time();
		tnode->NodeSampling(curv, signchange, sample_points, sample_grads, sampling_idx, oversample);
		// sampling_time = get_time() - sampling_time;
		// tnode->NodeCalcNode(sample_points, sample_grads, sampling_idx, oversample);
	}
	
	// judge this node need calculate iso-surface
	float cellsize = 2 * tnode->half_length;

	// check max/min sizes of cells
	bool issmall = (cellsize - min_radius) < _TOLERANCE;// || tnode->depth >= _DEPTH_MAX;
	if (issmall)
	{
		// it's a leaf
		tnode->type = LEAF;
		// calc_time = get_time();
		tnode->NodeCalcNode(sample_points, sample_grads, sampling_idx, oversample);
		// calc_time = get_time() - calc_time;
		// printf("generate:%f, sampling:%f, calc:%f\n", generate_time, sampling_time, calc_time);
		return;
	}
	//static float maxsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MIN);
	// check for qef error
	//bool badqef = (qef_error / cellsize) > _BAD_QEF;

	// check curvature
	bool badcurv = curv < _FLATNESS;

	bool recur = isbig || (signchange && badcurv);	//(badcurv || badqef)

	if (recur)
	{
		tnode->type = INTERNAL;
		// find points and function values in the subdivided cell

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};

		// create children
		for (int t = 0; t < 8; t++)
		{
			Index i = t;
			tnode->children[i] = new TNode(this);
			tnode->children[i]->depth = tnode->depth + 1;
			tnode->children[i]->half_length = tnode->half_length / 2;
			tnode->children[i]->center =
				tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
			tnode->children[i]->node[0] = tnode->children[i]->center[0];
			tnode->children[i]->node[1] = tnode->children[i]->center[1];
			tnode->children[i]->node[2] = tnode->children[i]->center[2];
			// #pragma omp task
			// eval(tnode->children[i]);
		}
		// printf("generate:%f, sampling:%f\n", generate_time, sampling_time);
	}
	else
	{
		tnode->type = LEAF;
		// calc_time = get_time();
		tnode->NodeCalcNode(sample_points, sample_grads, sampling_idx, oversample);
		// calc_time = get_time() - calc_time;
		// printf("generate:%f, sampling:%f, calc:%f\n", generate_time, sampling_time, calc_time);
	}
}
/*
void get_division_depth(TNode* root, char& cdepth, std::vector<TNode*>& layer_nodes)
{
	std::queue<TNode*> layer_list;
	int layer_num = 0, layer_search_num = 0;
	int layer_depth = 0;
	TNode* temp;
	layer_list.push(root);

	auto queue2vect = [&](std::queue<TNode*>& q, std::vector<TNode*>& v)
	{
		while (!q.empty())
		{
			v.push_back(q.front());
			q.pop();
		}
	};

	if (OMP_THREADS_NUM <= 1)
	{
		cdepth = layer_depth;
		queue2vect(layer_list, layer_nodes);
		return;
	}

	do 
	{
		layer_depth++;
		layer_search_num = layer_list.size();
		for (size_t i = 0; i < layer_search_num; i++)
		{
			temp = layer_list.front();
			layer_list.pop();
			switch (temp->type)
			{
			case EMPTY:
			case LEAF:
				layer_list.push(temp);
				break;
			case INTERNAL:
				for (TNode* child: temp->children)
				{
					layer_list.push(child);
				}
				break;
			default:
				printf("Error: Get Uncertain Node During Octree division;");
				exit(1);
			}
		}
	} while (layer_list.size() < OMP_THREADS_NUM);	
	cdepth = layer_depth;
	queue2vect(layer_list, layer_nodes);
	return;
}
*/
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
			for (queue_flag = 0; queue_flag < inProcessSize && !WaitingStack.empty(); queue_flag++)
			{
				ProcessArray[queue_flag] = WaitingStack.back();
				WaitingStack.pop_back();
			}
			sample_points;
			sample_grads;
			cuvrs = new float[queue_flag];
			min_raiduses = new float[queue_flag];
			emptys = new bool[queue_flag];
			oversamples = new int[queue_flag]();
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
			for (size_t i = 0; i < queue_flag; i++)
			{
				all_sampling += pow(oversamples[i]+1, 3);
			}
			printf("%d\n", all_sampling);
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
			for (size_t i = 0; i < queue_flag; i++)
			{
				for (size_t j = 0; j < 8; j++)
				{
					if (ProcessArray[i]->children[0] == 0)
					{
						break;
					}
					
					if (ProcessArray[i]->children[j]->type == UNCERTAIN)
					{
						WaitingStack.push_back(ProcessArray[i]->children[j]);
					}
				}
			}
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
			
			WaitingStack.push_back(root);
			while (!WaitingStack.empty())
			{
				for (queue_flag = 0; queue_flag && !WaitingStack.empty(); queue_flag++)
				{
					ProcessArray[queue_flag] = WaitingStack.back();
					WaitingStack.pop_back();
				}
				
				cuda_node_calc_const_r_kernel (
					blocksPerGrid, threadsPerBlock,
					particles_gpu, Gs_gpu, splashs_gpu, particles_gradients_gpu, 
					hash_list_gpu, index_list_gpu, start_list_keys_gpu, start_list_values_gpu, end_list_keys_gpu, end_list_values_gpu, 
					types_gpu, depths_gpu, centers_gpu, half_lengthes_gpu, tnode_num_gpu, nodes_gpu
				);
				
				for (size_t i = 0; i < queue_flag; i++)
				{
					ProcessArray[i]->type = types_cpu[i];
					if (ProcessArray[i]->type == EMPTY || ProcessArray[i]->type == LEAF)
					{
						ProcessArray[i]->node[0] = nodes_cpu[i * 3 + 0];
						ProcessArray[i]->node[1] = nodes_cpu[i * 3 + 1];
						ProcessArray[i]->node[2] = nodes_cpu[i * 3 + 2];
						ProcessArray[i]->node[3] = nodes_cpu[i * 3 + 3];
					} else if (ProcessArray[i]->type == INTERNAL) {
						for (Index t = 0; t.v < 8; t++)
						{
							ProcessArray[i]->children[t.v] = new TNode(this);
							ProcessArray[i]->children[t.v]->depth = ProcessArray[i]->depth + 1;
							ProcessArray[i]->children[t.v]->half_length = ProcessArray[i]->half_length / 2;
							ProcessArray[i]->children[t.v]->center =
								ProcessArray[i]->center + (Eigen::Vector3f(sign(t.x), sign(t.y), sign(t.z)) * ProcessArray[i]->half_length / 2);
							ProcessArray[i]->children[t.v]->node[0] = ProcessArray[i]->children[t.v]->center[0];
							ProcessArray[i]->children[t.v]->node[1] = ProcessArray[i]->children[t.v]->center[1];
							ProcessArray[i]->children[t.v]->node[2] = ProcessArray[i]->children[t.v]->center[2];
							WaitingStack.push_back(ProcessArray[i]->children[t.v]);
						}
					} else {
						exit(1);
					}
				}
			}
		} else {
			printf("GPU acceleration for multi resolution particle data is not yet supported.");
			exit(0);
		}
	}
	double t_finish = get_time();
	printf("Time generating tree = %f\n", t_finish - t_start);	
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

	genIsoOurs();
	genIsoOurs();
	//genIsoOurs();

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);
	if (IS_CONST_RADIUS)
	{
    	delete(_hashgrid);
	} else {
		delete(_searcher);
	}
}

