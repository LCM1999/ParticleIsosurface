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

#define DEFAULT_INF_FACTOR 4.0

SurfReconstructor::SurfReconstructor(std::vector<Eigen::Vector3d>& particles, 
std::vector<double>* radiuses, Mesh* mesh, 
double radius, double iso_factor, double smooth_factor)
{
	_GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalRadiuses = radiuses;
	_RADIUS = radius;
	_NEIGHBOR_FACTOR = DEFAULT_INF_FACTOR;
	_SMOOTH_FACTOR = smooth_factor;
	_ISO_FACTOR = iso_factor;

	WaitingStack.clear();

	queue_flag = 0;

	_OurMesh = mesh;
}

inline void SurfReconstructor::loadRootBox()
{
	_BoundingBox[0] = _BoundingBox[2] = _BoundingBox[4] = FLT_MAX;
	_BoundingBox[1] = _BoundingBox[3] = _BoundingBox[5] = -FLT_MAX;
	for (const Eigen::Vector3d& p: _GlobalParticles)
	{
		if (p.x() < _BoundingBox[0]) _BoundingBox[0] = p.x();
		if (p.x() > _BoundingBox[1]) _BoundingBox[1] = p.x();
		if (p.y() < _BoundingBox[2]) _BoundingBox[2] = p.y();
		if (p.y() > _BoundingBox[3]) _BoundingBox[3] = p.y();
		if (p.z() < _BoundingBox[4]) _BoundingBox[4] = p.z();
		if (p.z() > _BoundingBox[5]) _BoundingBox[5] = p.z();
	}
	// _BoundingBox[0] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.x() < b.x(); })).x();
	// _BoundingBox[1] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.x() < b.x(); })).x();
	// _BoundingBox[2] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.y() < b.y(); })).y();
	// _BoundingBox[3] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.y() < b.y(); })).y();
	// _BoundingBox[4] = (*std::min_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.z() < b.z(); })).z();
	// _BoundingBox[5] = (*std::max_element(_GlobalParticles.begin(), _GlobalParticles.end(), 
	// [&] (Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.z() < b.z(); })).z();
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
    double maxLen, resizeLen;
	double r = _RADIUS;
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / r))));
	resizeLen = pow(2, _DEPTH_MAX) * r;
	while (resizeLen - maxLen < (_NEIGHBOR_FACTOR * _RADIUS * 2))
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
	
	_DEPTH_MIN = (_DEPTH_MAX - 2);	
}

void SurfReconstructor::resizeRootBoxVarR()
{
	double maxLen, resizeLen;
	double minR = _searcher->getMinRadius(), maxR = _searcher->getMaxRadius(), avgR = _searcher->getAvgRadius();
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / minR))));
	resizeLen = pow(2, _DEPTH_MAX) * minR;
	while (resizeLen - maxLen < (_NEIGHBOR_FACTOR * maxR * 2))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * maxR;
	}
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}

	_DEPTH_MIN = int(ceil(log2(ceil(maxLen / maxR)))) - 1; //, _DEPTH_MAX - int(_DEPTH_MAX / 3));
}


void SurfReconstructor::checkEmptyAndCalcCurv(TNode* tnode, bool& empty, double& curv, double& min_radius)
{
	Eigen::Vector3d norms(0, 0, 0);
	int area = 0;
	std::vector<int> insides;
	min_radius = IS_CONST_RADIUS ? _RADIUS : FLT_MAX;
	const Eigen::Vector3d 
	box1 = tnode->center - Eigen::Vector3d(tnode->half_length, tnode->half_length, tnode->half_length),
	box2 = tnode->center + Eigen::Vector3d(tnode->half_length, tnode->half_length, tnode->half_length);
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
					if (CALC_P_NORMAL)
					{
						Eigen::Vector3d tempNorm = _evaluator->PariclesNormals[in];
						if (tempNorm == Eigen::Vector3d(0, 0, 0))	{continue;}
						norms += tempNorm;
						area++;
					}
					
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
}

void SurfReconstructor::beforeSampleEval(TNode* tnode, double& curv, double& min_radius, bool& empty)
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
		checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
		if (empty)
		{
			_evaluator->SingleEval((Eigen::Vector3d&)tnode->node, tnode->node[3]);
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
	TNode* tnode, double& curv, double& min_radius, 
	double* sample_points, double* sample_grads)
{
	bool isbig = (tnode->depth < _DEPTH_MIN);
	bool signchange = false;
	double cellsize = 2 * tnode->half_length;

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
		tnode->NodeCalcNode(sample_points, sample_grads, cellsize);
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
		tnode->NodeCalcNode(sample_points, sample_grads, cellsize);
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
		root = new TNode(this, 0);
		root->center << _RootCenter[0], _RootCenter[1], _RootCenter[2];
		root->node << _RootCenter[0], _RootCenter[1], _RootCenter[2], 0.0;
		root->half_length = _RootHalfLength;
		_OurRoot = root;
	} else if (_STATE == 1) {
		printf("-= Generate Surface =-\n");
		double t_gen_mesh = get_time();
		m->tris.reserve(1000000);
		VisitorExtract v(this, m);
		TraversalData td(_OurRoot);
		traverse_node<trav_vert>(v, td);
		v.calc_vertices();
		v.generate_mesh();
		if (GEN_SPLASH)
		{
			std::vector<Eigen::Vector3d> splash_pos;
			std::vector<double> splash_radiuses;
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
		}
		double t_alldone = get_time();
		printf("Time generating polygons = %f\n", t_alldone - t_gen_mesh);
		return;
	}
	int depth = 0;
	double half = root->half_length;
	double* sample_points;
	double* sample_grads;
	double* cuvrs;
	double* min_raiduses;
	bool* emptys;
	WaitingStack.push_back(root);
	while (!WaitingStack.empty())
	{
		printf("%d \n", WaitingStack.size());
		for (queue_flag = 0; queue_flag < inProcessSize && !WaitingStack.empty(); queue_flag++)
		{
			ProcessArray[queue_flag] = WaitingStack.back();
			WaitingStack.pop_back();
		}
		cuvrs = new double[queue_flag];
		min_raiduses = new double[queue_flag];
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
		sample_points = new double[int(pow(getOverSampleQEF()+1, 3)) * 4 * queue_flag];
		sample_grads = new double[int(pow(getOverSampleQEF()+1, 3)) * 3 * queue_flag];
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
		depth++;
		half/=2;
	}
	delete[] sample_points;
	delete[] sample_grads;
	delete[] cuvrs;
	delete[] min_raiduses;
	delete[] emptys;
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
    	_hashgrid = new HashGrid(&_GlobalParticles, _BoundingBox, _RADIUS, _NEIGHBOR_FACTOR);
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
    printf("   Max Scalar Value = %f\n", _MAX_SCALAR);
	
	_ISO_VALUE = (IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR(_ISO_FACTOR) : _evaluator->RecommendIsoValueVarR(_ISO_FACTOR));
    printf("   Recommend Iso Value = %f\n", _ISO_VALUE);
	temp_time = get_time();
	if (CALC_P_NORMAL)
	{
		_evaluator->CalcParticlesNormal();
		last_temp_time = temp_time;
		temp_time = get_time();
		printf("   Calculate Particals Normal Time = %f\n", temp_time - last_temp_time);
	}

	genIsoOurs();
	genIsoOurs();

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);
	// if (IS_CONST_RADIUS)
	// {
    // 	delete(_hashgrid);
	// } else {
	// 	delete(_searcher);
	// }
}

