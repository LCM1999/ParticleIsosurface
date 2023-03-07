#include <queue>
#include <stack>
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


SurfReconstructor::SurfReconstructor(std::vector<Eigen::Vector3f>& particles, 
std::vector<float>* radiuses, Mesh& mesh, 
float radius, float inf_factor)
{
	_GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalRadiuses = radiuses;
	_RADIUS = radius;
	_INFLUENCE_FACTOR = inf_factor;

	_OurMesh = &mesh;
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
    double maxLen, resizeLen;
	float r = _RADIUS;
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / r))));
	resizeLen = pow(2, _DEPTH_MAX) * r;
	while (resizeLen - maxLen < (_INFLUENCE_FACTOR * _RADIUS))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * r;
	}
	resizeLen *= 0.995;
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}
	
	_DEPTH_MIN = (_DEPTH_MAX - int(_DEPTH_MAX / 3));
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
	while (resizeLen - maxLen < (_INFLUENCE_FACTOR * maxR))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * avgR;
	}
	resizeLen *= 0.995;
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}

	_DEPTH_MIN = std::min(int(ceil(log2(ceil(maxLen / maxR)))) - 2, _DEPTH_MAX - int(_DEPTH_MAX / 3));
}


void SurfReconstructor::checkEmptyAndCalcCurv(TNode* tnode, bool& empty, float& curv, float& min_radius)
{
	Eigen::Vector3f norms(0, 0, 0);
	float area = 0;
	std::vector<int> insides;
	min_radius = FLT_MAX;
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
				if (_GlobalParticles[in].x() >= (box1.x() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)) && 
					_GlobalParticles[in].x() <= (box2.x() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)) &&
					_GlobalParticles[in].y() >= (box1.y() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)) && 
					_GlobalParticles[in].y() <= (box2.y() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)) &&
					_GlobalParticles[in].z() >= (box1.z() - ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)) && 
					_GlobalParticles[in].z() <= (box2.z() + ((IS_CONST_RADIUS ? _RADIUS : _GlobalRadiuses->at(in)) * 2.5f)))
				{
					all_splash = false;
					Eigen::Vector3f tempNorm = _evaluator->PariclesNormals[in];
					if (tempNorm == Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX))	{continue;}
					norms += tempNorm;
					area += tempNorm.norm();

					if (!IS_CONST_RADIUS && min_radius > _GlobalRadiuses->at(in))
					{
						min_radius = _GlobalRadiuses->at(in);
					}
				}
			}
		}
		empty = all_splash;
	}
	curv = (area == 0) ? 0.0 : (norms.norm() / area);
	if (IS_CONST_RADIUS || min_radius == FLT_MAX)
	{
		min_radius = _RADIUS;
	}
}

void SurfReconstructor::eval(TNode* tnode, Eigen::Vector3f* grad)
{
	float qef_error = 0, curv = 0, min_radius;
	bool signchange = false, recur = false, next = false, empty;
	Eigen::Vector4f* verts = nullptr;

	switch (tnode->type)
	{
	case EMPTY:
	case LEAF:
		return;
	case INTERNAL:
		next = true;
		break;
	case UNCERTAIN:
	{
		// evaluate QEF samples
		checkEmptyAndCalcCurv(tnode, empty, curv, min_radius);
		if (empty)
		{
			tnode->node[0] = tnode->center[0];
			tnode->node[1] = tnode->center[1];
			tnode->node[2] = tnode->center[2];
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
			tnode->type = EMPTY;
			return;
		}
		else if (tnode->depth <= _DEPTH_MIN)
		{
			tnode->node[0] = tnode->center[0];
			tnode->node[1] = tnode->center[1];
			tnode->node[2] = tnode->center[2];
			_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
		}
		else
		{
			verts = new Eigen::Vector4f[8];
			tnode->vertAll(curv, signchange, grad, verts, qef_error, min_radius);
		}
		if (std::isnan(curv))
		{
			curv = 0;
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

	if (next)
	{
		for (Index i; i < 8; i++)
		{
			eval(tnode->children[i], grad);
		}
	}
	else if (recur)
	{
		tnode->type = INTERNAL;
		// find points and function values in the subdivided cell
		Eigen::Vector3f g[3][3][3];
		float temp;
		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 3; y++)
			{
				for (int z = 0; z < 3; z++)
				{
					if (x == 1 || y == 1 || z == 1) {
						_evaluator->SingleEval(tnode->center + (Eigen::Vector3f((x - 1), (y - 1), (z - 1)) * tnode->half_length), temp, g[x][y][z]);
					}
					else {
						g[x][y][z] = grad[Index(x >> 1, y >> 1, z >> 1)];
					}
				}
			}
		}

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};

		// create children
		for (int t = 0; t < 8; t++)
		{
			Index i = t;
			tnode->children[i] = new TNode(this, tnode->nId * 8 + i + 1);
			tnode->children[i]->depth = tnode->depth + 1;
			tnode->children[i]->half_length = tnode->half_length / 2;
			tnode->children[i]->center =
				tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
			for (Index j; j < 8; j++)
			{
				grad[j] = g[i.x + j.x][i.y + j.y][i.z + j.z];
			}
			#pragma omp task
			eval(tnode->children[i], grad);
		}
	}
	else
	{
		tnode->type = LEAF;
	}
}

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

void SurfReconstructor::genIsoOurs()
{
    double t_start = get_time();

	TNode* root;
	Mesh* m = _OurMesh;
	TNode* loaded_tree = nullptr;
	Eigen::Vector3f grad[9];

	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	if (_STATE == 0)
	{
		printf("-= Calculating Tree Structure =-\n");
		root = new TNode(this, 0);
		root->center = Eigen::Vector3f(_RootCenter[0], _RootCenter[1], _RootCenter[2]);
		root->half_length = _RootHalfLength;
		float temp;
		for (Index i; i < 8; i++) {
			_evaluator->SingleEval(root->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * root->half_length), temp, grad[i]);
		}
		_OurRoot = root;
	}
	else if (_STATE == 1)
	{
		printf("-= Our Method =-\n");
		root = _OurRoot;
	}
	else if (_STATE == 2)
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
#pragma omp parallel
{
	#pragma omp single
	{
		#pragma omp task
		eval(root, grad);
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
    	_hashgrid = new HashGrid(_GlobalParticles, _BoundingBox, _INFLUENCE_FACTOR * _RADIUS);
	} else {
		_searcher = new MultiLevelSearcher(&_GlobalParticles, _GlobalRadiuses, _INFLUENCE_FACTOR);
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
	shrinkBox();
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
	genIsoOurs();

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);

}

