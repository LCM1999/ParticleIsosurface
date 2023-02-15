#include <queue>
#include <stack>
#include <omp.h>
//#include <metis.h>

#include "surface_reconstructor.h"
#include "hash_grid.h"
#include "evaluator.h"
#include "iso_method_ours.h"
#include "global.h"
#include "visitorextract.h"
#include "traverse.h"


SurfReconstructor::SurfReconstructor(std::vector<Eigen::Vector3f>& particles, 
std::vector<float>* densities, std::vector<float>* masses, std::vector<float>* radiuses, Mesh& mesh, 
float density, float mass, float radius, float inf_factor)
{
	_GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalDensities = densities;
	_GlobalMasses = masses;
	_GlobalRadiuses = radiuses;
	_DENSITY = density;
	_MASS = mass;
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

void SurfReconstructor::resizeRootBox()
{
    double maxLen, resizeLen;
	float r = _RADIUS;
	maxLen = (std::max)({ 
		(_BoundingBox[1] - _BoundingBox[0]) , 
		(_BoundingBox[3] - _BoundingBox[2]) , 
		(_BoundingBox[5] - _BoundingBox[4]) });
	_DEPTH_MAX = int(ceil(log2(ceil(maxLen / r))));
	resizeLen = pow(2, _DEPTH_MAX) * r;
	while (resizeLen - maxLen < (_INFLUENCE_FACTOR * _RADIUS * 2))
	{
		_DEPTH_MAX++;
		resizeLen = pow(2, _DEPTH_MAX) * r;
	}
	resizeLen *= 0.99;
	_RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
		_BoundingBox[i * 2] = center - _RootHalfLength;
		_BoundingBox[i * 2 + 1] = center + _RootHalfLength;
		_RootCenter[i] = center;
	}
	
	//_DEPTH_MIN = _DEPTH_MAX - 2;
	_DEPTH_MIN = (_DEPTH_MAX - int(_DEPTH_MAX / 3));
}

void SurfReconstructor::checkEmptyAndCalcCurv(TNode* tnode, bool& empty, float& curv)
{
	empty = true;
	Eigen::Vector3i min_xyz_idx, max_xyz_idx;
	_hashgrid->CalcXYZIdx((tnode->center - Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length)), min_xyz_idx);
	_hashgrid->CalcXYZIdx((tnode->center + Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length)), max_xyz_idx);
	//min_xyz_idx = origin_xyz_idx - Eigen::Vector3i(1, 1, 1);
	//max_xyz_idx = origin_xyz_idx + Eigen::Vector3i(1, 1, 1);
	long long temp_hash;
	Eigen::Vector3f norms(0, 0, 0);
	float area = 0;
	for (int x = min_xyz_idx[0] - 1; x < max_xyz_idx[0] + 1 && empty; x++)
	{
		for (int y = min_xyz_idx[1] - 1; y < max_xyz_idx[1] + 1 && empty; y++)
		{
			for (int z = min_xyz_idx[2] - 1; z < max_xyz_idx[2] + 1 && empty; z++)
			{
				if (empty)
				{
					temp_hash = _hashgrid->CalcCellHash(Eigen::Vector3i(x, y, z));
					if (temp_hash < 0)
						continue;
					if ((_hashgrid->StartList.find(temp_hash) != _hashgrid->StartList.end()) && (_hashgrid->EndList.find(temp_hash) != _hashgrid->EndList.end()))
					{
						if ((_hashgrid->EndList[temp_hash] - _hashgrid->StartList[temp_hash]) == 0)
						{
							empty = true;
						} else {
							empty = true;
							for (int countIndex = _hashgrid->StartList[temp_hash]; countIndex < _hashgrid->EndList[temp_hash]; countIndex++)
							{
								if (!_evaluator->CheckSplash(_hashgrid->IndexList[countIndex]))
								{
									empty = false;
									break;
								}
							}
						}
					}
				} 
				if (!empty &&
					x >= min_xyz_idx[0] && x < max_xyz_idx[0] && 
					y >= min_xyz_idx[1] && y < max_xyz_idx[1] && 
					z >= min_xyz_idx[2] && z < max_xyz_idx[2])
				{
					if ((_hashgrid->StartList.find(temp_hash) != _hashgrid->StartList.end()) && (_hashgrid->EndList.find(temp_hash) != _hashgrid->EndList.end()))
					{
						for (int countIndex = _hashgrid->StartList[temp_hash]; countIndex < _hashgrid->EndList[temp_hash]; countIndex++)
						{
							Eigen::Vector3f tempNorm = _evaluator->PariclesNormals[_hashgrid->IndexList[countIndex]];
							if (tempNorm == Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX))
							{
								continue;
							}
							norms += tempNorm;
							area += tempNorm.norm();
							// auto iter = _evaluator->SurfaceNormals.find(_hashgrid->IndexList[countIndex]);
							// if (iter != _evaluator->SurfaceNormals.end())
							// {
							// }
						}
					}
				}
			}
		}
	}
	curv = (area == 0) ? 0.0 : (norms.norm() / area);
}

void SurfReconstructor::eval(TNode* tnode, Eigen::Vector3f* grad, TNode* guide)
{
//	if (tnode->nId % _RECORD_STEP == 0 && _STATE == 0)
//	{
//		if (_NEED_RECORD)
//		{
//#pragma omp critical
//			{
//				RecordProgress(_OurRoot, (_OutputPath + "/" + _RECORD_PREFIX + std::to_string(_INDEX) + "_" + std::to_string(tnode->nId) + ".txt").c_str());
//			}
//		}
//		printf("Record at %llu\n", tnode->nId);
//	}
	float qef_error = 0, curv = 0;
	bool signchange = false, recur = false, next = false, empty;

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
		if (!guide || (guide && guide->children[0] == 0))
		{
			// evaluate QEF samples
			checkEmptyAndCalcCurv(tnode, empty, curv);
			if (empty)
			{
				tnode->node[0] = tnode->center[0];
				tnode->node[1] = tnode->center[1];
				tnode->node[2] = tnode->center[2];
				_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
				tnode->type = EMPTY;
				//EMPTY_VOLUME += pow(2 * tnode->half_length, 3);
				//printStatus();
				return;
			}
			else if (tnode->depth < _DEPTH_MIN)
			{
				tnode->node[0] = tnode->center[0];
				tnode->node[1] = tnode->center[1];
				tnode->node[2] = tnode->center[2];
				_evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
			}
			else
			{
				tnode->vertAll(curv, signchange, grad, qef_error);
			}
		}
		if (std::isnan(curv))
		{
			curv = 0;
		}
		
		//printf("%f ", curv);
		// judge this node need calculate iso-surface
		float cellsize = 2 * tnode->half_length;

		if (!guide)
		{
			// check max/min sizes of cells

			// static float minsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MAX);
			bool issmall = (cellsize - _RADIUS) < _TOLERANCE;// || depth >= DEPTH_MAX;
			if (issmall)
			{
				// it's a leaf
				tnode->type = LEAF;
				//DONE_VOLUME += pow(2 * tnode->half_length, 3);
				//printStatus();
				return;
			}
			//static float maxsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MIN);
			bool isbig = tnode->depth <= _DEPTH_MIN;
			//
			//// check for a sign change
			//if (!isbig)
			//	signchange |= changeSignDMC();

			// check for qef error
			//bool badqef = (qef_error / cellsize) > _BAD_QEF;

			// check curvature
			bool badcurv = curv < _FLATNESS;

			recur = isbig || (signchange && badcurv);	//(badcurv || badqef)
		}
		else
		{
			recur = guide->children[0] != 0;
		}
	}
	break;
	default:
		break;
	}

	if (next)
	{
		if (guide)
		{
			for (Index i; i < 8; i++)
			{
				eval(tnode->children[i], grad, guide->children[i]);
			}
		}
		else
		{
			for (Index i; i < 8; i++)
			{
				eval(tnode->children[i], grad, 0);
			}
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
		if (guide)
		{
			for (int t = 0; t < 8; t++)
			{
				Index i = t;
				tnode->children[i]->depth = tnode->depth + 1;
				tnode->children[i]->half_length = tnode->half_length / 2;
				tnode->children[i]->center =
					tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
				for (Index j; j < 8; j++)
				{
					grad[j] = g[i.x + j.x][i.y + j.y][i.z + j.z];
				}
				#pragma omp task
				eval(tnode->children[i], grad, guide->children[i]);
			}
		}
		else
		{
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
				eval(tnode->children[i], grad, 0);
			}
		}
	}
	else
	{
		tnode->type = LEAF;
		//DONE_VOLUME += pow(2 * tnode->half_length, 3);
		//printStatus();
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

	TNode* guide;
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
		guide = _OurRoot;
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
		guide = _OurRoot;
		root = guide;
	}
	else if (_STATE == 2)
	{
		/*----------Make Graph----------*/
		/*
		char cdepth = 0;
		std::vector<TNode*> layer_nodes;
		get_division_depth(_OurRoot, cdepth, layer_nodes);
		printf("   cDepth = %d, layerNodesNum = %d\n", cdepth, layer_nodes.size());
		Graph layerGraph(layer_nodes);
		VisitorExtract gv(this, cdepth, &layerGraph);
		TraversalData gtd(_OurRoot);
		traverse_node<trav_face>(gv, gtd);
		int* xadj = new int[layerGraph.vNum + 1];
		int* adjncy = new int[layerGraph.eNum * 2];
		layerGraph.getXAdjAdjncy(xadj, adjncy);

		std::vector<idx_t> part(layerGraph.vNum, 0);
		idx_t nParts = OMP_THREADS_NUM;
		idx_t obj_val;

		int ret = METIS_PartGraphRecursive(&layerGraph.vNum, &layerGraph.ncon, xadj, adjncy, layerGraph.vwgt.data(),
			NULL, layerGraph.ewgt.data(), &nParts, NULL, NULL, NULL, &obj_val, part.data());
		
		if (ret == rstatus_et::METIS_OK)
		{
			printf("-= METIS DIVISION SUCCESS =-\n");

		}
//		else
//		{
			printf("-= METIS ERROR, USE DEFAULT SINGLE THREAD =-\n");
		*/
		printf("-= Generate Surface =-\n");
		double t_gen_mesh = get_time();
		m->tris.reserve(1000000);
		VisitorExtract v(this, m);
		TraversalData td(_OurRoot);
		traverse_node<trav_vert>(v, td);
		std::vector<Eigen::Vector3f> splash_pos;
		for (int pIdx = 0; pIdx < getGlobalParticlesNum(); pIdx++)
		{
			if (_evaluator->CheckSplash(pIdx))
			{
				splash_pos.push_back(_evaluator->GlobalPoses->at(pIdx));
			}
		}
		m->AppendSplash_ConstR(splash_pos, _RADIUS);
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
		eval(root, grad, guide);
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
	
	printf("-= Resize Box =-\n");
	resizeRootBox();
	printf("   MAX_DEPTH = %d, MIN_DEPTH = %d\n", _DEPTH_MAX, _DEPTH_MIN);

	temp_time = get_time();

	printf("-= Build Hash Grid =-\n");
    _hashgrid = new HashGrid(this, _GlobalParticles, _BoundingBox, _INFLUENCE_FACTOR * _RADIUS * 2);
    last_temp_time = temp_time;
    temp_time = get_time();

	printf("   Build Hash Grid Time = %f \n", temp_time - last_temp_time);

    printf("-= Initialize Evaluator =-\n");

	_evaluator = new Evaluator(this, &_GlobalParticles, _GlobalDensities, _GlobalMasses, _GlobalRadiuses, _DENSITY, _MASS, _RADIUS);
	last_temp_time = temp_time;
	temp_time = get_time();

	printf("   Initialize Evaluator Time = %f \n", temp_time - last_temp_time);

	//int max_density_index = std::distance(_GlobalDensities.begin(), std::max_element(_GlobalDensities.begin(), _GlobalDensities.end()));
	// _evaluator->SingleEval(_GlobalParticles[max_density_index], _MAX_SCALAR, *(Eigen::Vector3f*)NULL);
	_MAX_SCALAR = _evaluator->CalculateMaxScalar();

	_ISO_VALUE = _evaluator->RecommendIsoValue();
    printf("   Recommend Iso Value = %f\n", _ISO_VALUE);
	_evaluator->CalcParticlesNormal();
	last_temp_time = temp_time;
	temp_time = get_time();
    printf("   Calculate Particals Normal Time = %f\n", temp_time - last_temp_time);

	genIsoOurs();
	genIsoOurs();
	genIsoOurs();

	printf("-=  Total time= %f  =-\n", get_time() - time_all_start);

}

