#pragma once 
#include <iterator>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "iso_common.h"
#include "index.h"

class SurfReconstructor;
class TNode;
class Mesh;
class Graph;

struct TraversalData
{
	TraversalData(){}
	TraversalData(TNode *t);

	TNode *n; // node
	char depth; 

	void gen_trav(TraversalData &c, Index i);
};

struct VisitorExtract
{
	VisitorExtract(SurfReconstructor* surf_constructor, Mesh* m_);
	VisitorExtract(SurfReconstructor* surf_constructor, const char cdepth, Graph* g_);
	VisitorExtract(SurfReconstructor* surf_constructor, Mesh* m_, std::vector<TNode*>* part_);
	SurfReconstructor* constructor;
	Mesh* m;
	Graph* g;
	char constrained_depth = 0;
	std::vector<TNode*>* part = nullptr;

	bool belong2part(TraversalData& td);

	bool on_vert(TraversalData& a, TraversalData& b, TraversalData& c, TraversalData& d,
		TraversalData& aa, TraversalData& ba, TraversalData& ca, TraversalData& da);

	bool on_node(TraversalData &td);

	bool on_edge(TraversalData& td00, TraversalData& td10, TraversalData& td01, TraversalData& td11);

	bool on_face(TraversalData &td0, TraversalData &td1, char orient);
};