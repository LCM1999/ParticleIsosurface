#pragma once 
#include <iterator>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "iso_common.h"
#include "iso_method_ours.h"
#include "global.h"
#include "index.h"

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
	VisitorExtract(Mesh* m_);
	VisitorExtract(const char cdepth, Graph* g_);
	VisitorExtract(Mesh* m_, std::vector<TNode*>* part_);
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