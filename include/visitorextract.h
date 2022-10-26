#pragma once 
#include "iso_common.h"
#include "vect.h"
#include "index.h"
#include <iterator>
#include <vector>

struct Mesh;
struct TNode;
struct Graph;

struct TraversalData
{
	TraversalData(){}
	TraversalData(TNode *t);

	TNode *n; // node
	char depth; 

	void gen_trav(TraversalData &c, Index i);
};

template<class T>
struct Triangle
{
	Triangle() {};
	Triangle(const T& a, const T& b, const T& c)
	{
		v[0] = a;
		v[1] = b;
		v[2] = c;
	}

	T v[3];

	bool operator<(const Triangle<T>& t) const
	{
		return lexicographical_compare(v, v+3, t.v, t.v+3);
	}
};

#ifdef USE_DMT
#include <algorithm>

using namespace std;

struct TopoEdge
{
	TopoEdge(){}
	TopoEdge(vect3f &c, vect3f &d)
	{
		v[0] = c;
		v[1] = d;
		fix();
	}

	vect3f v[2];

	void fix()
	{
		if (v[1] < v[0])
		{
			swap(v[0], v[1]);
		}
	}
	bool operator<(const TopoEdge &a) const 
	{
		return lexicographical_compare(v, v+2, a.v, a.v+2);
	}
};
#endif

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

#ifdef USE_DMT
	bool on_edge(TraversalData &td00, TraversalData &td10, TraversalData &td01, TraversalData &td11, char orient);
#endif // USE_DMT

#ifdef USE_DMC
	bool on_edge(TraversalData& td00, TraversalData& td10, TraversalData& td01, TraversalData& td11);
#endif // USE_DMC

	bool on_face(TraversalData &td0, TraversalData &td1, char orient);

#ifdef USE_DMT

#ifdef JOIN_VERTS
	void processTet(vect4f *p, vect3f *topo);
#else
	void processTet(vect4f *p);
#endif

#endif
};