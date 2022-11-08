#pragma once

#include <vector>
#include <math.h>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <Eigen/Dense>


class SurfReconstructor;
struct TNode;

struct vect3i
{
	int v[3];

	vect3i() {}

	vect3i(Eigen::Vector3i& e)
	{
		for (size_t i = 0; i < 3; i++)
		{
			v[i] = e.data()[i];
		}
	}

	int &operator[](const int i)
	{
		assert(i >= 0 && i < 3);
		return v[i];
	}

	bool operator<(const vect3i &a) const 
	{
		return std::lexicographical_compare(v, v+3, a.v, a.v+3);
	}
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
		return std::lexicographical_compare(v, v+3, t.v, t.v+3);
	}
};


struct Mesh
{
	Mesh(float mesh_tolerance = 1e4);
    float MESH_TOLERANCE;
	std::map<vect3i, int> vertices_map;
	std::vector<Eigen::Vector3f> vertices;
	std::map<Triangle<vect3i>, int> tris_map;
	std::vector<Triangle<int>> tris;
	unsigned int verticesNum = 0;
	unsigned int trianglesNum = 0;

	int insert_vert(const Eigen::Vector3f& p);
	vect3i vect3f2vect3i(const Eigen::Vector3f& a);
	Eigen::Vector3f vect3i2vect3f(const vect3i& a);
	bool similiar_point(Eigen::Vector3f& v1, Eigen::Vector3f& v2);
	void insert_tri(int t0, int t1, int t2);
	void reset();

	const int triangle_edge2vert[3][2] = { {1, 2}, {2, 0}, {0, 1} };

	std::vector<Eigen::Vector3f> norms;
};


struct Graph
{
	Graph() {};
	int vNum;
	int eNum;
	int ncon;
	std::map<unsigned __int64, int> vidx_map;
	std::vector<std::vector<int>> gAdj;
	std::vector<int> vwgt;
	std::vector<int> ewgt;

	Graph(std::vector<TNode*>& layer_nodes, int vwn = 1);

	void appendEdge(const unsigned __int64 nId1, const unsigned __int64 nId2)
	{
		gAdj[vidx_map[nId1]].push_back(vidx_map[nId2]);
		eNum++;
		gAdj[vidx_map[nId2]].push_back(vidx_map[nId1]);
		eNum++;
	}

	void getXAdjAdjncy(int* xadj, int* adjncy)
	{
		int adjncyIdx = 0;
		for (size_t i = 0; i < vNum; i++)
		{
			xadj[i] = adjncyIdx;
			for (size_t j = 0; j < gAdj[i].size(); j++)
			{
				adjncy[xadj[i] + j] = gAdj[i][j];
			}
			adjncyIdx += gAdj[i].size();
		}
		xadj[vNum] = adjncyIdx;
		ewgt.resize(eNum, 1);
	}
};
