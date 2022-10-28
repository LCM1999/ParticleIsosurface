#pragma once

#include <vector>
#include <math.h>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <Eigen/Dense>
#include "iso_common.h"
#include "iso_method_ours.h"

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
	std::map<vect3i, int> vertices_map;
	std::vector<Eigen::Vector3f> vertices;
	std::map<Triangle<vect3i>, int> tris_map;
	std::vector<Triangle<int>> tris;
	unsigned int verticesNum = 0;
	unsigned int trianglesNum = 0;

	int insert_vert(const Eigen::Vector3f& p)
	{
		vect3i tmp = vect3f2vect3i(p);
		if (vertices_map.find(tmp) == vertices_map.end())
		{
			verticesNum++;
			vertices_map[tmp] = verticesNum;
			vertices.push_back(p);
		}
		return vertices_map[tmp];
	}

	vect3i vect3f2vect3i(const Eigen::Vector3f& a)
	{
		vect3i r;
		for (size_t i = 0; i < 3; i++)
		{
			r[i] = int(round(a[i] * MESH_TOLERANCE));
		}
		return r;
	}

	Eigen::Vector3f vect3i2vect3f(const vect3i& a)
	{
		Eigen::Vector3f r;
		for (size_t i = 0; i < 3; i++)
		{
			r[i] = a.v[i] / MESH_TOLERANCE;
		}
		return r;
	}

	bool similiar_point(Eigen::Vector3f& v1, Eigen::Vector3f& v2)
	{
		for (size_t i = 0; i < 3; i++)
		{
			if (abs(v1[i] - v2[i]) >= MESH_TOLERANCE)
			{
				return false;
			}
		}
		return true;
	}

	void insert_tri(int t0, int t1, int t2)
	{
		if ((t0 == t1 || t1 == t2 || t0 == t2))
		{
			return;
		}
		
		vect3i t0_i = vect3f2vect3i(vertices[(t0 - 1)]);
		vect3i t1_i = vect3f2vect3i(vertices[(t1 - 1)]);
		vect3i t2_i = vect3f2vect3i(vertices[(t2 - 1)]);
		Triangle<vect3i> ti(t0_i, t1_i, t2_i);
		//float length[3];
		//int top, bottom1, bottom2;
		//float height, half;
		//double area;
		//half = 0;
		//for (size_t i = 0; i < 3; i++)
		//{
		//	length[i] = (vertices[t.v[triangle_edge2vert[i][1]] - 1] - vertices[t.v[triangle_edge2vert[i][0]] - 1]).length();
		//	half += length[i];
		//}
		//half /= 2;
		//for (size_t i = 0; i < 3; i++)
		//{
		//	if (length[i] >= length[triangle_edge2vert[i][0]] && length[i] >= length[triangle_edge2vert[i][1]])
		//	{
		//		top = i; bottom1 = triangle_edge2vert[i][0]; bottom2 = triangle_edge2vert[i][1];
		//		break;
		//	}
		//}
		//area = sqrt(half * (half - length[0]) * (half - length[1]) * (half - length[2]));
		//height = area * 2 / length[top];
		//if ((height / length[top]) < LOW_MESH_QUALITY)
		//{
		//	vertices[t.v[top] - 1] =
		//		(vertices[t.v[bottom1] - 1] * (length[bottom1] / (length[bottom1] + length[bottom2])) +
		//			vertices[t.v[bottom2] - 1] * (length[bottom2] / (length[bottom1] + length[bottom2])));
		//	//printf("Elimit: %d, %d;  ", t.v[top], t.v[top]);
		//}
		//else
		//{
		//}
		if (tris_map.find(ti) == tris_map.end())
		{
			trianglesNum++;
			tris_map[ti] = trianglesNum;
			Triangle<int> tv(t0, t1, t2);
			tris.push_back(tv);
		}
	}

	void reset()
	{
		vertices_map.clear();
		vertices.clear();
		tris.clear();
		verticesNum = 0;
	}

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

	Graph(std::vector<TNode*>& layer_nodes, int vwn = 1)
	{
		vNum = 0;
		eNum = 0;
		ncon = vwn;
		for (size_t i = 0; i < layer_nodes.size(); i++)
		{
			vidx_map[layer_nodes[i]->nId] = vNum;
			vNum++;
			for (size_t j = 0; j < ncon; j++)
			{
				vwgt.push_back(layer_nodes[i]->getWeight());
			}
		}
		gAdj.resize(vNum);
	}

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

struct Global
{
	Global();
	int method;
	TNode* ourRoot;
	Mesh ourMesh;
};

extern Global g;
