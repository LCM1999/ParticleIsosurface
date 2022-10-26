#pragma once

#include <string>
#include <vector>
#include <math.h>
#include "iso_common.h"
#include "vect.h"
#include "visitorextract.h"
#include <map>
#include <iterator>
using namespace std;


struct TNode;
struct Graph;

struct Mesh
{
#ifdef USE_DMT
	vector< vect<3, vect3f> > tris;
#endif // USE_DMT

#ifdef JOIN_VERTS
	vector< vect<3, TopoEdge> > topoTris;
#endif

#ifdef USE_DMC
	map<vect3i, int> vertices_map;
	vector<vect3f> vertices;
	map<Triangle<vect3i>, int> tris_map;
	vector<Triangle<int>> tris;
	unsigned int verticesNum = 0;
	unsigned int trianglesNum = 0;

	int insert_vert(const vect3f& p)
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

	vect3i vect3f2vect3i(const vect3f& a)
	{
		vect3i r;
		for (size_t i = 0; i < 3; i++)
		{
			r[i] = int(round(a.v[i] * MESH_TOLERANCE));
		}
		return r;
	}

	vect3f vect3i2vect3f(const vect3i& a)
	{
		vect3f r;
		for (size_t i = 0; i < 3; i++)
		{
			r[i] = a.v[i] / MESH_TOLERANCE;
		}
		return r;
	}

	bool similiar_point(vect3f& v1, vect3f& v2)
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
		Triangle<vect3i> ti = Triangle<vect3i>(t0_i, t1_i, t2_i);
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
	
#endif // USE_DMC

	vector<vect3f> norms;
};

struct Global
{
	Global();
	int method;
	TNode* ourRoot;
	Mesh ourMesh, rootsMesh, dmcMesh, dcMesh;
};

extern Global g;
