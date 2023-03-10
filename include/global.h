#pragma once

#include <vector>
#include <array>
#include <math.h>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <Eigen/Dense>
#include <unordered_map>


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

	vect3i(int a, int b, int c)
	{
		v[0] = a;
		v[1] = b;
		v[2] = c;
	}

	int operator[](const int i) const
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
		//std::sort(v.begin(), v.end());
	}

	std::array<T, 3> v;

	T operator[](const int i) const
	{
		assert(i >= 0 && i < 3);
		return v[i];
	}

	bool operator<(const Triangle<T>& t) const
	{
		return std::lexicographical_compare(v.begin(), v.end(), t.v.begin(), t.v.end());
	}
};


struct tri_hash{
	size_t operator()(const Triangle<int>& t) const{
		return size_t((t[0] * t[1] * t[2]) << (t[0] + t[1] + t[2]));
	}
};

struct tri_cmp{
	bool operator()(const Triangle<int>& t1, const Triangle<int>& t2) const {
		return t1 < t2;
	}
};

struct Mesh
{
	Mesh(float mesh_tolerance = 1e4);
    float MESH_TOLERANCE;
	std::vector<Eigen::Vector3f> IcosaTable;
	std::map<vect3i, int> vertices_map;
	std::vector<Eigen::Vector3f> vertices;
	std::unordered_map<Triangle<int>, int, tri_hash, tri_cmp> tris_map;
	std::vector<Triangle<int>> tris;
	unsigned int verticesNum = 0;
	unsigned int trianglesNum = 0;
	const int theta = 5;
	const int phi = 5;

	int insert_vert(const Eigen::Vector3f& p);
	vect3i vect3f2vect3i(const Eigen::Vector3f& a);
	Eigen::Vector3f vect3i2vect3f(const vect3i& a);
	bool similiar_point(Eigen::Vector3f& v1, Eigen::Vector3f& v2);
	void insert_tri(int t0, int t1, int t2);
	void reset();

	const int triangle_edge2vert[3][2] = { {1, 2}, {2, 0}, {0, 1} };

	std::vector<Eigen::Vector3f> norms;

	void BuildIcosaTable();
	void AppendSplash_ConstR(std::vector<Eigen::Vector3f>& splash_particles, const float radius);
	void AppendSplash_VarR(std::vector<Eigen::Vector3f>& splash_particles, std::vector<float>& splash_radius);
};
