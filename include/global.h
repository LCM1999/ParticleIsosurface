#pragma once

#include <vector>
#include <array>
#include <math.h>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <Eigen/Dense>
#include "rply.h"

struct TNode;

template<class T>
struct vect3
{
	T v[3];

	vect3() {}

	vect3(T a, T b, T c) {
		v[0] = a;
		v[1] = b;
		v[2] = c;
	}

	vect3(Eigen::Matrix<T, 3, 1> e)
	{
		v[0] = e[0];
		v[1] = e[1];
		v[2] = e[2];
	}

	T &operator[](const int i)
	{
		assert(i >= 0 && i < 3);
		return v[i];
	}

	bool operator<(const vect3 &a) const 
	{
		return std::lexicographical_compare(v, v+3, a.v, a.v+3);
	}
};

typedef vect3<double> Vertex;

struct Triangle
{
	Triangle() {};
	Triangle(const int& a, const int& b, const int& c)
	{
		v[0] = a;
		v[1] = b;
		v[2] = c;
		//std::sort(v.begin(), v.end());
	}

	std::array<int, 3> v;

	bool operator<(const Triangle& t) const
	{
		std::array<int, 3> temp_v(v);
		std::array<int, 3> temp_t = {t.v[0], t.v[1], t.v[2]};
		std::sort(temp_v.begin(), temp_v.end());
		std::sort(temp_t.begin(), temp_t.end());

		return std::lexicographical_compare(temp_v.begin(), temp_v.end(), temp_t.begin(), temp_t.end());
	}
};

template<class T>
std::string generate_string(std::vector<T> ids)
{
	std::sort(ids.begin(), ids.end());
	std::string key = "";
	for (auto id: ids)
	{
		key += std::to_string(id);
		key += ",";
	}
	return key;
}

class Mesh
{
public:
	Mesh(int mesh_precision = 1e4);
    int MESH_PRECISION;
	std::vector<Eigen::Vector3d> IcosaTable;
	std::map<std::string, int> vertices_map;
	std::vector<Vertex> vertices;
	std::map<std::string, int> tris_map;
	std::vector<Triangle> tris;
	unsigned int verticesNum = 0;
	unsigned int trianglesNum = 0;
	const int theta = 5;
	const int phi = 5;

	int insert_vert(unsigned long long id1, unsigned long long id2, const Eigen::Vector3d& p);
	vect3<long long> vect3f2vect3i(vect3<double>& a);
	vect3<double> vect3i2vect3f(vect3<int>& a);
	void insert_tri(int t0, int t1, int t2);
	void reset();

	const int triangle_edge2vert[3][2] = { {1, 2}, {2, 0}, {0, 1} };

	//std::vector<Eigen::Vector3d> norms;

	void BuildIcosaTable();
	void AppendSplash_ConstR(std::vector<Eigen::Vector3d>& splash_particles, const double radius);
	void AppendSplash_VarR(std::vector<Eigen::Vector3d>& splash_particles, std::vector<double>& splash_radius);
};
