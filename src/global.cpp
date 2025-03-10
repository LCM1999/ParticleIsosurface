#include "global.h"
#include "iso_method_ours.h"

#include <sstream>

Mesh::Mesh(const int mesh_precision)
{
	MESH_PRECISION = mesh_precision;
	reset();
	BuildIcosaTable();
}

int Mesh::insert_vert(unsigned long long id1, unsigned long long id2, const Eigen::Vector3f& p)
{
	// vect3<float> tp(p);
	// vect3<long long> tmp = vect3f2vect3i(tp);
	// // vect3<float> tmp(float(precise(p[0])), float(precise(p[1])), float(precise(p[2])));
	// if (vertices_map.find(tmp) == vertices_map.end())
	// {
	// 	verticesNum++;
	// 	vertices_map[tmp] = verticesNum;
	// 	// vertices.push_back(vect3i2vect3f(tmp));
	// 	vertices.push_back(p);
	// }
	auto [iterator, inserted] = 
		vertices_map.try_emplace(generate_string(std::vector<unsigned long long>({id1, id2})), verticesNum);
	if (inserted) {
		verticesNum++;
		vertices.push_back(p);
	}
	return iterator->second + 1;
}

void Mesh::insert_tri(int t0, int t1, int t2)
{
	// if ((t0 == t1 || t1 == t2 || t0 == t2))
	// {
	// 	return;
	// }
	// Triangle tv(t0, t1, t2);
	// if (tris_map.find(tv) == tris_map.end())
	// {
	// 	trianglesNum++;
	// 	tris_map[tv] = trianglesNum;
	// 	tris.push_back(tv);
	// }
	auto [iterator, inserted] = 
		tris_map.try_emplace(generate_string(std::vector<int>({t0, t1, t2})), trianglesNum);
	if (inserted) {
		trianglesNum++;
		tris.push_back(Triangle(t0, t1, t2));
	}
}

void Mesh::reset()
{
	vertices_map.clear();
	vertices.clear();
	tris.clear();
	verticesNum = 0;
	IcosaTable.clear();
	IcosaTable.resize(12);
}

void Mesh::BuildIcosaTable()
{
	const float PI = 3.1415926f;
	const float H_ANGLE = PI / 180 * 72;    // 72 degree = 360 / 5
	const float V_ANGLE = atanf(1.0f / 2); 
	float z, xy;                            // coords
	float hAngle1 = -PI / 2 - H_ANGLE / 2;  // start from -126 deg at 1st row
	float hAngle2 = -PI / 2;				// start from -90 deg at 2nd row

	IcosaTable[0] = Eigen::Vector3f(0, 0, 1);
	int i1, i2;

	for (size_t i = 1; i <= 5; i++)
	{
		i1 = i;
		i2 = i + 5;
		z = std::sin(V_ANGLE);
		xy = std::cos(V_ANGLE);

		IcosaTable[i1] = Eigen::Vector3f(xy * cos(hAngle1), xy * sin(hAngle1), z);
		IcosaTable[i2] = Eigen::Vector3f(xy * cos(hAngle2), xy * sin(hAngle2), -z);

		hAngle1 += H_ANGLE;
		hAngle2 += H_ANGLE;
	}

	IcosaTable[11] = Eigen::Vector3f(0, 0, -1);	
}

void Mesh::AppendSplash_ConstR(std::vector<Eigen::Vector3f>& splash_particles, const float radius)
{
	std::vector<int> tmp_vec_indices;
	for (const Eigen::Vector3f& pos : splash_particles)
	{
		tmp_vec_indices.clear();
		tmp_vec_indices.resize(12);
		for (size_t i = 0; i < 12; i++)
		{
			verticesNum++;
			vertices.push_back(Vertex(pos + IcosaTable[i] * radius));
			tmp_vec_indices[i] = verticesNum;
		}
		for (size_t i = 0; i < 5; i++)
		{
			insert_tri(tmp_vec_indices[0],	 				tmp_vec_indices[1 + i], 			tmp_vec_indices[(1 + i) % 5 + 1]);
			insert_tri(tmp_vec_indices[1 + i], 				tmp_vec_indices[(1 + i) % 5 + 1], 	tmp_vec_indices[6 + i]);
			insert_tri(tmp_vec_indices[(1 + i) % 5 + 1], 	tmp_vec_indices[6 + i], 			tmp_vec_indices[(1 + i) % 5 + 6]);
			insert_tri(tmp_vec_indices[6 + i], 				tmp_vec_indices[(1 + i) % 5 + 6], 	tmp_vec_indices[11]);
		}
	}
}

void Mesh::AppendSplash_VarR(std::vector<Eigen::Vector3f>& splash_particles, std::vector<float>& splash_radius)
{
	std::vector<int> tmp_vec_indices;
	for (int spi = 0; spi < splash_particles.size(); spi++)
	{
		tmp_vec_indices.clear();
		tmp_vec_indices.resize(12);
		for (size_t i = 0; i < 12; i++)
		{
			verticesNum++;
			vertices.push_back(Vertex(splash_particles[spi] + IcosaTable[i] * splash_radius[spi]));
			tmp_vec_indices[i] = verticesNum;
		}
		for (size_t i = 0; i < 5; i++)
		{
			insert_tri(tmp_vec_indices[0],	 				tmp_vec_indices[1 + i], 			tmp_vec_indices[(1 + i) % 5 + 1]);
			insert_tri(tmp_vec_indices[1 + i], 				tmp_vec_indices[(1 + i) % 5 + 1], 	tmp_vec_indices[6 + i]);
			insert_tri(tmp_vec_indices[(1 + i) % 5 + 1], 	tmp_vec_indices[6 + i], 			tmp_vec_indices[(1 + i) % 5 + 6]);
			insert_tri(tmp_vec_indices[6 + i], 				tmp_vec_indices[(1 + i) % 5 + 6], 	tmp_vec_indices[11]);
		}
	}
}
