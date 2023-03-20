#include "global.h"
#include "iso_method_ours.h"

#include <sstream>

Mesh::Mesh(const int mesh_precision)
{
	MESH_PRECISION = mesh_precision;
	reset();
	BuildIcosaTable();
}

int Mesh::insert_vert(const Eigen::Vector3f& p)
{
	vect3<int> tmp = vect3f2vect3i(vect3<float>(p));
	// vect3<float> tmp(float(precise(p[0])), float(precise(p[1])), float(precise(p[2])));
	if (vertices_map.find(tmp) == vertices_map.end())
	{
		verticesNum++;
		vertices_map[tmp] = verticesNum;
		vertices.push_back(vect3i2vect3f(tmp));
	}
	return vertices_map[tmp];
}

double Mesh::precise(double x)
{
	std::stringstream is;
    double res;
    is.precision(MESH_PRECISION);
    is << x;
    is >> res;
    return res;
}

vect3<int> Mesh::vect3f2vect3i(vect3<float>& a)
{
	vect3<int> r;
	for (size_t i = 0; i < 3; i++)
	{
		r[i] = int(round(a[i] * MESH_PRECISION));
	}
	return r;
}

vect3<float> Mesh::vect3i2vect3f(vect3<int>& a)
{
	vect3<float> r;
	for (size_t i = 0; i < 3; i++)
	{
		r[i] = a[i] / float(MESH_PRECISION);
	}
	return r;
}

bool Mesh::similiar_point(Eigen::Vector3f& v1, Eigen::Vector3f& v2)
{
	for (size_t i = 0; i < 3; i++)
	{
		if (abs(v1[i] - v2[i]) >= MESH_PRECISION)
		{
			return false;
		}
	}
	return true;
}

void Mesh::insert_tri(int t0, int t1, int t2)
{
	if ((t0 == t1 || t1 == t2 || t0 == t2))
	{
		return;
	}
	
	// vect3i t0_i = vect3f2vect3i(vertices[(t0 - 1)]);
	// vect3i t1_i = vect3f2vect3i(vertices[(t1 - 1)]);
	// vect3i t2_i = vect3f2vect3i(vertices[(t2 - 1)]);
	// Triangle<vect3i> ti(t0_i, t1_i, t2_i);
	// //float length[3];
	// //int top, bottom1, bottom2;
	// //float height, half;
	// //double area;
	// //half = 0;
	// //for (size_t i = 0; i < 3; i++)
	// //{
	// //	length[i] = (vertices[t.v[triangle_edge2vert[i][1]] - 1] - vertices[t.v[triangle_edge2vert[i][0]] - 1]).length();
	// //	half += length[i];
	// //}
	// //half /= 2;
	// //for (size_t i = 0; i < 3; i++)
	// //{
	// //	if (length[i] >= length[triangle_edge2vert[i][0]] && length[i] >= length[triangle_edge2vert[i][1]])
	// //	{
	// //		top = i; bottom1 = triangle_edge2vert[i][0]; bottom2 = triangle_edge2vert[i][1];
	// //		break;
	// //	}
	// //}
	// //area = sqrt(half * (half - length[0]) * (half - length[1]) * (half - length[2]));
	// //height = area * 2 / length[top];
	// //if ((height / length[top]) < LOW_MESH_QUALITY)
	// //{
	// //	vertices[t.v[top] - 1] =
	// //		(vertices[t.v[bottom1] - 1] * (length[bottom1] / (length[bottom1] + length[bottom2])) +
	// //			vertices[t.v[bottom2] - 1] * (length[bottom2] / (length[bottom1] + length[bottom2])));
	// //	//printf("Elimit: %d, %d;  ", t.v[top], t.v[top]);
	// //}
	// //else
	// //{
	// //}
	// if (tris_map.find(ti) == tris_map.end())
	// {
	// 	trianglesNum++;
	// 	tris_map[ti] = trianglesNum;
	// }
	Triangle<int> tv(t0, t1, t2);
	if (tris_map.find(tv) == tris_map.end())
	{
		trianglesNum++;
		tris_map[tv] = trianglesNum;
		tris.push_back(tv);
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
			tmp_vec_indices[i] = insert_vert(pos + IcosaTable[i] * radius);
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
			tmp_vec_indices[i] = insert_vert(splash_particles[spi] + IcosaTable[i] * splash_radius[spi]);
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
