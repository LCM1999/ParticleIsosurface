#include "global.h"
#include "iso_method_ours.h"

Mesh::Mesh(const float p_radius, const float mesh_tolerance)
{
	P_RADIUS = p_radius;
	MESH_TOLERANCE = mesh_tolerance;
	reset();
	BuildIcosaTable();
}

int Mesh::insert_vert(const Eigen::Vector3f& p)
{
	if (abs(p[2]) > 100.0 || abs(p[1]) > 100.0 || abs(p[0]) > 100.0)
	{
		printf("");
	}
	vect3i tmp = vect3f2vect3i(p);
	if (vertices_map.find(tmp) == vertices_map.end())
	{
		verticesNum++;
		vertices_map[tmp] = verticesNum;
		vertices.push_back(p);
		if (abs(p[2]) > 100.0 || abs(p[1]) > 100.0 || abs(p[0]) > 100.0)
		{
			printf("");
		}
	}
	return vertices_map[tmp];
}

vect3i Mesh::vect3f2vect3i(const Eigen::Vector3f& a)
{
	vect3i r;
	for (size_t i = 0; i < 3; i++)
	{
		r[i] = int(round(a[i] * MESH_TOLERANCE));
	}
	return r;
}

Eigen::Vector3f Mesh::vect3i2vect3f(const vect3i& a)
{
	Eigen::Vector3f r;
	for (size_t i = 0; i < 3; i++)
	{
		r[i] = a.v[i] / MESH_TOLERANCE;
	}
	return r;
}

bool Mesh::similiar_point(Eigen::Vector3f& v1, Eigen::Vector3f& v2)
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

void Mesh::insert_tri(int t0, int t1, int t2)
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

	IcosaTable[0] = Eigen::Vector3f(0, 0, P_RADIUS);
	int i1, i2;

	for (size_t i = 1; i <= 5; i++)
	{
		i1 = i;
		i2 = i + 5;
		z = P_RADIUS * std::sin(V_ANGLE);
		xy = P_RADIUS * std::cos(V_ANGLE);

		IcosaTable[i1] = Eigen::Vector3f(xy * cos(hAngle1), xy * sin(hAngle1), z);
		IcosaTable[i2] = Eigen::Vector3f(xy * cos(hAngle2), xy * sin(hAngle2), -z);

		hAngle1 += H_ANGLE;
		hAngle2 += H_ANGLE;
	}

	IcosaTable[11] = Eigen::Vector3f(0, 0, -P_RADIUS);	
}

void Mesh::AppendSplash(std::vector<Eigen::Vector3f>& splash_particles)
{
	std::vector<int> tmp_vec_indices;
	for (const Eigen::Vector3f& pos : splash_particles)
	{
		tmp_vec_indices.clear();
		tmp_vec_indices.resize(12);
		for (size_t i = 0; i < 12; i++)
		{
			tmp_vec_indices[i] = insert_vert(pos + IcosaTable[i]);
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

Graph::Graph(std::vector<TNode*>& layer_nodes, int vwn)
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

