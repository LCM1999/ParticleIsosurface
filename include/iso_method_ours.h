#pragma once

#include <float.h>
#include <Eigen/Dense>
#include "iso_common.h"
#include "qefnorm.h"
#include "cube_arrays.h"
#include "utils.h"
#include "evaluator.h"
#include "index.h"

typedef Eigen::Vector<float, 5> Vector5f;
typedef Eigen::Vector<double, 5> Vector5d;
typedef Eigen::Vector<float, 6> Vector6f;
typedef Eigen::Vector<double, 6> Vector6d;
typedef Eigen::Vector<float, 7> Vector7f;
typedef Eigen::Vector<double, 7> Vector7d;
typedef Eigen::Matrix<float, 5, 5> Matrix5f;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<float, 7, 7> Matrix7f;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;

extern int tree_cells;

const short EMPTY = 0;
const short INTERNAL = 1;
const short LEAF = 2;
const short UNCERTAIN = 3;

const short CONTAIN = 0;
const short CONTAIN_IN = 1;
const short INTERSECT = 2;
const short DISJOINT = 3;

struct TNode
{
	TNode()
	{
		children[0] = children[1] = children[2] = children[3] = 
			children[4] = children[5] = children[6] = children[7] = 0;
		nId = tree_cells++;
		//tree_cells++;
		type = UNCERTAIN;
	}

	TNode(int id)
	{
		children[0] = children[1] = children[2] = children[3] =
			children[4] = children[5] = children[6] = children[7] = 0;
		nId = id;
		//tree_cells++;
		type = UNCERTAIN;
	}

	~TNode()
	{
		defoliate();
	}

	Eigen::Vector3f center;
	float half_length;
	Eigen::Vector4f node = Eigen::Vector4f::Zero();

	short depth = 0;
	unsigned __int64 nId;
	short type;

	TNode *children[8];

	bool changeSignDMC(Eigen::Vector4f* verts);


	int getWeight()
	{
		switch (type)
		{
		case EMPTY:
			return 0;
		case LEAF:
			return 1;
		case INTERNAL:
		{
			int sum = 0;
			for (TNode* child: children)
			{
				sum += child->getWeight();
			}
			return sum;
		}
		default:
			printf("ERROR: Get Uncertain Node During getWeight\n");
			exit(1);
		}
	}

	void defoliate();

	bool is_leaf()
	{
		return type == LEAF || type == EMPTY;
	}

	template <class T, class U>
	static double calcError(T& p, U& plane_norms, U& plane_pts)
	{
		assert(plane_norms.size() > 0);
		double err = 0;
		int plane_num = plane_norms.size();
		for (int i = 0; i < plane_norms.size(); i++)
		{
			if (((Eigen::Vector3f&)plane_norms[i]).isNan())
			{
				plane_num--;
				continue;
			}
			err += squared(p[3] - (plane_norms[i] * (p - plane_pts[i]))) / (1 + plane_norms[i].squaredNorm());
		}
		return err;
	}

	template <class T>
	static auto squared(const T& t)
	{
		return t * t;
	}

	template <class T, class U, class V>
	static double calcErrorDMC(T& p, U& verts, V& verts_grad)
	{
		double err = 0;
		for (size_t i = 0; i < 8; i++)
		{
			err += squared(p[3] - verts[i][3] - verts_grad[i].dot((p - verts[i]).head(3)));
		}
		return err;
	}

	void vertAll(float& curv, bool& signchange, Eigen::Vector3f* grad, float& qef_error, bool pass_face, bool pass_edge)
	{
		bool origin_sign;
		signchange = false;

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};
		
		Eigen::Vector4f verts[8];
		for (Index i = 0; i < 8; i++)
		{
			verts[i][0] = center[0] + sign(i.x) * half_length;
			verts[i][1] = center[1] + sign(i.y) * half_length;
			verts[i][2] = center[2] + sign(i.z) * half_length;
		}

		const float cellsize = 2 * half_length;

		const float border = BORDER * cellsize;

		float sampling_step = P_RADIUS / 2;

		int oversample = int(ceil(cellsize / sampling_step) + 1);

		if (depth < DEPTH_MIN)
		{
			oversample = OVERSAMPLE_QEF;
		}

		bool is_out;
		double err;

		std::vector<Eigen::Vector3f> sample_points;
		std::vector<float> field_scalars;
		std::vector<Eigen::Vector3f> field_gradient;

		for (int z = 0; z <= oversample; z++)
		{
			for (int y = 0; y <= oversample; y++)
			{
				for (int x = 0; x <= oversample; x++)
				{
					sample_points.push_back(
						Eigen::Vector3f(
							(1 - float(x) / oversample) * verts[0][0] + (float(x) / oversample) * verts[7][0],
							(1 - float(y) / oversample) * verts[0][1] + (float(y) / oversample) * verts[7][1],
							(1 - float(z) / oversample) * verts[0][2] + (float(z) / oversample) * verts[7][2])
						);
				}
			}
		}

		evaluator->GridEval(sample_points, field_scalars, field_gradient, signchange, oversample);

		for (Index i = 0; i < 8; i++)
		{
			verts[i][3] = field_scalars[i.x * oversample + i.y * oversample * (oversample + 1) + i.z * oversample * (oversample + 1) * (oversample + 1)];
		}

		// calculate curvature
		Eigen::Vector3f norms(0, 0, 0);
		float area = 0;
		for (Eigen::Vector3f n : field_gradient)
		{
			norms += n;
			area += n.norm();
		}
		curv = norms.norm() / area;

		/*--------------------VERT NODE-----------------------*/
		QEFNormal<double, 4> node_q;
		node_q.zero();

		Eigen::Vector4f node_mid = Eigen::Vector4f::Zero();
		std::vector<Eigen::Vector3f> node_plane_norms, node_plane_pts;

		int node_index;
		for (int z = 0; z <= oversample; z++)
		{
			for (int y = 0; y <= oversample; y++)
			{
				for (int x = 0; x <= oversample; x++)
				{
					node_index = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
					Eigen::Vector3f p(sample_points[node_index][0], sample_points[node_index][1], sample_points[node_index][2]);
					Vector5f pl = Vector5f::Zero();

					pl[0] = field_gradient[node_index][0];
					pl[1] = field_gradient[node_index][1];
					pl[2] = field_gradient[node_index][2];
					pl[3] = -1;
					pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + field_scalars[node_index];

					node_q.combineSelf(Vector5d(pl.cast<double>()).data());

					node_mid += Eigen::Vector4f(sample_points[node_index][0], sample_points[node_index][1], sample_points[node_index][2], field_scalars[node_index]);

					node_plane_pts.push_back(p);
					node_plane_norms.push_back(Eigen::Vector3f(pl[0], pl[1], pl[2]));
				}
			}
		}
		node_mid /= (oversample + 1) * (oversample + 1) * (oversample + 1);

		// build system to solve
		const int node_n = 4;
		Eigen::Matrix4d node_A = Eigen::Matrix4d::Zero();
		double node_B[node_n];

		for (int i = 0; i < node_n; i++)
		{
			int index = ((2 * node_n + 3 - i) * i) / 2;
			for (int j = i; j < node_n; j++)
			{
				node_A(i, j) = node_q.data[index + j - i];
				node_A(j, i) = node_A(i, j);
			}
			node_B[i] = -node_q.data[index + node_n - i];
		}

		// minimize QEF constrained to cell

		is_out = true;
		err = 1e30;
		Eigen::Vector3f node_mine(verts[0][0] + border, verts[0][1] + border, verts[0][2] + border);
		Eigen::Vector3f node_maxe(verts[7][0] - border, verts[7][1] - border, verts[7][2] - border);

		Eigen::Vector4f pc = Eigen::Vector4f::Zero();
		Eigen::Vector3f pcg = Eigen::Vector3f::Zero();

		for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
		{
			if (cell_dim == 3)
			{
				// find minimal point
				Eigen::Vector4d rvalue = Eigen::Vector4d::Zero();
				Eigen::Matrix4d inv = node_A.inverse();

				for (int i = 0; i < node_n; i++)
				{
					rvalue[i] = 0;
					for (int j = 0; j < node_n; j++)
						rvalue[i] += inv(j, i) * node_B[j];
				}

				pc << rvalue[0], rvalue[1], rvalue[2];
				evaluator->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);

				// check bounds
				if (pc[0] >= node_mine[0] && pc[0] <= node_maxe[0] &&
					pc[1] >= node_mine[1] && pc[1] <= node_maxe[1] &&
					pc[2] >= node_mine[2] && pc[2] <= node_maxe[2])
				{
					is_out = false;
					err = calcErrorDMC(pc, verts, grad);
					node << pc;
				}
			}
			else if (cell_dim == 2)
			{
				for (int face = 0; face < 6; face++)
				{
					int dir = face / 2;
					int side = face % 2;
					Eigen::Vector3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					Matrix5d AC = Matrix5d::Zero();
					double BC[node_n + 1];
					for (int i = 0; i < node_n + 1; i++)
					{
						for (int j = 0; j < node_n + 1; j++)
						{
							AC(i, j) = (i < node_n&& j < node_n ? node_A(i, j) : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					AC(node_n, dir) = AC(dir, node_n) = 1;
					BC[node_n] = corners[side][dir];

					// find minimal point
					double rvalue[node_n + 1];
					Matrix5d inv = AC.inverse();

					for (int i = 0; i < node_n + 1; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 1; j++)
							rvalue[i] += inv(j, i) * BC[j];
					}

					pc << rvalue[0], rvalue[1], rvalue[2];
					evaluator->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);

					// check bounds
					int dp = (dir + 1) % 3;
					int dpp = (dir + 2) % 3;
					if (pc[dp] >= node_mine[dp] && pc[dp] <= node_maxe[dp] &&
						pc[dpp] >= node_mine[dpp] && pc[dpp] <= node_maxe[dpp])
					{
						is_out = false;
						
						double e = calcErrorDMC(pc, verts, grad);

						if (e < err)
						{
							err = e;
							node << pc;
						}
					}
				}
			}
			else if (cell_dim == 1)
			{
				for (int edge = 0; edge < 12; edge++)
				{
					int dir = edge / 4;
					int side = edge % 4;
					Eigen::Vector3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					Matrix6d AC = Matrix6d::Zero();
					double BC[node_n + 2];
					for (int i = 0; i < node_n + 2; i++)
					{
						for (int j = 0; j < node_n + 2; j++)
						{
							AC(i, j) = (i < node_n&& j < node_n ? node_A(i, j) : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					int dp = (dir + 1) % 3;
					int dpp = (dir + 2) % 3;
					AC(node_n, dp) = AC(dp, node_n) = 1;
					AC(node_n + 1, dpp) = AC(dpp, node_n + 1) = 1;
					BC[node_n] = corners[side & 1][dp];
					BC[node_n + 1] = corners[side >> 1][dpp];

					// find minimal point
					double rvalue[node_n + 2];
					Matrix6d inv = AC.inverse();

					for (int i = 0; i < node_n + 2; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 2; j++)
							rvalue[i] += inv(j, i) * BC[j];
					}

					pc << rvalue[0], rvalue[1], rvalue[2];
					evaluator->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);

					// check bounds
					if (pc[dir] >= node_mine[dir] && pc[dir] <= node_maxe[dir])
					{
						is_out = false;

						double e = calcErrorDMC(pc, verts, grad);

						if (e < err)
						{
							err = e;
							node << pc;
						}
					}
				}
			}
			else if (cell_dim == 0)
			{
				for (int vertex = 0; vertex < 8; vertex++)
				{
					Eigen::Vector3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					Matrix7d AC = Matrix7d::Zero();
					double BC[node_n + 3];
					for (int i = 0; i < node_n + 3; i++)
					{
						for (int j = 0; j < node_n + 3; j++)
						{
							AC(i, j) = (i < node_n&& j < node_n ? node_A(i, j) : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					for (int i = 0; i < 3; i++)
					{
						AC(node_n + i, i) = AC(i, node_n + i) = 1;
						BC[node_n + i] = corners[(vertex >> i) & 1][i];
					}
					// find minimal point
					double rvalue[node_n + 3];
					Matrix7d inv = AC.inverse();

					for (int i = 0; i < node_n + 3; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 3; j++)
							rvalue[i] += inv(j, i) * BC[j];
					}

					pc << rvalue[0], rvalue[1], rvalue[2];
					evaluator->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);
					// check bounds
					double e = calcErrorDMC(pc, verts, grad);

					if (e < err)
					{
						err = e;
						node << pc;
					}
				}
			}
		}
		evaluator->SingleEval(node.head(3), node[3], grad[8]);

			/*
		if ((node[0] < verts[0][0] || node[0] > verts[7][0] ||
			node[1] < verts[0][1] || node[1] > verts[7][1] ||
			node[2] < verts[0][2] || node[2] > verts[7][2] 
			))
		{
			printf("Out of node. \n");
			printf("%f %f %f %f\n", node[0], node[1], node[2], node[3]);
			printf("%f %f %f %f\n", pc[0], pc[1], pc[2], pc[3]);
			if (depth <= DEPTH_MIN)
			{
				node = (verts[0] + verts[7]) / 2;
				evaluator->SingleEval(node.head(3), node[3], grad[8]);
				return;
			}
			double minError = 1e30;
			int index;
			int sample_num = std::max((int(cellsize / (P_RADIUS / 2)) + 1), oversample);
			std::vector<Eigen::Vector3f> errors_samples;
			std::vector<float> errors_scalar;
			std::vector<float> errors_value;
			std::vector<Eigen::Vector3f> errors_grad;
			if (((sample_num == oversample)) || abs(node[3]) >= ISO_VALUE)
			{
				sample_num = oversample;
				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_samples = sample_points;
							errors_scalar = field_scalars;
							errors_value.push_back(
								calcErrorDMC(
									Eigen::Vector4f(
										errors_samples[index][0],
										errors_samples[index][1],
										errors_samples[index][2],
										errors_scalar[index]), 
									verts, grad));
							errors_grad = field_gradient;
						}
					}
				}
			}
			else
			{
				printf("Out of node. ");
				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_samples.push_back(Eigen::Vector3f(
								(1 - float(x) / sample_num) * verts[0][0] + (float(x) / sample_num) * verts[7][0],
								(1 - float(y) / sample_num) * verts[0][1] + (float(y) / sample_num) * verts[7][1],
								(1 - float(z) / sample_num) * verts[0][2] + (float(z) / sample_num) * verts[7][2]));
						}
					}
				}

				evaluator->GridEval(errors_samples, errors_scalar, errors_grad, signchange, sample_num);

				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_value.push_back(
								calcErrorDMC(
									Eigen::Vector4f(
										errors_samples[index][0],
										errors_samples[index][1],
										errors_samples[index][2],
										errors_scalar[index]),
									verts, grad));
						}
					}
				}
				norms.setZero();
				area = 0;
				for (Eigen::Vector3f n : errors_grad)
				{
					norms += n;
					area += n.norm();
				}
				curv = norms.norm() / area;
			}
			int minIndex = std::distance(errors_value.begin(),
				std::min_element(errors_value.begin(), errors_value.end()));
			node << errors_samples[minIndex];
			grad[8] = errors_grad[minIndex];
		}
			*/

		qef_error += err;
	}

	int CountLeaves()
	{
		if (type == LEAF || type == EMPTY)
		{
			return 1;
		}
		int count = 0;
		for (TNode* child : children)
		{
			count += child->CountLeaves();
		}
		return count;
	}

};


const int edgevmap[12][2] = { {0,4},{1,5},{2,6},{3,7},{0,2},{1,3},{4,6},{5,7},{0,1},{2,3},{4,5},{6,7} };
const int edgemask[3] = { 5, 3, 6 };

// direction from parent st to each of the eight child st
// st is the corner of the cube with minimum (x,y,z) coordinates
//const int vertMap[8][3] = { {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1} };
const int vertMap[8][3] = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };

// map from the 6 faces of the cube to the 4 vertices that bound the face
const int faceMap[6][4] = { {4, 8, 5, 9}, {6, 10, 7, 11},{0, 8, 1, 10},{2, 9, 3, 11},{0, 4, 2, 6},{1, 5, 3, 7} };

// first used by cellProcCount()
// used in cellProcContour(). 
// between 8 child-nodes there are 12 faces.
// first two numbers are child-pairs, to be processed by faceProcContour()
// the last number is "dir" ?
const int cellProcFaceMask[12][3] = { {0,4,0},{1,5,0},{2,6,0},{3,7,0},{0,2,1},{4,6,1},{1,3,1},{5,7,1},{0,1,2},{2,3,2},{4,5,2},{6,7,2} };


// then used in cellProcContour() when calling edgeProc()
// between 8 children there are 6 common edges
// table lists the 4 children that share the edge
// the last number is "dir" ?
const int cellProcEdgeMask[6][5] = { {0,1,2,3,0},{4,5,6,7,0},{0,4,1,5,1},{2,6,3,7,1},{0,2,4,6,2},{1,3,5,7,2} };

// usde by faceProcCount()
const int faceProcFaceMask[3][4][3] = {
	{{4,0,0},{5,1,0},{6,2,0},{7,3,0}},
	{{2,0,1},{6,4,1},{3,1,1},{7,5,1}},
	{{1,0,2},{3,2,2},{5,4,2},{7,6,2}}
};
const int faceProcEdgeMask[3][4][6] = {
	{{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
	{{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
	{{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}}
};
const int edgeProcEdgeMask[3][2][5] = {
	{{3,2,1,0,0},{7,6,5,4,0}},
	{{5,1,4,0,1},{7,3,6,2,1}},
	{{6,4,2,0,2},{7,5,3,1,2}},
};
const int processEdgeMask[3][4] = { {3,2,1,0},{7,5,6,4},{11,10,9,8} };

const int dirCell[3][4][3] = {
	{{0,-1,-1},{0,-1,0},{0,0,-1},{0,0,0}},
	{{-1,0,-1},{-1,0,0},{0,0,-1},{0,0,0}},
	{{-1,-1,0},{-1,0,0},{0,-1,0},{0,0,0}}
};
const int dirEdge[3][4] = {
	{3,2,1,0},
	{7,6,5,4},
	{11,10,9,8}
};


void gen_iso_ours();
