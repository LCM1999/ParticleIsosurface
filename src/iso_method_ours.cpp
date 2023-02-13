#include <math.h>
#include <stdlib.h>

#include "iso_common.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "timer.h"
#include "qefnorm.h"
#include "evaluator.h"

int tree_cells = 0;
const std::string SAVE_RECORD_NAME = "./record_II.txt";

TNode::TNode(SurfReconstructor* surf_constructor, int id)
{
	constructor = surf_constructor;
	children[0] = children[1] = children[2] = children[3] =
		children[4] = children[5] = children[6] = children[7] = 0;
	nId = id;
	type = UNCERTAIN;
}


void TNode::vertAll(float& curv, bool& signchange, Eigen::Vector3f* grad, float& qef_error)
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
	const float border = constructor->getBorder() * cellsize;
	float sampling_step = constructor->getPRadius() / 2;
	int oversample = int(ceil(cellsize / sampling_step) + 1);
	if (depth < constructor->getDepthMin())
	{
		oversample = constructor->getOverSampleQEF();
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
	constructor->getEvaluator()->GridEval(sample_points, field_scalars, field_gradient, signchange, oversample);
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
	if (curv == 0)
	{
		curv = norms.norm() / area;
	}
	else if (curv < 0) {
		curv = 0;
	} 
	else {
		curv = std::min(norms.norm() / area, curv);
	}
	
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
			pc << rvalue[0], rvalue[1], rvalue[2], 0.0f;
			constructor->getEvaluator()->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);
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
				pc << rvalue[0], rvalue[1], rvalue[2], 0.0f;
				constructor->getEvaluator()->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);
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
				pc << rvalue[0], rvalue[1], rvalue[2], 0.0f;
				constructor->getEvaluator()->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);
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
				pc << rvalue[0], rvalue[1], rvalue[2], 0.0f;
				constructor->getEvaluator()->SingleEval((Eigen::Vector3f&)pc, pc[3], pcg);
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
	constructor->getEvaluator()->SingleEval(node.head(3), node[3], grad[8]);
	qef_error += err;
}

bool TNode::changeSignDMC(Eigen::Vector4f* verts)
{
	return  sign(verts[0]) != sign(verts[1]) ||
		sign(verts[0]) != sign(verts[2]) ||
		sign(verts[0]) != sign(verts[3]) ||
		sign(verts[0]) != sign(verts[4]) ||
		sign(verts[0]) != sign(verts[5]) ||
		sign(verts[0]) != sign(verts[6]) ||
		sign(verts[0]) != sign(verts[7]) ||
		sign(verts[0]) != sign(node);
}

void TNode::defoliate()
{
	for (int i = 0; i < 8; i++)
	{
		delete children[i];
		children[i] = 0;
	}
}

