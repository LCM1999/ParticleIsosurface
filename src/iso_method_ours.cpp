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

TNode::TNode(SurfReconstructor* surf_constructor)
{
	constructor = surf_constructor;
	children[0] = children[1] = children[2] = children[3] =
		children[4] = children[5] = children[6] = children[7] = 0;
	type = UNCERTAIN;
}

double TNode::calcErrorDMC(Eigen::Vector4f p, std::vector<Eigen::Vector4f>& verts, std::vector<Eigen::Vector3f>& verts_grad)
{
	double err = 0;
	for (size_t i = 0; i < verts.size(); i++)
	{
		err += squared(p[3] - verts_grad[i].dot((p - verts[i]).head(3))) / (1 + verts_grad[i].squaredNorm());
	}
	return err;
}

void TNode::vertAll(float& curv, bool& signchange, float& qef_error, float& sample_radius)
{
	bool origin_sign;
	signchange = false;
	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	double sampling_time = 0, features_time = 0, error_time = 0, temp_time = 0;

	Eigen::Vector3f verts[8];
	
	for (Index i = 0; i < 8; i++)
	{
		verts[i][0] = center[0] + sign(i.x) * half_length;
		verts[i][1] = center[1] + sign(i.y) * half_length;
		verts[i][2] = center[2] + sign(i.z) * half_length;
	}
	const float cellsize = 2 * half_length;
	const float border = constructor->getBorder() * cellsize;
	float sampling_step = sample_radius;
	int oversample = int(ceil(cellsize / sampling_step) + 2);
	if (depth < constructor->getDepthMin())
	{
		oversample = constructor->getOverSampleQEF();
	}
	bool is_out;
	double err;
	std::vector<Eigen::Vector4f> sample_points;
	std::vector<Eigen::Vector3f> field_gradient;
	for (int z = 0; z <= oversample; z++)
	{
		for (int y = 0; y <= oversample; y++)
		{
			for (int x = 0; x <= oversample; x++)
			{
				sample_points.push_back(
					Eigen::Vector4f(
						(1 - float(x) / oversample) * verts[0][0] + (float(x) / oversample) * verts[7][0],
						(1 - float(y) / oversample) * verts[0][1] + (float(y) / oversample) * verts[7][1],
						(1 - float(z) / oversample) * verts[0][2] + (float(z) / oversample) * verts[7][2],
						0));
			}
		}
	}
	sampling_time = get_time();
	constructor->getEvaluator()->GridEval(sample_points, field_gradient, signchange, oversample, false);
	sampling_time = get_time() - sampling_time;
	// for (Index i = 0; i < 8; i++)
	// {
	// 	verts[i][3] = sample_points[i.x * oversample + i.y * oversample * (oversample + 1) + i.z * oversample * (oversample + 1) * (oversample + 1)][3];
	// }
	// calculate curvature
	Eigen::Vector3f norms(0, 0, 0);
	float area = 0;
	for (Eigen::Vector3f n : field_gradient)
	{
		n.normalize();
		norms += n;
		area += n.norm();
	}
	// if (curv == 0)
	// {
	float field_curv = (norms.norm() / area);
	if (!std::isnan(field_curv))
	{
		curv += field_curv;
		curv /= 2;
	}
	// }
	// else if (curv < 0) {
	// 	curv = 0;
	// } 
	// else {
	// 	curv = std::min(norms.norm() / area, curv);
	// }
	
	/*--------------------VERT NODE-----------------------*/
	temp_time = get_time();
	QEFNormal<double, 4> node_q;
	node_q.zero();
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
				pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + sample_points[node_index][3];
				node_q.combineSelf(Vector5d(pl.cast<double>()).data());
				node_plane_pts.push_back(p);
				node_plane_norms.push_back(Eigen::Vector3f(pl[0], pl[1], pl[2]));
			}
		}
	}
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
	features_time += get_time() - temp_time;
	for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
	{
		if (cell_dim == 3)
		{
			// find minimal point
			temp_time = get_time();
			Eigen::Vector4d rvalue = Eigen::Vector4d::Zero();
			Eigen::Matrix4d inv = node_A.inverse();
			for (int i = 0; i < node_n; i++)
			{
				rvalue[i] = 0;
				for (int j = 0; j < node_n; j++)
					rvalue[i] += inv(j, i) * node_B[j];
			}
			pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
			features_time += get_time() - temp_time;
			// check bounds
			if (pc[0] >= node_mine[0] && pc[0] <= node_maxe[0] &&
				pc[1] >= node_mine[1] && pc[1] <= node_maxe[1] &&
				pc[2] >= node_mine[2] && pc[2] <= node_maxe[2])
			{
				is_out = false;
				temp_time = get_time();
				err = calcErrorDMC(pc, sample_points, field_gradient);
				error_time += get_time() - temp_time;
				node << pc;
			}
		}
		else if (cell_dim == 2)
		{
			for (int face = 0; face < 6; face++)
			{
				temp_time = get_time();
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				features_time += get_time() - temp_time;
				// check bounds
				int dp = (dir + 1) % 3;
				int dpp = (dir + 2) % 3;
				if (pc[dp] >= node_mine[dp] && pc[dp] <= node_maxe[dp] &&
					pc[dpp] >= node_mine[dpp] && pc[dpp] <= node_maxe[dpp])
				{
					is_out = false;
					temp_time = get_time();
					double e = calcErrorDMC(pc, sample_points, field_gradient);
					error_time += get_time() - temp_time;
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
				temp_time = get_time();
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				features_time += get_time() - temp_time;
				// check bounds
				if (pc[dir] >= node_mine[dir] && pc[dir] <= node_maxe[dir])
				{
					is_out = false;
					temp_time = get_time();
					double e = calcErrorDMC(pc, sample_points, field_gradient);
					error_time += get_time() - temp_time;
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
				temp_time = get_time();
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				features_time += get_time() - temp_time;
				// check bounds
				temp_time = get_time();
				double e = calcErrorDMC(pc, sample_points, field_gradient);
				error_time += get_time() - temp_time;
				if (e < err)
				{
					err = e;
					node << pc;
				}
			}
		}
	}
	// printf("%f %f %f\n", sampling_time, features_time, error_time);
	constructor->getEvaluator()->SingleEval(node.head(3), node[3], pcg);
	qef_error += err;
}

void TNode::GenerateSampling(
	std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
	int& oversample, const float sample_radius)
{
	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	const float cellsize = 2 * half_length;
	const float border = constructor->getBorder() * cellsize;
	Eigen::Vector3f minV(center[0] - half_length + border, center[1] - half_length + border, center[2] - half_length + border);
	Eigen::Vector3f maxV(center[0] + half_length - border, center[1] + half_length - border, center[2] + half_length - border);

	oversample = int(ceil(cellsize / sample_radius) + 2);
	if (depth < constructor->getDepthMin())
	{
		oversample = constructor->getOverSampleQEF();
	}
	for (int z = 0; z <= oversample; z++)
	{
		for (int y = 0; y <= oversample; y++)
		{
			for (int x = 0; x <= oversample; x++)
			{
				sample_points.push_back(
					Eigen::Vector4f(
						(1 - float(x) / oversample) * minV[0] + (float(x) / oversample) * maxV[0],
						(1 - float(y) / oversample) * minV[1] + (float(y) / oversample) * maxV[1],
						(1 - float(z) / oversample) * minV[2] + (float(z) / oversample) * maxV[2],
						0));
			}
		}
	}
}

void TNode::NodeSampling(
	float& curv, bool& signchange, 
	std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
	const int oversample)
{
	bool origin_sign;
	signchange = false;
	constructor->getEvaluator()->GridEval(sample_points, sample_grads, signchange, oversample, false);

	Eigen::Vector3f norms(0, 0, 0);
	float area = 0;
	for (Eigen::Vector3f n : sample_grads)
	{
		n.normalize();
		norms += n;
		area += n.norm();
	}

	float field_curv = (area == 0) ? 0.0 : (norms.norm() / area);
	if (field_curv == 0.0)
	{
		return;
	} else if (curv == 0.0) {
		curv = field_curv;
	} else {
		curv = std::min(curv, field_curv);
	}
}

void TNode::NodeCalcNode(
	std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
	const int oversample)
{
	const float cellsize = 2 * half_length;
	const float border = constructor->getBorder() * cellsize;
	Eigen::Vector3f minV(center[0] - half_length + border, center[1] - half_length + border, center[2] - half_length + border);
	Eigen::Vector3f maxV(center[0] + half_length - border, center[1] + half_length - border, center[2] + half_length - border);
	QEFNormal<double, 4> node_q;
	node_q.zero();
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
				pl[0] = sample_grads[node_index][0];
				pl[1] = sample_grads[node_index][1];
				pl[2] = sample_grads[node_index][2];
				pl[3] = -1;
				pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + sample_points[node_index][3];
				node_q.combineSelf(Vector5d(pl.cast<double>()).data());
				node_plane_pts.push_back(p);
				node_plane_norms.push_back(Eigen::Vector3f(pl[0], pl[1], pl[2]));
			}
		}
	}
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
	bool is_out = true;
	float err = 1e30;
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
			pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
			// constructor->getEvaluator()->SingleEval(pc.head(3), pc[3], pcg);
			// check bounds
			if (pc[0] >= minV[0] && pc[0] <= maxV[0] &&
				pc[1] >= minV[1] && pc[1] <= maxV[1] &&
				pc[2] >= minV[2] && pc[2] <= maxV[2])
			{
				is_out = false;
				err = calcErrorDMC(pc, sample_points, sample_grads);
				node << pc;
			}
		}
		else if (cell_dim == 2)
		{
			for (int face = 0; face < 6; face++)
			{
				int dir = face / 2;
				int side = face % 2;
				Eigen::Vector3f corners[2] = { minV, maxV };
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				// constructor->getEvaluator()->SingleEval(pc.head(3), pc[3], pcg);
				// check bounds
				int dp = (dir + 1) % 3;
				int dpp = (dir + 2) % 3;
				if (pc[dp] >= minV[dp] && pc[dp] <= maxV[dp] &&
					pc[dpp] >= minV[dpp] && pc[dpp] <= maxV[dpp])
				{
					is_out = false;
					double e = calcErrorDMC(pc, sample_points, sample_grads);
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
				Eigen::Vector3f corners[2] = { minV, maxV };
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				// constructor->getEvaluator()->SingleEval(pc.head(3), pc[3], pcg);
				// check bounds
				if (pc[dir] >= minV[dir] && pc[dir] <= maxV[dir])
				{
					is_out = false;
					double e = calcErrorDMC(pc, sample_points, sample_grads);
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
				Eigen::Vector3f corners[2] = { minV, maxV };
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
				pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
				// constructor->getEvaluator()->SingleEval(pc.head(3), pc[3], pcg);
				// check bounds
				double e = calcErrorDMC(pc, sample_points, sample_grads);
				if (e < err)
				{
					err = e;
					node << pc;
				}
			}
		}
	}
	constructor->getEvaluator()->SingleEval(node.head(3), node[3], pcg);
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

