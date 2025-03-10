#include <math.h>
#include <stdlib.h>

#include "iso_common.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "timer.h"
#include "qefnorm.h"
#include "evaluator.h"

static float sign(unsigned int x)
{
	return x ? 1 : -1;
};

TNode::TNode(SurfReconstructor* surf_constructor, unsigned long long id)
{
	constructor = surf_constructor;
	children[0] = children[1] = children[2] = children[3] =
		children[4] = children[5] = children[6] = children[7] = 0;
	type = UNCERTAIN;
	this->id = id;
}

TNode::TNode(SurfReconstructor* surf_constructor, std::shared_ptr<TNode> parent, Index i)
{
	constructor = surf_constructor;
	depth = parent->depth + 1;
	half_length = parent->half_length / 2;
	center = parent->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * half_length);
	node[0] = center[0];
	node[1] = center[1];
	node[2] = center[2];
	children[0] = children[1] = children[2] = children[3] =
		children[4] = children[5] = children[6] = children[7] = 0;
	type = UNCERTAIN;
	this->id = parent->id * 8 + i.v;
}

float TNode::calcErrorDMC(Eigen::Vector4f p, float* verts, float* verts_grad, const int oversample)
{
	float err = 0;
	for (size_t i = 0; i < pow(oversample + 1, 3); i++)
	{
		Eigen::Vector4f v(verts[i * 4 + 0], verts[i * 4 + 1], verts[i * 4 + 2], verts[i * 4 + 3]);
		Eigen::Vector3f g(verts_grad[i * 3 + 0], verts_grad[i * 3 + 1], verts_grad[i * 3 + 2]);
		err += squared(g.dot((p - v).head(3)) - p[3]) / (1 + g.squaredNorm());
	}
	return err;
}

void TNode::GenerateSampling(
	float* sample_points)
{
	const float cellsize = 2 * half_length;
	Eigen::Vector3f minV(center[0] - half_length, center[1] - half_length, center[2] - half_length);
	Eigen::Vector3f maxV(center[0] + half_length, center[1] + half_length, center[2] + half_length);

	for (int z = 0; z <= constructor->getOverSampleQEF(); z++)
	{
		for (int y = 0; y <= constructor->getOverSampleQEF(); y++)
		{
			for (int x = 0; x <= constructor->getOverSampleQEF(); x++)
			{
				sample_points[(z * (constructor->getOverSampleQEF()+1) * (constructor->getOverSampleQEF()+1) + y * (constructor->getOverSampleQEF()+1) + x) * 4 + 0] = 
				(1 - float(x) / constructor->getOverSampleQEF()) * minV[0] + (float(x) / constructor->getOverSampleQEF()) * maxV[0];
				sample_points[(z * (constructor->getOverSampleQEF()+1) * (constructor->getOverSampleQEF()+1) + y * (constructor->getOverSampleQEF()+1) + x) * 4 + 1] = 
				(1 - float(y) / constructor->getOverSampleQEF()) * minV[1] + (float(y) / constructor->getOverSampleQEF()) * maxV[1];
				sample_points[(z * (constructor->getOverSampleQEF()+1) * (constructor->getOverSampleQEF()+1) + y * (constructor->getOverSampleQEF()+1) + x) * 4 + 2] = 
				(1 - float(z) / constructor->getOverSampleQEF()) * minV[2] + (float(z) / constructor->getOverSampleQEF()) * maxV[2];
				sample_points[(z * (constructor->getOverSampleQEF()+1) * (constructor->getOverSampleQEF()+1) + y * (constructor->getOverSampleQEF()+1) + x) * 4 + 3] = 0;
			}
		}
	}
}

void TNode::NodeSampling(
	float& curv, bool& signchange, float cellsize,
	float* sample_points, float* sample_grads)
{
	bool origin_sign;
	signchange = false;
	constructor->getEvaluator()->GridEval(sample_points, sample_grads, cellsize, signchange, constructor->getOverSampleQEF(), false);
	// Eigen::Vector3f norms(0, 0, 0);
	// float area = 0;
	// for (int i = 0; i < pow(constructor->getOverSampleQEF() + 1, 3); i++)
	// {
	// 	Eigen::Vector3f n(
	// 		sample_grads[i * 3 + 0], 
	// 		sample_grads[i * 3 + 1], 
	// 		sample_grads[i * 3 + 2]);
	// 	// n.normalize();
	// 	norms += n;
	// 	area += n.norm();
	// }

	// float field_curv = (area == 0) ? 1.0 : (norms.norm() / area);
	// curv = std::min(curv, field_curv);
}

void TNode::NodeCalcNode(float* sample_points, float* sample_grads, float cellsize)
{
	const float border = constructor->getBorder() * cellsize;
	Eigen::Vector3f minV(center[0] - half_length + border, center[1] - half_length + border, center[2] - half_length + border);
	Eigen::Vector3f maxV(center[0] + half_length - border, center[1] + half_length - border, center[2] + half_length - border);
	QEFNormal<float, 4> node_q;
	node_q.zero();
	std::vector<Eigen::Vector3f> node_plane_norms, node_plane_pts;
	int node_index;
	for (int z = 0; z <= constructor->getOverSampleQEF(); z++)
	{
		for (int y = 0; y <= constructor->getOverSampleQEF(); y++)
		{
			for (int x = 0; x <= constructor->getOverSampleQEF(); x++)
			{
				node_index = (z * (constructor->getOverSampleQEF() + 1) * (constructor->getOverSampleQEF() + 1) + y * (constructor->getOverSampleQEF() + 1) + x);
				Eigen::Vector3f p(
					sample_points[node_index * 4 + 0], 
					sample_points[node_index * 4 + 1], 
					sample_points[node_index * 4 + 2]);
				Vector5f pl = Vector5f::Zero();
				pl[0] = sample_grads[node_index * 3 + 0];
				pl[1] = sample_grads[node_index * 3 + 1];
				pl[2] = sample_grads[node_index * 3 + 2];
				pl[3] = -1;
				pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + sample_points[node_index * 4 + 3];
				node_q.combineSelf(Vector5f(pl).data());
				node_plane_pts.push_back(p);
				node_plane_norms.push_back(Eigen::Vector3f(pl[0], pl[1], pl[2]));
			}
		}
	}
	// build system to solve
	const int node_n = 4;
	Eigen::Matrix4f node_A = Eigen::Matrix4f::Zero();
	float node_B[node_n];
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
	// find minimal point
	Eigen::Vector4f rvalue = Eigen::Vector4f::Zero();
	Eigen::Matrix4f inv = node_A.inverse();
	for (int i = 0; i < node_n; i++)
	{
		rvalue[i] = 0;
		for (int j = 0; j < node_n; j++)
			rvalue[i] += inv(j, i) * node_B[j];
	}
	pc << rvalue[0], rvalue[1], rvalue[2], rvalue[3];
	if (pc[0] >= minV[0] && pc[0] <= maxV[0] &&
		pc[1] >= minV[1] && pc[1] <= maxV[1] &&
		pc[2] >= minV[2] && pc[2] <= maxV[2])
	{
		is_out = false;
		err = calcErrorDMC(pc, sample_points, sample_grads, constructor->getOverSampleQEF());
		node << pc;
	}
	constructor->getEvaluator()->SingleEval(node.head(3), node[3]);
	if (err/cellsize < 0.0001)
	{
		node[3] = 0.0;
	}
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


