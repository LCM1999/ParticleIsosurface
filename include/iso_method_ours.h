#pragma once

#include <float.h>
#include <Eigen/Dense>
#include "utils.h"
#include "index.h"
#include <math.h>

class SurfReconstructor;

typedef Eigen::Vector<float, 5> Vector5f;
typedef Eigen::Vector<float, 5> Vector5f;
typedef Eigen::Vector<float, 6> Vector6f;
typedef Eigen::Vector<float, 6> Vector6f;
typedef Eigen::Vector<float, 7> Vector7f;
typedef Eigen::Vector<float, 7> Vector7f;
typedef Eigen::Matrix<float, 5, 5> Matrix5f;
typedef Eigen::Matrix<float, 5, 5> Matrix5f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 7, 7> Matrix7f;
typedef Eigen::Matrix<float, 7, 7> Matrix7f;

extern int tree_cells;

const char EMPTY = 0;
const char INTERNAL = 1;
const char LEAF = 2;
const char UNCERTAIN = 3;

struct TNode
{
	TNode() {}

	TNode(SurfReconstructor* surf_constructor, unsigned long long id);

	TNode(SurfReconstructor* surf_constructor, TNode* parent, Index i);

	~TNode()
	{
		defoliate();
	}

	SurfReconstructor* constructor;

	Eigen::Vector3f center;
	float half_length;
	Eigen::Vector4f node = Eigen::Vector4f::Zero();

	char depth = 0;
	char type;
	unsigned long long id;

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

	bool contain(const Eigen::Vector3f& pos)
	{
		if (std::abs(pos[0] - center[0]) <= half_length &&
			std::abs(pos[1] - center[1]) <= half_length &&
			std::abs(pos[2] - center[2]) <= half_length)
		{
			return true;
		}
		return false;
	}


	template <class T>
	static auto squared(const T& t)
	{
		return t * t;
	}

	float calcErrorDMC(
		Eigen::Vector4f p, float* verts, float* verts_grad, const int oversample);

	void vertAll(float& curv, bool& signchange, float& qef_error, float& sample);

	void GenerateSampling(float* sample_points);

	void NodeSampling(
		float& curv, bool& signchange, float cellsize,
		float* sample_points, float* sample_grads);

	void NodeCalcNode(float* sample_points, float* sample_grads, float cellsize);

	// void NodeFeatureCalc();

	// void NodeErrorMinimize();

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
