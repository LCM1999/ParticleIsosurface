#pragma once

#include <float.h>
#include <Eigen/Dense>
#include "utils.h"
#include "index.h"
#include <math.h>

class SurfReconstructor;

typedef Eigen::Vector<double, 5> Vector5d;
typedef Eigen::Vector<double, 5> Vector5d;
typedef Eigen::Vector<double, 6> Vector6d;
typedef Eigen::Vector<double, 6> Vector6d;
typedef Eigen::Vector<double, 7> Vector7d;
typedef Eigen::Vector<double, 7> Vector7d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;

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

	Eigen::Vector3d center;
	double half_length;
	Eigen::Vector4d node = Eigen::Vector4d::Zero();

	char depth = 0;
	char type;
	unsigned long long id;

	TNode *children[8];

	bool changeSignDMC(Eigen::Vector4d* verts);

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

	bool contain(const Eigen::Vector3d& pos)
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

	double calcErrorDMC(
		Eigen::Vector4d p, double* verts, double* verts_grad, const int oversample);

	void vertAll(double& curv, bool& signchange, double& qef_error, double& sample);

	void GenerateSampling(double* sample_points);

	void NodeSampling(
		double& curv, bool& signchange, double cellsize,
		double* sample_points, double* sample_grads);

	void NodeCalcNode(double* sample_points, double* sample_grads, double cellsize);

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
