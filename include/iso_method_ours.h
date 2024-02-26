#pragma once

#include <float.h>
#include <Eigen/Dense>
#include "utils.h"
#include "index.h"
#include <math.h>
#include <memory>

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

const char EMPTY = 0;
const char INTERNAL = 1;
const char LEAF = 2;
const char UNCERTAIN = 3;

struct TNode
{
	TNode() {}

	TNode(SurfReconstructor* surf_constructor, unsigned long long id);

	TNode(SurfReconstructor* surf_constructor, std::shared_ptr<TNode> parent, Index i);

	~TNode() {}

	SurfReconstructor* constructor;

	Eigen::Vector3f center;
	float half_length;
	Eigen::Vector4f node = Eigen::Vector4f::Zero();

	char depth = 0;
	char type;
	unsigned long long id;

	std::array<std::shared_ptr<TNode>, 8> children;

	bool changeSignDMC(Eigen::Vector4f* verts);

	// void defoliate();

	bool is_leaf()
	{
		return type == LEAF || type == EMPTY;
	}

	template <class T>
	static auto squared(const T& t)
	{
		return t * t;
	}

	float calcErrorDMC(
		Eigen::Vector4f p, float* verts, float* verts_grad, const int oversample);

	void GenerateSampling(float* sample_points);

	void NodeSampling(
		float& curv, bool& signchange, float cellsize,
		float* sample_points, float* sample_grads);

	void NodeCalcNode(float* sample_points, float* sample_grads, float cellsize);

};
