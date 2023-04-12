#pragma once

#include <float.h>
#include <Eigen/Dense>
#include "cube_arrays.h"
#include "utils.h"
#include "index.h"
#include <math.h>

class SurfReconstructor;

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
	TNode() {}

	TNode(SurfReconstructor* surf_constructor);

	~TNode()
	{
		defoliate();
	}

	SurfReconstructor* constructor;

	Eigen::Vector3f center;
	float half_length;
	Eigen::Vector4f node = Eigen::Vector4f::Zero();

	short depth = 0;
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

	double calcErrorDMC(
		Eigen::Vector4f p, std::vector<Eigen::Vector4f>& verts, std::vector<Eigen::Vector3f>& verts_grad);

	void vertAll(float& curv, bool& signchange, float& qef_error, float& sample);

	void GenerateSampling(
		std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
		int& oversample, const float sample_radius);

	void NodeSampling(
		float& curv, bool& signchange, 
		std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
		const int oversample);

	void NodeCalcNode(
		std::vector<Eigen::Vector4f>& sample_points, std::vector<Eigen::Vector3f>& sample_grads, 
		const int oversample);

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


