#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <coord_struct.h>
//#include <Eigen/Dense>
#include "iso_common.h"
#include <algorithm>

class HashGrid
{
public:
	HashGrid();
	~HashGrid() {};

	/* Constructor for const radius*/
	HashGrid(std::vector<cstoneOctree::Vec3f>* particles, float* bounding, float radius, float inf_factor);
	/* Constructor for variable radius*/
	HashGrid(std::vector<cstoneOctree::Vec3f>* particles, std::vector<float>* radiuses,
		std::vector<unsigned int>& pIndexes, float* bounding, unsigned int radiusId, float inf_factor);

	float CellSize;
	float Bounding[6];
	unsigned int XYZCellNum[3];
	unsigned long long CellNum;
	std::vector<unsigned int> PIndexes;
	std::vector<long long> HashList;
	std::vector<int> IndexList;
	std::map<long long, int> StartList;
	std::map<long long, int> EndList;
	void GetPIdxEstimate(const cstoneOctree::Vec3f& pos, int& estimate);
	void GetPIdxList(const cstoneOctree::Vec3f& pos, std::vector<int>& pIdxList);
	void GetPIdxList(const cstoneOctree::Vec3f& pos, int& numNeighbors, int ngmax, int* pIdxList);
	void CalcXYZIdx(const cstoneOctree::Vec3f& pos, cstoneOctree::Vec3i& xyzIdx);
	long long CalcCellHash(const cstoneOctree::Vec3i& xyzIdx);
	void GetInCellList(const long long hash, std::vector<int>& pIdxList);
	void GetInBoxEstimate(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& inCells);
    void GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, std::vector<int>& insides);
    void GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& numNeighbors, int ngmax, int* insides);
private:
	void BuildTable(const int particlesNum, const std::vector<cstoneOctree::Vec3f>* particles);
	void CalcHashList(const int particlesNum, const std::vector<cstoneOctree::Vec3f>* particles);
	void FindStartEnd(const int particlesNum);
	// void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};

