#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "iso_common.h"

class HashGrid
{
public:
	HashGrid();
	~HashGrid() {};

	/* Constructor for const radius*/
	HashGrid(std::vector<Eigen::Vector3f>* particles, float* bounding, float radius, float inf_factor);
	/* Constructor for variable radius*/
	HashGrid(std::vector<Eigen::Vector3f>* particles, std::vector<float>* radiuses, 
		std::vector<unsigned int>& pIndexes, float* bounding, unsigned int radiusId, float inf_factor);

	std::vector<Eigen::Vector3f>* Particles;
	std::vector<unsigned int> PIndexes;
	unsigned int ParticlesNum;
	unsigned int RadiusId;
	float Radius;
	float CellSize;
	float Bounding[6];
	unsigned int XYZCellNum[3];
	unsigned long long CellNum;
	std::vector<long long> HashList;
	std::vector<int> IndexList;
	std::map<long long, int> StartList;
	std::map<long long, int> EndList;
	void GetPIdxList(const Eigen::Vector3f& pos, std::vector<int>& pIdxList);
	void CalcXYZIdx(const Eigen::Vector3f& pos, Eigen::Vector3i& xyzIdx);
	long long CalcCellHash(const Eigen::Vector3i& xyzIdx);
	void GetInCellList(const long long hash, std::vector<int>& pIdxList);
    void GetInBoxParticles(const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, std::vector<int>& insides);
	// void FindParticlesNeighbor(const int& pIdx, std::vector<int>& pIdxList);
private:
	void BuildTable();
	void CalcHashList();
	void FindStartEnd();
	// void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};

