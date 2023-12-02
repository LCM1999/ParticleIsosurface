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

	HashGrid(std::vector<Eigen::Vector3d>& particles, double* bounding, double radius, double inf_factor);

	std::vector<Eigen::Vector3d>* Particles;
	unsigned int ParticlesNum;
	double Radius;
	double CellSize;
	double Bounding[6];
	unsigned int XYZCellNum[3];
	unsigned long long CellNum;
	std::vector<long long> HashList;
	std::vector<int> IndexList;
	std::map<long long, int> StartList;
	std::map<long long, int> EndList;
	void GetPIdxList(const Eigen::Vector3d& pos, std::vector<int>& pIdxList);
	void CalcXYZIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& xyzIdx);
	long long CalcCellHash(const Eigen::Vector3i& xyzIdx);
	void GetInCellList(const long long hash, std::vector<int>& pIdxList);
    void GetInBoxParticles(const Eigen::Vector3d& box1, const Eigen::Vector3d& box2, std::vector<int>& insides);
	// void FindParticlesNeighbor(const int& pIdx, std::vector<int>& pIdxList);
private:
	void BuildTable();
	void CalcHashList();
	void FindStartEnd();
	// void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};

