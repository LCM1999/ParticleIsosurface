#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "iso_common.h"

class SurfReconstructor;

class HashGrid
{
public:
	HashGrid();
	~HashGrid();

	HashGrid(SurfReconstructor* surf_constrcutor, std::vector<Eigen::Vector3f>& particles, double* bounding, double cellsize);

	SurfReconstructor* constructor;
	std::vector<Eigen::Vector3f>* Particles;
	double CellSize;
	double Bounding[6];
	unsigned int XYZCellNum[3];
	unsigned __int64 CellNum;
	std::vector<__int64> HashList;
	std::vector<int> IndexList;
	std::map<__int64, int> StartList;
	std::map<__int64, int> EndList;
	void GetPIdxList(const Eigen::Vector3f& pos, std::vector<int>& pIdxList);
	void CalcXYZIdx(const Eigen::Vector3f& pos, Eigen::Vector3i& xyzIdx);
	__int64 CalcCellHash(const Eigen::Vector3i& xyzIdx);
	// void FindParticlesNeighbor(const int& pIdx, std::vector<int>& pIdxList);
private:
	void BuildTable();
	void CalcHashList();
	void FindStartEnd();
	// void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};
