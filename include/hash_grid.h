#pragma once

#include <iostream>
#include <vector>
#include <map>
#include "iso_common.h"
#include "vect.h"

class HashGrid
{
public:
	HashGrid();
	~HashGrid();

	HashGrid(std::vector<vect3f>& particles, double* bounding, double cellsize);

	std::vector<vect3f>* Particles;
	double CellSize;
	double Bounding[6];
	unsigned int XYZCellNum[3];
	unsigned __int64 CellNum;
	std::vector<__int64> HashList;
	std::vector<int> IndexList;
	std::map<__int64, int> StartList;
	std::map<__int64, int> EndList;
	void GetPIdxList(const vect3f& pos, std::vector<int>& pIdxList);
	void CalcXYZIdx(const vect3f& pos, vect3i& xyzIdx);
	__int64 CalcCellHash(const vect3i& xyzIdx);
	// void FindParticlesNeighbor(const int& pIdx, std::vector<int>& pIdxList);
private:
	void BuildTable();
	void CalcHashList();
	void FindStartEnd();
	// void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};

inline HashGrid::HashGrid(std::vector<vect3f>& particles, double* bounding, double cellsize)
{
	Particles = &particles;
	CellSize = cellsize;

	int i = 0;
	double length = 0.0;
	double center = 0.0;

	for (i = 0; i < 3; i++)
	{
		length = ((ceil((bounding[i * 2 + 1] - bounding[i * 2]) / CellSize)) * CellSize);
		center = (bounding[i * 2] + bounding[i * 2 + 1]) / 2;
		Bounding[i * 2] = center - length / 2;
		Bounding[i * 2 + 1] = center + length / 2;
	}
	XYZCellNum[0] = int(ceil((Bounding[1] - Bounding[0]) / CellSize));
	XYZCellNum[1] = int(ceil((Bounding[3] - Bounding[2]) / CellSize));
	XYZCellNum[2] = int(ceil((Bounding[5] - Bounding[4]) / CellSize));
	CellNum = (__int64)XYZCellNum[0] * (__int64)XYZCellNum[1] * (__int64)XYZCellNum[2];

	HashList.resize(GlobalParticlesNum, 0);
	IndexList.resize(GlobalParticlesNum, 0);
	// StartList.resize(CellNum, 0);
	// EndList.resize(CellNum, 0);

	BuildTable();
	HashList.clear();
}

inline void HashGrid::BuildTable()
{
	CalcHashList();
	std::sort(IndexList.begin(), IndexList.end(),
		[&](const int& a, const int& b) {
			return (HashList[a] < HashList[b]);
		}
	);
	std::vector<__int64> temp(HashList);
	for (int i = 0; i < GlobalParticlesNum; i++)
	{
		HashList[i] = temp[IndexList[i]];
	}
	FindStartEnd();
}

inline void HashGrid::CalcHashList()
{
	vect3i xyzIdx;
	for (size_t index = 0; index < GlobalParticlesNum; index++)
	{
		CalcXYZIdx((Particles->at(index)), xyzIdx);
		HashList[index] = CalcCellHash(xyzIdx);
		IndexList[index] = index;
	}
}

inline void HashGrid::FindStartEnd()
{
	int index, hash, count = 0, previous = -1;
	
	for (size_t index = 0; index < GlobalParticlesNum; index++)
	{
		hash = HashList[index];
		if (hash < 0)
		{
			continue;
		}
		if (hash != previous)
		{
			StartList[hash] = count;
			previous = hash;
		}
		count++;
		EndList[hash] = count;
	}
}

inline void HashGrid::CalcXYZIdx(const vect3f& pos, vect3i& xyzIdx)
{
	xyzIdx.zero();
	for (int i = 0; i < 3; i++)
		xyzIdx[i] = int((pos.v[i] - Bounding[i * 2]) / CellSize);
}

inline __int64 HashGrid::CalcCellHash(const vect3i& xyzIdx)
{
	if (xyzIdx.v[0] < 0 || xyzIdx.v[0] >= XYZCellNum[0] ||
		xyzIdx.v[1] < 0 || xyzIdx.v[1] >= XYZCellNum[1] ||
		xyzIdx.v[2] < 0 || xyzIdx.v[2] >= XYZCellNum[2])
		return -1;
	return (__int64)xyzIdx.v[2] * (__int64)XYZCellNum[0] * (__int64)XYZCellNum[1] + 
		(__int64)xyzIdx.v[1] * (__int64)XYZCellNum[0] + (__int64)xyzIdx.v[0];
}

inline void HashGrid::GetPIdxList(const vect3f& pos, std::vector<int>& pIdxList)
{
	pIdxList.clear();
	vect3i xyzIdx;
	__int64 neighbor_hash;
	int countIndex, startIndex, endIndex;
	CalcXYZIdx(pos, xyzIdx);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				neighbor_hash = CalcCellHash((xyzIdx + vect3i(x, y, z)));
				if (neighbor_hash < 0)
					continue;
				if ((StartList.find(neighbor_hash) != StartList.end()) && (EndList.find(neighbor_hash) != EndList.end()))
				{
					startIndex = StartList[neighbor_hash];
					endIndex = EndList[neighbor_hash];
				}
				else
				{
					continue;
				}
				for (int countIndex = startIndex; countIndex < endIndex; countIndex++)
					pIdxList.push_back(IndexList[countIndex]);
			}
		}
	}
}





