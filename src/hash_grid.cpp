#include "hash_grid.h"
#include "surface_reconstructor.h"


HashGrid::HashGrid(SurfReconstructor* surf_constructor, std::vector<Eigen::Vector3f>& particles, double* bounding, double cellsize)
{
	constructor = surf_constructor;
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
	CellNum = (long long)XYZCellNum[0] * (long long)XYZCellNum[1] * (long long)XYZCellNum[2];

	HashList.resize(constructor->getGlobalParticlesNum(), 0);
	IndexList.resize(constructor->getGlobalParticlesNum(), 0);

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
	std::vector<long long> temp(HashList);
	for (int i = 0; i < constructor->getGlobalParticlesNum(); i++)
	{
		HashList[i] = temp[IndexList[i]];
	}
	FindStartEnd();
}

inline void HashGrid::CalcHashList()
{
	Eigen::Vector3i xyzIdx;
	for (size_t index = 0; index < constructor->getGlobalParticlesNum(); index++)
	{
		CalcXYZIdx((Particles->at(index)), xyzIdx);
		HashList[index] = CalcCellHash(xyzIdx);
		IndexList[index] = index;
	}
}

inline void HashGrid::FindStartEnd()
{
	int index, hash, count = 0, previous = -1;
	
	for (size_t index = 0; index < constructor->getGlobalParticlesNum(); index++)
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

void HashGrid::CalcXYZIdx(const Eigen::Vector3f& pos, Eigen::Vector3i& xyzIdx)
{
	xyzIdx.setZero();
	for (int i = 0; i < 3; i++)
		xyzIdx[i] = int((pos[i] - Bounding[i * 2]) / CellSize);
}

long long HashGrid::CalcCellHash(const Eigen::Vector3i& xyzIdx)
{
	if (xyzIdx[0] < 0 || xyzIdx[0] >= XYZCellNum[0] ||
		xyzIdx[1] < 0 || xyzIdx[1] >= XYZCellNum[1] ||
		xyzIdx[2] < 0 || xyzIdx[2] >= XYZCellNum[2])
		return -1;
	return (long long)xyzIdx[2] * (long long)XYZCellNum[0] * (long long)XYZCellNum[1] + 
		(long long)xyzIdx[1] * (long long)XYZCellNum[0] + (long long)xyzIdx[0];
}

void HashGrid::GetPIdxList(const Eigen::Vector3f& pos, std::vector<int>& pIdxList)
{
	pIdxList.clear();
	Eigen::Vector3i xyzIdx;
	long long neighbor_hash;
	int countIndex, startIndex, endIndex;
	CalcXYZIdx(pos, xyzIdx);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				neighbor_hash = CalcCellHash((xyzIdx + Eigen::Vector3i(x, y, z)));
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

