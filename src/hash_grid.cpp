#include "iso_common.h"
#include "hash_grid.h"

HashGrid::HashGrid(std::vector<Eigen::Vector3f>* particles, float* bounding, float radius, float inf_factor)
{
	assert(IS_CONST_RADIUS);
	Particles = particles;
	ParticlesNum = Particles->size();

	Radius = radius;
	CellSize = radius * inf_factor + 2 * radius;

	int i = 0;
	float length = 0.0;
	float center = 0.0;

	for (i = 0; i < 3; i++)
	{
		length = ((ceil((bounding[i * 2 + 1] - bounding[i * 2]) / CellSize)) * CellSize);
		center = (bounding[i * 2] + bounding[i * 2 + 1]) / 2;
		Bounding[i * 2] = center - length / 2;
		Bounding[i * 2 + 1] = center + length / 2;
	}
	XYZCellNum[0] = std::max(int(ceil((Bounding[1] - Bounding[0]) / CellSize)), 1);
	XYZCellNum[1] = std::max(int(ceil((Bounding[3] - Bounding[2]) / CellSize)), 1);
	XYZCellNum[2] = std::max(int(ceil((Bounding[5] - Bounding[4]) / CellSize)), 1);
	CellNum = (long long)XYZCellNum[0] * (long long)XYZCellNum[1] * (long long)XYZCellNum[2];

	HashList.resize(ParticlesNum, 0);
	IndexList.resize(ParticlesNum, 0);

	BuildTable();
	HashList.clear();
}

HashGrid::HashGrid(std::vector<Eigen::Vector3f>* particles, std::vector<float>* radiuses, 
	std::vector<unsigned int>& pIndexes, float* bounding, unsigned int radiusId, float inf_factor)
{
	assert(!IS_CONST_RADIUS);
	Particles = particles;
	PIndexes.assign(pIndexes.begin(), pIndexes.end());
	ParticlesNum = PIndexes.size();

	RadiusId = radiusId;
	Radius = radiuses->at(RadiusId);
	CellSize = Radius * inf_factor + 2 * Radius;

	int i = 0;
	float length = 0.0;
	float center = 0.0;

	for (i = 0; i < 3; i++)
	{
		length = ((ceil((bounding[i * 2 + 1] - bounding[i * 2]) / CellSize)) * CellSize);
		center = (bounding[i * 2] + bounding[i * 2 + 1]) / 2;
		Bounding[i * 2] = center - length / 2;
		Bounding[i * 2 + 1] = center + length / 2;
	}
	XYZCellNum[0] = std::max(int(ceil((Bounding[1] - Bounding[0]) / CellSize)), 1);
	XYZCellNum[1] = std::max(int(ceil((Bounding[3] - Bounding[2]) / CellSize)), 1);
	XYZCellNum[2] = std::max(int(ceil((Bounding[5] - Bounding[4]) / CellSize)), 1);
	CellNum = (long long)XYZCellNum[0] * (long long)XYZCellNum[1] * (long long)XYZCellNum[2];

	HashList.resize(ParticlesNum, 0);
	IndexList.resize(ParticlesNum, 0);

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
	for (int i = 0; i < ParticlesNum; i++)
	{
		HashList[i] = temp[IndexList[i]];
	}
	FindStartEnd();
}

inline void HashGrid::CalcHashList()
{
	Eigen::Vector3i xyzIdx;
	for (size_t index = 0; index < ParticlesNum; index++)
	{
		if (IS_CONST_RADIUS)
		{
			CalcXYZIdx(Particles->at(index), xyzIdx);
		} else {
			CalcXYZIdx(Particles->at(PIndexes[index]), xyzIdx);
		}
		HashList[index] = CalcCellHash(xyzIdx);
		IndexList[index] = index;
	}
}

inline void HashGrid::FindStartEnd()
{
	int index, hash, count = 0, previous = -1;
	
	for (size_t index = 0; index < ParticlesNum; index++)
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
	if (xyzIdx[0] < 0 || xyzIdx[0] > XYZCellNum[0] ||
		xyzIdx[1] < 0 || xyzIdx[1] > XYZCellNum[1] ||
		xyzIdx[2] < 0 || xyzIdx[2] > XYZCellNum[2])
		return -1;
	return (long long)xyzIdx[2] * (long long)XYZCellNum[0] * (long long)XYZCellNum[1] + 
		(long long)xyzIdx[1] * (long long)XYZCellNum[0] + (long long)xyzIdx[0];
}

void HashGrid::GetInCellList(const long long hash, std::vector<int>& pIdxList)
{
	int countIndex, startIndex, endIndex;
	if ((StartList.find(hash) != StartList.end()) && (EndList.find(hash) != EndList.end()))
	{
		startIndex = StartList[hash];
		endIndex = EndList[hash];
	}
	else
	{
		return;
	}
	for (int countIndex = startIndex; countIndex < endIndex; countIndex++)
		pIdxList.push_back(IndexList[countIndex]);
}

void HashGrid::GetInBoxParticles(
	const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, 
	std::vector<int>& insides)
{
	Eigen::Vector3i minXyzIdx, maxXyzIdx;
	CalcXYZIdx(box1, minXyzIdx);
	CalcXYZIdx(box2, maxXyzIdx);

	long long temp_hash;
	for (int x = (minXyzIdx.x()-1); x <= (maxXyzIdx.x()+1); x++)
    {
        for (int y = (minXyzIdx.y()-1); y <= (maxXyzIdx.y()+1); y++)
        {
            for (int z = (minXyzIdx.z()-1); z <= (maxXyzIdx.z()+1); z++)
            {
                temp_hash = CalcCellHash(Eigen::Vector3i(x, y, z));
                if (temp_hash < 0) {continue;}
                GetInCellList(temp_hash, insides);
            }
        }
    }
}

void HashGrid::GetPIdxList(const Eigen::Vector3f& pos, std::vector<int>& pIdxList)
{
	pIdxList.clear();
	Eigen::Vector3i xyzIdx;
	long long neighbor_hash;
	CalcXYZIdx(pos, xyzIdx);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				neighbor_hash = CalcCellHash((xyzIdx + Eigen::Vector3i(x, y, z)));
				if (neighbor_hash < 0) {continue;}
				GetInCellList(neighbor_hash, pIdxList);
			}
		}
	}
}

