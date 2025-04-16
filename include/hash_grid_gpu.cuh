#pragma once
#ifndef HASH_GRID_CUH
#define HASH_GRID_CUH

#include <iostream>
#include <vector>
// #include <map>
#include <coord_struct.h>
//#include <Eigen/Dense>
// #include "iso_common.h"
#include <algorithm>
#include "cuda_def.h"

struct HashGridGPU
{
    HOST HashGridGPU();
	HOST ~HashGridGPU() {
        if (PIndexes != nullptr)
        {
            delete[] PIndexes;
            PIndexes = nullptr;
        }
        // if (HashList != nullptr)
        // {
        //     delete[] HashList;
        //     HashList = nullptr;
        // }
        if (IndexList != nullptr)
        {
            delete[] IndexList;
            IndexList = nullptr;
        }
        if (StartList != nullptr)
        {
            delete[] StartList;
            StartList = nullptr;
        }
        if (EndList != nullptr)
        {
            delete[] EndList;
            EndList = nullptr;
        }
    };

	/* Constructor for variable radius*/
	HashGridGPU(std::vector<cstoneOctree::Vec3f>* particles, std::vector<float>* radiuses,
		std::vector<unsigned>& pIndexes, float* bounding, unsigned radiusId, float inf_factor)
    {
        int particlesNum = pIndexes.size();
        PIndexes = new unsigned[particlesNum];
        // PIndexes.resize(particlesNum);
        std::copy(pIndexes.begin(), pIndexes.end(), PIndexes);
        CellSize = radiuses->at(radiusId) * inf_factor;
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
    	CellNum = XYZCellNum[0] * XYZCellNum[1] * XYZCellNum[2];
        // HashList = new int64_t[particlesNum];
        HashList.resize(particlesNum, 0);
        IndexList = new int[particlesNum];
        StartList = new int[CellNum];
        EndList = new int[CellNum];
        std::memset(StartList, -1, CellNum * sizeof(int));
        std::memset(EndList, -1, CellNum * sizeof(int));
        // IndexList.resize(particlesNum, 0);
        // StartList.resize(CellNum, -1);
        // EndList.resize(CellNum, -1);
        BuildTable(particlesNum, particles);
        HashList.clear();
        // delete[] HashList;
        // HashList = nullptr;
    }

    HOST_DEVICE void CalcXYZIdx(const cstoneOctree::Vec3f& pos, cstoneOctree::Vec3i& xyzIdx) const
    {
        xyzIdx.setZero();
        for (int i = 0; i < 3; i++)
            xyzIdx[i] = int((pos[i] - Bounding[i * 2]) / CellSize);
    };

	HOST_DEVICE int64_t CalcCellHash(const cstoneOctree::Vec3i& xyzIdx) const
    {
        if (xyzIdx[0] < 0 || xyzIdx[0] >= XYZCellNum[0] ||
            xyzIdx[1] < 0 || xyzIdx[1] >= XYZCellNum[1] ||
            xyzIdx[2] < 0 || xyzIdx[2] >= XYZCellNum[2])
            return -1;
        return XYZCellNum[0] * XYZCellNum[1] * xyzIdx[2] + XYZCellNum[0] * xyzIdx[1] + xyzIdx[0];
    };

	HOST_DEVICE void GetPIdxEstimate(const cstoneOctree::Vec3f& pos, int& estimate)
    {
        cstoneOctree::Vec3i xyzIdx;
        int64_t neighbor_hash;
        CalcXYZIdx(pos, xyzIdx);
        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    neighbor_hash = CalcCellHash((xyzIdx + cstoneOctree::Vec3i(x, y, z)));
                    if (neighbor_hash < 0) {continue;}
                    // estimate += 9;
                    int countIndex, startIndex, endIndex;
                    if ((StartList[neighbor_hash] >= 0) && (EndList[neighbor_hash] >= 0))
                    {
                        startIndex = StartList[neighbor_hash];
                        endIndex = EndList[neighbor_hash];
                    }
                    else
                    {
                        continue;
                    }
                    estimate += endIndex - startIndex;
                }
            }
        }
    };
	
	HOST_DEVICE void GetPIdxList(const cstoneOctree::Vec3f& pos, int& numNeighbors, int ngmax, int* pIdxList)
    {
        cstoneOctree::Vec3i xyzIdx;
        int64_t neighbor_hash;
        CalcXYZIdx(pos, xyzIdx);
        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    neighbor_hash = CalcCellHash((xyzIdx + cstoneOctree::Vec3i(x, y, z)));
                    if (neighbor_hash < 0) {continue;}
                    int countIndex, startIndex, endIndex;
                    if ((StartList[neighbor_hash] >= 0) && (EndList[neighbor_hash] >= 0))
                    {
                        startIndex = StartList[neighbor_hash];
                        endIndex = EndList[neighbor_hash];
                    }
                    else
                    {
                        continue;
                    }
                    for (int countIndex = startIndex; countIndex < endIndex; countIndex++)
                    {
                        if (numNeighbors < ngmax)
                        {
                            pIdxList[numNeighbors] = PIndexes[IndexList[countIndex]];
                        }
                        numNeighbors++;
                    }
                }
            }
        }
    };
	
	HOST_DEVICE void GetInBoxEstimate(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& insides) const
    {
        cstoneOctree::Vec3i minXyzIdx, maxXyzIdx;
        for (size_t i = 0; i < 3; i++)
        {
            box1[i] = std::max(box1[i], Bounding[2*i]);
            box2[i] = std::min(box2[i], Bounding[2*i+1]);
        }
        CalcXYZIdx(box1, minXyzIdx);
        CalcXYZIdx(box2, maxXyzIdx);
        int64_t temp_hash;
        for (int x = (minXyzIdx.x-1); x <= (maxXyzIdx.x+1); x++)
        {
            for (int y = (minXyzIdx.y-1); y <= (maxXyzIdx.y+1); y++)
            {
                for (int z = (minXyzIdx.z-1); z <= (maxXyzIdx.z+1); z++)
                {
                    temp_hash = CalcCellHash(cstoneOctree::Vec3i(x, y, z));
                    if (temp_hash < 0) {
                        continue;
                    }
                    int startIndex, endIndex;
                    if ((StartList[temp_hash] >= 0) && (EndList[temp_hash] >= 0))
                    {
                        startIndex = StartList[temp_hash];
                        endIndex = EndList[temp_hash];
                    }
                    else
                    {
                        continue;
                    }
                    insides += endIndex - startIndex;
                }
            }
        }
    };
    
    HOST_DEVICE void GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& numNeighbors, int ngmax, int* insides)
    {
        cstoneOctree::Vec3i minXyzIdx, maxXyzIdx;
        for (size_t i = 0; i < 3; i++)
        {
            box1[i] = std::max(box1[i], Bounding[2*i]);
            box2[i] = std::min(box2[i], Bounding[2*i+1]);
        }
        CalcXYZIdx(box1, minXyzIdx);
        CalcXYZIdx(box2, maxXyzIdx);
        int64_t temp_hash;
        for (int x = (minXyzIdx.x-1); x <= (maxXyzIdx.x+1); x++)
        {
            for (int y = (minXyzIdx.y-1); y <= (maxXyzIdx.y+1); y++)
            {
                for (int z = (minXyzIdx.z-1); z <= (maxXyzIdx.z+1); z++)
                {
                    temp_hash = CalcCellHash(cstoneOctree::Vec3i(x, y, z));
                    if (temp_hash < 0) {
                        continue;
                    }
                    int countIndex, startIndex, endIndex;
                    if ((StartList[temp_hash] >= 0) && (EndList[temp_hash] >= 0))
                    {
                        startIndex = StartList[temp_hash];
                        endIndex = EndList[temp_hash];
                    }
                    else
                    {
                        continue;
                    }
                    for (int countIndex = startIndex; countIndex < endIndex; countIndex++)
                    {
                        if (numNeighbors < ngmax)
                        {
                            insides[numNeighbors] = PIndexes[IndexList[countIndex]];
                        }
                        numNeighbors++;
                    }
                }
            }
        }
    };
	
    HOST void BuildTable(const int particlesNum, const std::vector<cstoneOctree::Vec3f>* particles)
    {
        CalcHashList(particlesNum, particles);
        std::sort(IndexList, IndexList + particlesNum,
            [&](const int& a, const int& b) {
                return (HashList[a] < HashList[b]);
            }
        );
        std::vector<int64_t> temp(HashList);
        for (int i = 0; i < particlesNum; i++)
        {
            HashList[i] = temp[IndexList[i]];
        }
        FindStartEnd(particlesNum);
    };

	HOST void CalcHashList(const int particlesNum, const std::vector<cstoneOctree::Vec3f>* particles) 
    {
        cstoneOctree::Vec3i xyzIdx;
        for (size_t index = 0; index < particlesNum; index++)
        {
            CalcXYZIdx(particles->at(PIndexes[index]), xyzIdx);
            HashList[index] = CalcCellHash(xyzIdx);
            IndexList[index] = index;
        }
    };

	HOST void FindStartEnd(const int particlesNum) 
    {
        int index, hash, count = 0, previous = -1;
	
        for (size_t index = 0; index < particlesNum; index++)
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
    };
            
    float CellSize;
    float Bounding[6];
    uint64_t XYZCellNum[3];
    uint64_t CellNum;
    unsigned* PIndexes;
	std::vector<int64_t> HashList;
	int* IndexList;
	int* StartList;
	int* EndList;
};

#endif
