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
    std::vector<Eigen::Vector3f>* Particles; // 粒子坐标
    double CellSize;
    double Bounding[6];
    unsigned int XYZCellNum[3]; // 网格长宽高
    unsigned long long CellNum; // 总网格个数
    std::vector<long long> HashList; // 网格坐标哈希，从小到大排序
    std::vector<int> IndexList; // 粒子编号，按照其网格坐标哈希从小到大排序
    std::map<long long, int> StartList; // 哈希值对应最小的网格编号
    std::map<long long, int> EndList; // 哈希值对应的最大的网格编号 + 1
    void GetPIdxList(const Eigen::Vector3f& pos, std::vector<int>& pIdxList);
    /**
     * @brief 将粒子坐标转变为网格坐标（即每个方位的第几个格子）
     * @param pos 粒子坐标
     * @param xyzIdx 网格坐标
    */
    void CalcXYZIdx(const Eigen::Vector3f& pos, Eigen::Vector3i& xyzIdx);
    /**
     * @brief 将网格坐标转变为单个数值哈希
     * @param xyzIdx 网格坐标
     * @return 单个数值哈希
    */
    long long CalcCellHash(const Eigen::Vector3i& xyzIdx);
    // void FindParticlesNeighbor(const int& pIdx, std::vector<int>& pIdxList);
private:
    void BuildTable();
    void CalcHashList();
    void FindStartEnd();
    // void GetNeighborHashs(vect3d* pos, int* neighborHashs);
};

