#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <queue>
#include <algorithm>

class SurfReconstructor;
enum SplitAxis: unsigned int
{
    X = 0,
    Y = 1,
    Z = 2,
    NOT_SET = 3
};
/**
 * @brief 粒子坐标-ID键值对
*/
struct ParticleIDPair
{
    Eigen::Vector3f coordinate;
    int id;
    bool operator < (const ParticleIDPair& _p)
    {
        for (int i = 0; i < 3; ++i)
            if (coordinate[i] != _p.coordinate[i]) return coordinate[i] < _p.coordinate[i];
        return id < _p.id;
    }
};
/**
 * @brief KDTree 子结点
*/
struct KDTreeNode
{
    ParticleIDPair particle; // 划分粒子
    SplitAxis axis; // 分割轴
    Eigen::Vector3f reigon_min, reigon_max; // 记录目前划分区域的范围
    unsigned int id_min, id_max; // 建树后，划分的结点必定连续，记录最小最大编号
    KDTreeNode* son[2]; // 划分子结点
};
/**
 * @brief 定义优先队列的比较模式
*/
struct CompareQueueItem
{
    bool operator () (std::pair<double, ParticleIDPair> a, std::pair<double, ParticleIDPair> b)
    {
        if (a.first != b.first) return a.first < b.first;
        else return a.second < b.second;
    }
};
/**
 * @brief KDTree结构
*/
class KDTree
{
private:
    std::vector<ParticleIDPair> particles; // 粒子坐标
    KDTreeNode* root; // KD-Tree 根
    std::priority_queue<std::pair<double, ParticleIDPair>, std::vector<std::pair<double, ParticleIDPair> >, CompareQueueItem> KNearestQueue; // 辅助计算最近邻居

    void DeleteTreeNode(KDTreeNode* node);
    KDTreeNode* BuildKDTree(int range_left, int range_right);
    void SaveQueueData(std::vector<int>* nearest_particle_ids, std::vector<Eigen::Vector3f>* nearest_particles, std::vector<double>* distance, unsigned int k = 0);
    void GetSplitPoint(int range_left, int range_right, ParticleIDPair& split_point, SplitAxis& split_axis, Eigen::Vector3f& region_min, Eigen::Vector3f& region_max);
    void GetKNearestSearch(KDTreeNode* node, const Eigen::Vector3f& target, unsigned int k);
    int GetCircleTreeIntersectCondition(KDTreeNode* node, const Eigen::Vector3f& target, double radius);
    void GetPointWithinRadiusSearch(KDTreeNode* node, const Eigen::Vector3f& target, double radius);
public:
    KDTree();
    ~KDTree();
    KDTree(std::vector<Eigen::Vector3f>* _particles, std::vector<int>* _index = nullptr);

    int GetKNearest(const Eigen::Vector3f& target, unsigned int k, std::vector<int>* nearest_particle_ids = nullptr, std::vector<Eigen::Vector3f>* nearest_particles = nullptr, std::vector<double>* distance = nullptr);
    int GetPointWithinRadius(const Eigen::Vector3f& target, double radius, std::vector<int>* nearest_particle_ids = nullptr, std::vector<Eigen::Vector3f>* nearest_particles = nullptr, std::vector<double>* distance = nullptr);
};