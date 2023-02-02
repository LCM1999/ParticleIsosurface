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
struct KDTreeNode
{
    std::pair<Eigen::Vector3f, int> particle; // 划分粒子
    SplitAxis axis; // 分割轴
    Eigen::Vector3f reigon_min, reigon_max; // 记录目前划分区域的范围
    unsigned int id_min, id_max; // 建树后，划分的结点必定连续，记录最小最大编号
    KDTreeNode* son[2]; // 划分子结点
};
struct CompareQueueItem
{
    bool operator () (std::pair<double, std::pair<Eigen::Vector3f, int>> a, std::pair<double, std::pair<Eigen::Vector3f, int>> b)
    {
        if (a.first != b.first) return a.first < b.first;
        else if (a.second.first[0] != b.second.first[0]) return a.second.first[0] < b.second.first[0];
        else if (a.second.first[1] != b.second.first[1]) return a.second.first[1] < b.second.first[1];
        else return a.second.first[2] < b.second.first[2];
    }
};

class KDTree
{
private:
    std::vector<std::pair<Eigen::Vector3f, unsigned int> > particles; // 粒子坐标
    KDTreeNode* root; // KD-Tree 根
    std::priority_queue<std::pair<double, std::pair<Eigen::Vector3f, int>>, std::vector<std::pair<double, std::pair<Eigen::Vector3f, int>> >, CompareQueueItem> KNearestQueue; // 辅助计算最近邻居

    void DeleteTreeNode(KDTreeNode* node);
    KDTreeNode* BuildKDTree(int range_left, int range_right);
    void SaveQueueData(std::vector<unsigned int>* nearest_particle_ids, std::vector<Eigen::Vector3f>* nearest_particles, std::vector<double>* distance, unsigned int k);
    void GetSplitPoint(int range_left, int range_right, std::pair<Eigen::Vector3f, int>& split_point, SplitAxis& split_axis, Eigen::Vector3f& region_min, Eigen::Vector3f& region_max);
    void GetKNearestSearch(KDTreeNode* node, const Eigen::Vector3f& target, unsigned int k);
    int GetCircleTreeIntersectCondition(KDTreeNode* node, const Eigen::Vector3f& target, double radius);
    void GetPointWithinRadiusSearch(KDTreeNode* node, const Eigen::Vector3f& target, double radius);
public:
    KDTree();
    ~KDTree();
    KDTree(std::vector<Eigen::Vector3f>* _particles, std::vector<int>* _index = nullptr);

    int GetKNearest(const Eigen::Vector3f& target, unsigned int k, std::vector<unsigned int>* nearest_particle_ids = nullptr, std::vector<Eigen::Vector3f>* nearest_particles = nullptr, std::vector<double>* distance = nullptr);
    int GetPointWithinRadius(const Eigen::Vector3f& target, double radius, std::vector<unsigned int>* nearest_particle_ids = nullptr, std::vector<Eigen::Vector3f>* nearest_particles = nullptr, std::vector<double>* distance = nullptr);
};