#pragma once
#include <vector>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <queue>
#include <algorithm>
#include "kdtree.h"

struct RadiusTreePair
{
    double max_radius;
    KDTree* tree;
    RadiusTreePair(double _m = 0.0, KDTree* _t = nullptr)
    {
        max_radius = _m;
        tree = _t;
    }
};

struct ParticleInfo
{
    Eigen::Vector3f coordinate;
    int id;
    double radius;
    ParticleInfo(Eigen::Vector3f _c, int _i, double _r)
    {
        coordinate = _c;
        id = _i;
        radius = _r;
    }
};

class KDTreeNeighborhoodSearcher
{
private:
    std::vector<ParticleInfo> particles; // 粒子信息
    std::vector<RadiusTreePair> groups; // 粒子分组
    std::vector<int> index_map; // 编号映射，在内部使用的都是 0-n 编号，需要输出时会处理
    void InitializeGroups();
public:
    KDTreeNeighborhoodSearcher(std::vector<Eigen::Vector3f>* _particles, std::vector<double>* _radius, std::vector<int>* _index = nullptr);
    ~KDTreeNeighborhoodSearcher();
    void GetNeighborhood(Eigen::Vector3f target, double radius, std::vector<int>* neighborhood_ids = nullptr, std::vector<Eigen::Vector3f>* neighborhood_coordinates = nullptr);
};