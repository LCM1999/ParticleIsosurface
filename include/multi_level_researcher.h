#pragma once

#include <vector>
#include <Eigen/Dense>

class HashGrid;

class MultiLevelSearcher
{
private:
    std::vector<HashGrid*> searchers;
    std::vector<std::vector<int>> sortedIndex;
    std::vector<float> checkedRadiuses;
    float maxRadius = 0, minRadius = 0, avgRadius = 0;
    float infFactor;

public:
    MultiLevelSearcher(std::vector<Eigen::Vector3f>* particles, std::vector<float>* radiuses, float inf_factor);
    MultiLevelSearcher(){};

    inline std::vector<float>* getCheckedRadiuses() {return &checkedRadiuses;}
    inline float getMaxRadius() {return maxRadius;}
    inline float getMinRadius() {return minRadius;}
    inline float getAvgRadius() {return avgRadius;}

    void GetNeighbors(const Eigen::Vector3f& pos, std::vector<int>& neighbors);
    void GetInBoxParticles(const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, std::vector<int>& insides);
};
