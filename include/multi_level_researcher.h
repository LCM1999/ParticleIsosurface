#pragma once

#include <vector>
#include <Eigen/Dense>

class HashGrid;

class MultiLevelSearcher
{
private:
    std::vector<HashGrid*> searchers;
    std::vector<std::vector<unsigned int>> sortedIndex;
    std::vector<double> checkedRadiuses;
    double maxRadius = 0, minRadius = 0, avgRadius = 0;
    double infFactor;
	unsigned int particlesNum;

public:
    MultiLevelSearcher(std::vector<Eigen::Vector3d>* particles, double* bounding, std::vector<double>* radiuses, double inf_factor);
    MultiLevelSearcher() {};
    ~MultiLevelSearcher() {};

    inline std::vector<double>* getCheckedRadiuses() {return &checkedRadiuses;}
    inline std::vector<HashGrid*>* getSearchers() {return &searchers;};
    inline double getMaxRadius() {return maxRadius;}
    inline double getMinRadius() {return minRadius;}
    inline double getAvgRadius() {return avgRadius;}
    inline double getParticlesNum() {return particlesNum;}


    void GetNeighbors(const Eigen::Vector3d& pos, std::vector<int>& neighbors);
    void GetInBoxParticles(const Eigen::Vector3d& box1, const Eigen::Vector3d& box2, std::vector<int>& insides);
};
