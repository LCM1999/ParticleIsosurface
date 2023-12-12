#pragma once

#include <vector>
#include <Eigen/Dense>

class HashGrid;

class MultiLevelSearcher
{
private:
    std::vector<HashGrid*> searchers;
    std::vector<std::vector<unsigned int>> sortedIndex;
    std::vector<float> checkedRadiuses;
    float maxRadius = 0, minRadius = 0, avgRadius = 0;
    float infFactor;
	unsigned int particlesNum;

public:
    MultiLevelSearcher(std::vector<Eigen::Vector3f>* particles, float* bounding, std::vector<float>* radiuses, float inf_factor);
    MultiLevelSearcher() {};
    ~MultiLevelSearcher() {
        for (size_t i = 0; i < searchers.size(); i++)
        {
            delete searchers[i];
            searchers[i] = 0;
        }
    };

    inline std::vector<float>* getCheckedRadiuses() {return &checkedRadiuses;}
    inline std::vector<HashGrid*>* getSearchers() {return &searchers;};
    inline float getMaxRadius() {return maxRadius;}
    inline float getMinRadius() {return minRadius;}
    inline float getAvgRadius() {return avgRadius;}
    inline float getParticlesNum() {return particlesNum;}

    void GetNeighbors(const Eigen::Vector3f& pos, std::vector<int>& neighbors);
    void GetInBoxParticles(const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, std::vector<int>& insides);
};
