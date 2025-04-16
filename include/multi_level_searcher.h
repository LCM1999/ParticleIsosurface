#pragma once
#ifndef MULTI_LEVEL_SEARCHER_H
#define MULTI_LEVEL_SEARCHER_H

#include <vector>
//#include <Eigen/Dense>
#include <coord_struct.h>

class HashGrid;

class MultiLevelSearcher
{
private:
    std::vector<HashGrid*> searchers;
    std::vector<int> maxRadiusParticleIds;
    float maxRadius = 0, minRadius = 0, avgRadius = 0;
    float infFactor;

public:
    MultiLevelSearcher(std::vector<cstoneOctree::Vec3f>* particles, float* bounding, std::vector<float>* radiuses, float inf_factor);
    MultiLevelSearcher() {};
    ~MultiLevelSearcher() {
        for (size_t i = 0; i < searchers.size(); i++)
        {
            delete searchers[i];
            searchers[i] = 0;
        }
    };

    inline std::vector<HashGrid*>* getSearchers() {return &searchers;};
    inline std::vector<int> getMaxRadiusParticleIds() {return maxRadiusParticleIds;};
    inline float getMaxRadius() {return maxRadius;}
    inline float getMinRadius() {return minRadius;}
    inline float getAvgRadius() {return avgRadius;}

    void GetNeighborsEstimate(const cstoneOctree::Vec3f& pos, int& estimate);
    void GetNeighbors(const cstoneOctree::Vec3f& pos, std::vector<int>& neighbors);
    void GetNeighbors(const cstoneOctree::Vec3f& pos, int& numNeighbors, int ngmax, int* neighbors);
    void GetInBoxEstimate(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, int& insides);
    void GetInBoxParticles(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, std::vector<int>& insides);
    void GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& numNeighbors, int ngmax, int* insides);
};

#endif