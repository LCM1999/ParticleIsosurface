#include <set>
#include "multi_level_searcher.h"
#include <cfloat>
#include "hash_grid.h"

MultiLevelSearcher::MultiLevelSearcher(std::vector<cstoneOctree::Vec3f>* particles, float* bounding, std::vector<float>* radiuses, float inf_factor)
{
    maxRadius = *std::max_element(radiuses->begin(), radiuses->end());   // * 1.01
    minRadius = *std::min_element(radiuses->begin(), radiuses->end());   // * 0.99
    infFactor = inf_factor;
    int particlesNum = particles->size();
    std::vector<std::pair<float, float>> bin_bounds;
    float bin_extent = minRadius * 0.5;
    int bins = std::max(int(ceil((maxRadius - minRadius) / bin_extent)), 1);
    bin_extent = (maxRadius - minRadius) / bins;
    std::vector<std::vector<unsigned>>sortedIndex(bins);
    if (bins == 1 || bin_extent == 0.0f)
    {
        for (int i = 0; i < radiuses->size(); i++)
        {
            sortedIndex[0].push_back(i);
        }
    } else {
        for (size_t i = 0; i < bins; i++)
        {
            bin_bounds.push_back(
                std::pair<float, float>(
                    minRadius+(i*bin_extent), 
                    std::min(minRadius+((i+1)*bin_extent), maxRadius)));
        }
        auto whichBin = [&](const float r)
        {
            for (auto tit = bin_bounds.begin(); tit < bin_bounds.end(); tit++)
            {
                if (r >= tit->first && r <= tit->second)
                    return static_cast<int>(std::distance(bin_bounds.begin(), tit));
            }
            return -1;
        };
    
        for (int i = 0; i < radiuses->size(); i++)
        {
            sortedIndex[whichBin(radiuses->at(i))].push_back(i);
        }
    }
    for (int i = 0; i < bins; i++)
    {
        if (sortedIndex[i].size() == 0) continue;
        float temp_bounding [6] = {0.0f};
        for (size_t j = 0; j < 6; j++)
        {
            temp_bounding[j] = bounding[j];
        }
        // temp_bounding[0] = temp_bounding[2] = temp_bounding[4] = FLT_MAX;
        // temp_bounding[1] = temp_bounding[3] = temp_bounding[5] = -FLT_MAX;
        for (auto pI: sortedIndex[i])
        {
            temp_bounding[0] = std::min(temp_bounding[0], particles->at(pI).x);
            temp_bounding[1] = std::max(temp_bounding[1], particles->at(pI).x);
            temp_bounding[2] = std::min(temp_bounding[2], particles->at(pI).y);
            temp_bounding[3] = std::max(temp_bounding[3], particles->at(pI).y);
            temp_bounding[4] = std::min(temp_bounding[4], particles->at(pI).z);
            temp_bounding[5] = std::max(temp_bounding[5], particles->at(pI).z);
        }
        unsigned int binRadiusId = *std::max_element(sortedIndex[i].begin(), sortedIndex[i].end(), 
            [&](unsigned int& a, unsigned int& b) {
                return radiuses->at(a) < radiuses->at(b);
            });
        maxRadiusParticleIds.push_back(binRadiusId);
        searchers.push_back(new HashGrid(particles, radiuses, sortedIndex[i], temp_bounding, binRadiusId, inf_factor));
    }
    printf("   Seachers level: %d.\n", searchers.size());
}

void MultiLevelSearcher::GetNeighborsEstimate(const cstoneOctree::Vec3f& pos, int& estimate)
{
    for (auto& searcher : searchers)
    {
        searcher->GetPIdxEstimate(pos, estimate);
    }
}

void MultiLevelSearcher::GetNeighbors(const cstoneOctree::Vec3f& pos, std::vector<int>& neighbors)
{
    for (auto& searcher : searchers)
    {
        searcher->GetPIdxList(pos, neighbors);
    }
}

void MultiLevelSearcher::GetNeighbors(const cstoneOctree::Vec3f& pos, int& numNeighbors, int ngmax, int* neighbors)
{
    for (auto& searcher : searchers)
    {
        searcher->GetPIdxList(pos, numNeighbors, ngmax, neighbors);
    }
}

void MultiLevelSearcher::GetInBoxEstimate(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, int& insides)
{
    for (auto& searcher : searchers)
    {
        searcher->GetInBoxEstimate(box1, box2, insides);
    }
}

void MultiLevelSearcher::GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& numNeighbors, int ngmax, int* insides)
{
    for (auto& searcher : searchers)
    {
        searcher->GetInBoxParticles(box1, box2, numNeighbors, ngmax, insides);
    }
}

void MultiLevelSearcher::GetInBoxParticles(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, std::vector<int>& insides)
{
    for (auto& searcher : searchers)
    {
        searcher->GetInBoxParticles(box1, box2, insides);
    }
}
