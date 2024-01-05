#include <set>
#include "multi_level_researcher.h"

#include "hash_grid.h"

MultiLevelSearcher::MultiLevelSearcher(std::vector<Eigen::Vector3f>* particles, float* bounding, std::vector<float>* radiuses, float inf_factor)
{
    // std::vector<std::vector<Eigen::Vector3f>> sortedParticles;
    maxRadius = *std::max_element(radiuses->begin(), radiuses->end()) * 1.01;
    minRadius = *std::min_element(radiuses->begin(), radiuses->end()) * 0.99;
    infFactor = inf_factor;
    particlesNum = particles->size();
    std::vector<std::pair<float, float>> bin_bounds;
    float bin_extent = minRadius * 0.5;
    int bins = std::max(int(ceil((maxRadius - minRadius) / bin_extent)), 1);
    bin_extent = (maxRadius - minRadius) / bins;
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
    // sortedParticles.clear();
    sortedIndex.clear();
    // sortedParticles.resize(bins);
    sortedIndex.resize(bins);
    // for (size_t i = 0; i < bins; i++)
    // {
    //     // sortedParticles[i].clear();
    //     sortedIndex[i].clear();
    // }

    for (int i = 0; i < radiuses->size(); i++)
    {
        // sortedParticles[whichBin(radiuses->at(i))].push_back(particles->at(i));
        sortedIndex[whichBin(radiuses->at(i))].push_back(i);
        avgRadius += radiuses->at(i);
    }
    avgRadius /= radiuses->size();
    if (avgRadius < minRadius) avgRadius = minRadius;
    if (avgRadius > maxRadius) avgRadius = maxRadius;
    for (int i = 0; i < bins; i++)
    {
        if (sortedIndex[i].size() == 0) continue;
        float temp_bounding [6] = {0.0f};
        temp_bounding[0] = bounding[0];
        temp_bounding[1] = bounding[1];
        temp_bounding[2] = bounding[2];
        temp_bounding[3] = bounding[3];
        temp_bounding[4] = bounding[4];
        temp_bounding[5] = bounding[5];
        unsigned int binRadiusId = *std::max_element(sortedIndex[i].begin(), sortedIndex[i].end(), 
            [&](unsigned int& a, unsigned int& b) {
                return radiuses->at(a) < radiuses->at(b);
            });
        // std::cout << sortedIndex[i].size() << ", " << radiuses->at(binRadiusId) << std::endl;
        searchers.push_back(new HashGrid(particles, radiuses, sortedIndex[i], temp_bounding, binRadiusId, inf_factor));
    }
    std::set<float> st(radiuses->begin(), radiuses->end());
    checkedRadiuses.assign(st.begin(), st.end());
    printf("   Seachers level: %d.\n", searchers.size());
}

void MultiLevelSearcher::GetNeighbors(const Eigen::Vector3f& pos, std::vector<int>& neighbors)
{
    std::vector<int> subNeighbors;
    size_t sIndexId = 0, searcherId = 0;
    for (sIndexId = 0; sIndexId < sortedIndex.size(); sIndexId++)
    {
        if (sortedIndex[sIndexId].empty())
        {
            continue;
        }
        subNeighbors.clear();
        searchers[searcherId]->GetPIdxList(pos, subNeighbors);
        for (size_t nId = 0; nId < subNeighbors.size(); nId++)
        {
            neighbors.push_back(sortedIndex[sIndexId][subNeighbors[nId]]);
        }
        searcherId++;
    }
}

void MultiLevelSearcher::GetInBoxParticles(const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, std::vector<int>& insides)
{
    std::vector<int> subInsides;
    size_t sIndexId = 0, searcherId = 0;
    for (sIndexId = 0; sIndexId < sortedIndex.size(); sIndexId++)
    {
        if (sortedIndex[sIndexId].empty())
        {
            continue;
        }
        subInsides.clear();
        searchers[searcherId]->GetInBoxParticles(box1, box2, subInsides);
        for (size_t nId = 0; nId < subInsides.size(); nId++)
        {
            insides.push_back(sortedIndex[sIndexId][subInsides[nId]]);
        }
        searcherId++;
    }
}
