#include <set>
#include "multi_level_researcher.h"

#include "hash_grid.h"

MultiLevelSearcher::MultiLevelSearcher(std::vector<Eigen::Vector3f>* particles, std::vector<float>* radiuses, float inf_factor)
{
    std::vector<std::vector<Eigen::Vector3f>> sortedParticles;
    maxRadius = *std::max_element(radiuses->begin(), radiuses->end());
    minRadius = *std::min_element(radiuses->begin(), radiuses->end());
    infFactor = inf_factor;
    std::vector<std::pair<float, float>> bin_bounds;
    float bin_extent = minRadius * 0.5;
    int bins = floor((maxRadius - minRadius) / bin_extent) + 1;
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
    
    sortedParticles.resize(bins);
    sortedIndex.resize(bins);
    for (int i = 0; i < radiuses->size(); i++)
    {
        sortedParticles[whichBin(radiuses->at(i))].push_back(particles->at(i));
        sortedIndex[whichBin(radiuses->at(i))].push_back(i);
        avgRadius += radiuses->at(i);
    }
    avgRadius /= radiuses->size();
    if (avgRadius < minRadius) avgRadius = minRadius;
    if (avgRadius > maxRadius) avgRadius = maxRadius;
    for (int i = 0; i < sortedParticles.size(); i++)
    {
        double bounding [6] = {0.0f};
        bounding[0] = (*std::min_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.x() < v2.x();
        })).x();
        bounding[2] = (*std::min_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.y() < v2.y();
        })).y();
        bounding[4] = (*std::min_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.z() < v2.z();
        })).z();
        bounding[1] = (*std::max_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.x() < v2.x();
        })).x();
        bounding[3] = (*std::max_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.y() < v2.y();
        })).y();
        bounding[5] = (*std::max_element(sortedParticles[i].begin(), sortedParticles[i].end(), [&](const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
            return v1.z() < v2.z();
        })).z();
        searchers.push_back(new HashGrid(sortedParticles[i], bounding, bin_bounds[i].second * infFactor));
    }
    std::set<float> st(radiuses->begin(), radiuses->end());
    checkedRadiuses.assign(st.begin(), st.end());
}

void MultiLevelSearcher::GetNeighbors(const Eigen::Vector3f& pos, std::vector<int>& neighbors)
{
    std::vector<int> subNeighbors;
    for (size_t searcherId = 0; searcherId < searchers.size(); searcherId++)
    {
        subNeighbors.clear();
        searchers[searcherId]->GetPIdxList(pos, subNeighbors);
        for (size_t nId = 0; nId < subNeighbors.size(); nId++)
        {
            neighbors.push_back(sortedIndex[searcherId][subNeighbors[nId]]);
        }
    }
}

void MultiLevelSearcher::GetInBoxParticles(const Eigen::Vector3f& box1, const Eigen::Vector3f& box2, std::vector<int>& insides)
{
    std::vector<int> subInsides;
    for (size_t searcherId = 0; searcherId < searchers.size(); searcherId++)
    {
        subInsides.clear();
        searchers[searcherId]->GetInBoxParticles(box1, box2, subInsides);
        for (size_t nId = 0; nId < subInsides.size(); nId++)
        {
            insides.push_back(sortedIndex[searcherId][subInsides[nId]]);
        }
    }
}
