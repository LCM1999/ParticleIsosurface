#include <set>
#include "multi_level_researcher.h"

#include "hash_grid.h"

MultiLevelSearcher::MultiLevelSearcher(std::vector<Eigen::Vector3d>* particles, double* bounding, std::vector<double>* radiuses, double inf_factor)
{
    std::vector<std::vector<Eigen::Vector3d>> sortedParticles;
    maxRadius = *std::max_element(radiuses->begin(), radiuses->end());
    minRadius = *std::min_element(radiuses->begin(), radiuses->end());
    infFactor = inf_factor;
    particlesNum = particles->size();
    std::vector<std::pair<double, double>> bin_bounds;
    double bin_extent = minRadius * 0.5;
    int bins = std::max(int(ceil((maxRadius - minRadius) / bin_extent)), 1);
    bin_extent = (maxRadius - minRadius) / bins;
    for (size_t i = 0; i < bins; i++)
    {
        bin_bounds.push_back(
            std::pair<double, double>(
                minRadius+(i*bin_extent), 
                std::min(minRadius+((i+1)*bin_extent), maxRadius)));
    }
    auto whichBin = [&](const double r)
    {
        for (auto tit = bin_bounds.begin(); tit < bin_bounds.end(); tit++)
        {
            if (r >= tit->first && r <= tit->second)
                return static_cast<int>(std::distance(bin_bounds.begin(), tit));
        }
        return -1;
    };
    sortedParticles.clear();
    sortedIndex.clear();
    sortedParticles.resize(bins);
    sortedIndex.resize(bins);
    for (size_t i = 0; i < bins; i++)
    {
        sortedParticles[i].clear();
        sortedIndex[i].clear();
    }

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
        double temp_bounding [6] = {0.0f};
        temp_bounding[0] = bounding[0];
        temp_bounding[1] = bounding[1];
        temp_bounding[2] = bounding[2];
        temp_bounding[3] = bounding[3];
        temp_bounding[4] = bounding[4];
        temp_bounding[5] = bounding[5];
        searchers.push_back(new HashGrid(sortedParticles[i], temp_bounding, bin_bounds[i].second, inf_factor));
    }
    std::set<double> st(radiuses->begin(), radiuses->end());
    checkedRadiuses.assign(st.begin(), st.end());
    printf("   Seachers level: %d.\n", searchers.size());
}

void MultiLevelSearcher::GetNeighbors(const Eigen::Vector3d& pos, std::vector<int>& neighbors)
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

void MultiLevelSearcher::GetInBoxParticles(const Eigen::Vector3d& box1, const Eigen::Vector3d& box2, std::vector<int>& insides)
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
