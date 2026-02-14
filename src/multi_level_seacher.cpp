#include <set>
#include "multi_level_searcher.h"
#include <cfloat>
#include "hash_grid.h"

MultiLevelSearcher::MultiLevelSearcher(
    std::vector<cstoneOctree::Vec3f>* particles, 
    float* bounding, std::vector<float>* radiuses, 
    float inf_factor, float scale)
{
    maxRadius = *std::max_element(radiuses->begin(), radiuses->end());   // * 1.01
    minRadius = *std::min_element(radiuses->begin(), radiuses->end());   // * 0.99
    infFactor = inf_factor;

    if (!(scale > 1.0f)) {
        scale = 1.0f;
    }
    std::vector<std::pair<float, float>> bin_bounds;
    
    if (maxRadius <= minRadius || scale == 1.0f) {
        bin_bounds.emplace_back(minRadius, maxRadius);
    } else {
        float bmin = minRadius;

        const float eps = 1e-20f;
        if (bmin < eps) bmin = eps;

        while (true) {
            float bmax = bmin * scale;

            if (bmax >= maxRadius) {
                bin_bounds.emplace_back(bmin, maxRadius);
                break;
            }

            bin_bounds.emplace_back(bmin, bmax);
            bmin = bmax;

            if (bin_bounds.size() > 4096) {
                bin_bounds.back().second = maxRadius;
                break;
            }
        }
    }

    const int bins = static_cast<int>(bin_bounds.size());
    std::vector<std::vector<unsigned>> sortedIndex(bins);

    auto whichBin = [&](float r) -> int
    {
        if (r <= bin_bounds.front().first) return 0;
        if (r >= bin_bounds.back().second) return bins - 1;

        for (int i = 0; i < bins; ++i) {
            const float lo = bin_bounds[i].first;
            const float hi = bin_bounds[i].second;
            const bool last = (i == bins - 1);

            if (r >= lo && (r < hi || (last && r <= hi)))
                return i;
        }
        return bins - 1; // 理论上不会到这，兜底
    };

    for (int i = 0; i < (int)radiuses->size(); i++) {
        const int b = whichBin(radiuses->at(i));
        sortedIndex[b].push_back(i);
    }

    for (int i = 0; i < bins; i++)
    {
        if (sortedIndex[i].empty()) continue;

        float temp_bounding[6] = {0.0f};
        for (size_t j = 0; j < 6; j++) temp_bounding[j] = bounding[j];

        for (auto pI : sortedIndex[i])
        {
            temp_bounding[0] = std::min(temp_bounding[0], particles->at(pI).x);
            temp_bounding[1] = std::max(temp_bounding[1], particles->at(pI).x);
            temp_bounding[2] = std::min(temp_bounding[2], particles->at(pI).y);
            temp_bounding[3] = std::max(temp_bounding[3], particles->at(pI).y);
            temp_bounding[4] = std::min(temp_bounding[4], particles->at(pI).z);
            temp_bounding[5] = std::max(temp_bounding[5], particles->at(pI).z);
        }

        unsigned int binRadiusId = *std::max_element(
            sortedIndex[i].begin(), sortedIndex[i].end(),
            [&](unsigned int& a, unsigned int& b) {
                return radiuses->at(a) < radiuses->at(b);
            });

        maxRadiusParticleIds.push_back(binRadiusId);
        searchers.push_back(new HashGrid(particles, radiuses, sortedIndex[i],
                                         temp_bounding, binRadiusId, inf_factor));
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
