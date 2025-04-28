#pragma once
#ifndef MULTI_LEVEL_SEACHER_GPU
#define MULTI_LEVEL_SEACHER_GPU

// #include <vector>
//#include <Eigen/Dense>
#include <coord_struct.h>
#include <thrust/host_vector.h>
#include "hash_grid_gpu.cuh"
#include <cfloat>

struct MultiLevelSearcherGPU
{
    std::vector<HashGridGPU*> searchers;
    std::vector<HashGridGPU*> h_searchers; // store the HashGridGPU* GPU pointers
    std::vector<int> maxRadiusParticleIds;
    float maxRadius = 0, minRadius = 0, avgRadius = 0;
    float infFactor;

    HOST MultiLevelSearcherGPU() {};
    HOST MultiLevelSearcherGPU(std::vector<cstoneOctree::Vec3f>* particles, float* bounding, std::vector<float>* radiuses, float inf_factor)
    {
        maxRadius = *std::max_element(radiuses->begin(), radiuses->end());   // * 1.01
        minRadius = *std::min_element(radiuses->begin(), radiuses->end());   // * 0.99
        infFactor = inf_factor;
        int particlesNum = particles->size();
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
        std::vector<std::vector<unsigned>>sortedIndex(bins);
    
        for (int i = 0; i < radiuses->size(); i++)
        {
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
            temp_bounding[0] = temp_bounding[2] = temp_bounding[4] = FLT_MAX;
            temp_bounding[1] = temp_bounding[3] = temp_bounding[5] = -FLT_MAX;
            for (auto pI: sortedIndex[i])
            {
                if (particles->at(pI).x < temp_bounding[0]) temp_bounding[0] = particles->at(pI).x;
                if (particles->at(pI).x > temp_bounding[1]) temp_bounding[1] = particles->at(pI).x;
                if (particles->at(pI).y < temp_bounding[2]) temp_bounding[2] = particles->at(pI).y;
                if (particles->at(pI).y > temp_bounding[3]) temp_bounding[3] = particles->at(pI).y;
                if (particles->at(pI).z < temp_bounding[4]) temp_bounding[4] = particles->at(pI).z;
                if (particles->at(pI).z > temp_bounding[5]) temp_bounding[5] = particles->at(pI).z;
            }
            unsigned int binRadiusId = *std::max_element(sortedIndex[i].begin(), sortedIndex[i].end(), 
                [&](unsigned int& a, unsigned int& b) {
                    return radiuses->at(a) < radiuses->at(b);
                });
            maxRadiusParticleIds.push_back(binRadiusId);
            searchers.push_back(new HashGridGPU(particles, radiuses, sortedIndex[i], temp_bounding, binRadiusId, inf_factor));
        }
        printf("   Seachers level: %d.\n", searchers.size());

        // assign data to h_searchers
        for(int i = 0; i < searchers.size(); i++){
            HashGridGPU* d_searcher_ptr;
            cudaMalloc(&d_searcher_ptr, sizeof(HashGridGPU));
            cudaMemcpy(&(d_searcher_ptr->CellSize), &(searchers[i]->CellSize), sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->Bounding), &(searchers[i]->Bounding), sizeof(float) * 6, cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->XYZCellNum), &(searchers[i]->XYZCellNum), sizeof(uint64_t) * 3, cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->CellNum), &(searchers[i]->CellNum), sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->particlesSize), &(searchers[i]->particlesSize), sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->PIndexes), &(searchers[i]->PIndexes), sizeof(unsigned*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->IndexList), &(searchers[i]->IndexList), sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->StartList), &(searchers[i]->StartList), sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->EndList), &(searchers[i]->EndList), sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->d_PIndexes), &(searchers[i]->d_PIndexes), sizeof(unsigned*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->d_IndexList), &(searchers[i]->d_IndexList), sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->d_StartList), &(searchers[i]->d_StartList), sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_searcher_ptr->d_EndList), &(searchers[i]->d_EndList), sizeof(int*), cudaMemcpyHostToDevice);
            // d_searcher_ptr->CellSize = searchers[i]->CellSize;
            // for(int i = 0; i < 5; i++){
            //     d_searcher_ptr->Bounding[i] = searchers[i]->Bounding[i];
            // }
            // for(int i = 0; i < 2; i++){
            //     d_searcher_ptr->XYZCellNum[i] = searchers[i]->XYZCellNum[i];
            // }
            // d_searcher_ptr->CellNum = searchers[i]->CellNum;
            // d_searcher_ptr->particlesSize = searchers[i]->particlesSize;
            // d_searcher_ptr->PIndexes = searchers[i]->PIndexes;
            // d_searcher_ptr->IndexList = searchers[i]->IndexList;
            // d_searcher_ptr->StartList = searchers[i]->StartList;
            // d_searcher_ptr->EndList = searchers[i]->EndList;
            // d_searcher_ptr->d_PIndexes = searchers[i]->d_PIndexes;
            // d_searcher_ptr->d_IndexList = searchers[i]->d_IndexList;
            // d_searcher_ptr->d_StartList = searchers[i]->d_StartList;
            // d_searcher_ptr->d_EndList = searchers[i]->d_EndList;

            h_searchers.push_back(d_searcher_ptr);
        }
    };

    HOST ~MultiLevelSearcherGPU() 
    {
        for (size_t i = 0; i < searchers.size(); i++)
        {
            delete searchers[i];
            searchers[i] = 0;
            cudaFree(h_searchers[i]);
        }
        
    };

    HOST int getSearchersNum() { return searchers.size(); }
    HOST HashGridGPU* getSearcher(int i) { return searchers[i]; }
    HOST std::vector<int> getMaxRadiusPaticleIds() {return std::vector<int>(maxRadiusParticleIds.begin(), maxRadiusParticleIds.end());}
    HOST_DEVICE float getMaxRadius() {return maxRadius;}
    HOST_DEVICE float getMinRadius() {return minRadius;}
    HOST_DEVICE float getAvgRadius() {return avgRadius;}

    HOST_DEVICE void GetNeighborsEstimate(const cstoneOctree::Vec3f& pos, int& estimate) 
    {
        for (auto& searcher : searchers)
        {
            searcher->GetPIdxEstimate(pos, estimate);
        }
    };
    
    HOST_DEVICE void GetNeighbors(const cstoneOctree::Vec3f& pos, int& numNeighbors, int ngmax, int* neighbors)
    {
        for (auto& searcher : searchers)
        {
            searcher->GetPIdxList(pos, numNeighbors, ngmax, neighbors);
        }
    };

    HOST_DEVICE void GetInBoxEstimate(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, int* estimateNeighborsNum){
        for(int i = 0; i < searchers.size(); i++){
            searchers[i]->GetInBoxEstimate(box1, box2, estimateNeighborsNum[i]);
        }
    }

    HOST_DEVICE void GetInBoxEstimate(const cstoneOctree::Vec3f& box1, const cstoneOctree::Vec3f& box2, int& insides)
    {
        for (auto& searcher : searchers)
        {
            searcher->GetInBoxEstimate(box1, box2, insides);
            printf("insides: %d\n", insides);
        }
    };
    
    HOST_DEVICE void GetInBoxParticles(cstoneOctree::Vec3f box1, cstoneOctree::Vec3f box2, int& numNeighbors, int ngmax, int* insides)
    {
        for (auto& searcher : searchers)
        {
            searcher->GetInBoxParticles(box1, box2, numNeighbors, ngmax, insides);
        }
    };
};

#endif