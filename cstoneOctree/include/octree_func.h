#pragma once


#include <cassert>
#include <cmath>
#include <morton.h>
#include <cuda_def.h>
#include <coord_struct.h>
#include <calculator.h>
#include <box.h>
#include <utils_helper.h>
#include <cuda_runtime.h>

namespace cstoneOctree{

// a struct to store the octree information for neighbor search
struct OctreeNs{
    const uint64_t* prefixes;
    const TreeNodeIndex* childOffsets;
    const TreeNodeIndex* internalToLeaf;
    const TreeNodeIndex* levelRange;

    const int* layout; // index of first particle for each leaf node
    const Vec3f* centers;
    const Vec3f* sizes;

    HOST_DEVICE OctreeNs(const uint64_t* prefixes, const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf, const TreeNodeIndex* levelRange, const int* layout, const Vec3f* centers, const Vec3f* sizes)
        : prefixes(prefixes), childOffsets(childOffsets), internalToLeaf(internalToLeaf), levelRange(levelRange), layout(layout), centers(centers), sizes(sizes) {}
    

};

// deep first traversal of the octree
template<class C, class A>
HOST_DEVICE void singleTraversal(const int* childOffsets, C&& continuationCriterion, A&& endpointAction)
{
    bool descend = continuationCriterion(0);
    if (!descend) return;

    if (childOffsets[0] == 0)
    {
        // root node is already the endpoint
        endpointAction(0);
        return;
    }

    TreeNodeIndex stack[128];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    do
    {
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = childOffsets[node] + octant;
            bool descend        = continuationCriterion(child);
            if (descend)
            {
                if (childOffsets[child] == 0) // ensure this is a leaf node
                {
                    // endpoint reached with child is a leaf node
                    endpointAction(child);
                }
                else
                {
                    assert(stackPos < 128);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

// Below are assistant functions for octree (vectors) genertions.
HOST_DEVICE uint64_t log8ceil(const uint64_t splitRange);

HOST_DEVICE unsigned octalDigit(uint64_t code, unsigned position);

HOST_DEVICE inline uint64_t nodeRange(const uint64_t treeLevel){
    assert(treeLevel <= 21);
    unsigned shifts = 3 * (21 - treeLevel);

    return uint64_t(1ull << shifts);
}

HOST_DEVICE uint64_t treeLevel(const uint64_t range);

HOST_DEVICE bool isPowerOf8(uint64_t n);

/**
 * @brief count how many of zeros before the first 1 in the binary representation of a number (leading zeros)
 * 
 * @param num morton code
 * @return the number of leading zeros
 */
HOST_DEVICE inline int countLeadingZeros(uint64_t num){
    if (num == 0)
    {
        return 64;
    }

    int count = 0;
    uint64_t mask = 1ULL << 63;

    while ((num & mask) == 0)
    {
        count++;
        mask >>= 1;
    }

    return count;
}

HOST_DEVICE uint64_t encodePlaceholderBit(uint64_t code, int prefixLength);

HOST_DEVICE uint64_t decodePlaceholderBit(uint64_t code);

HOST_DEVICE inline uint64_t decodePrefixLength(uint64_t code){
    return 8 * sizeof(uint64_t) - 1 - countLeadingZeros(code);
}


// Below are implementation of octree generation functions in CPU

uint64_t makeSplitsDecisionsFunctorCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, int idx);

void makeSplitsDecisionsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, std::vector<uint64_t>& nodeOps, int idx);

void makeSplitsDecisionsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, std::vector<uint64_t>& nodeOps);

void updateTreeArrayCPU(std::vector<uint64_t>& nodeOpsLayout, std::vector<uint64_t>& oldTree, std::vector<uint64_t>& newTree, int idx);

void updateTreeArrayCPU(std::vector<uint64_t>& nodeOpsLayout, std::vector<uint64_t>& oldTree, std::vector<uint64_t>& newTree);

void findCoverNodesCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, std::vector<uint64_t>& coverNodes); 

void updateNodeCountsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, std::vector<uint64_t>& counts, std::vector<uint64_t>& coverNodes);

uint64_t updateNodeCountsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, uint64_t idx);

void createUnsortedLayoutCPU(std::vector<uint64_t>& leaves, int numInternalNodes, int numLeafNodes, std::vector<uint64_t>& prefixes, std::vector<int>& internalToLeaf);

void createUnsortedLayoutCPU(std::vector<uint64_t>& leaves, int numInternalNodes, int numLeafNodes, std::vector<uint64_t>& prefixes, std::vector<int>& internalToLeaf, int idx);

void invertOrderCPU(std::vector<int>& internalToLeaf, std::vector<int>& leafToInternal, int numNodes, int numInternalNodes);

void invertOrderCPU(std::vector<int>& internalToLeaf, std::vector<int>& leafToInternal, int numNodes, int numInternalNodes, int idx);

void getLevelRangeCPU(std::vector<uint64_t>& prefixes, int numNodes, std::vector<int>& levelRange);	

void getLevelRangeCPU(std::vector<uint64_t>& prefixes, int numNodes, std::vector<int>& levelRange, int level);	

void linkOctreeCPU(std::vector<uint64_t>& prefixes, int numInternalNodes, std::vector<int>& leafToInternal, std::vector<int>& levelRange, std::vector<int>& childOffsets, std::vector<int>& parents);

void linkOctreeCPU(std::vector<uint64_t>& prefixes, int numInternalNodes, std::vector<int>& leafToInternal, std::vector<int>& levelRange, std::vector<int>& childOffsets, std::vector<int>& parents, int idx);

void calculateNodeCentersAndSizesCPU(std::vector<uint64_t>& prefixes, std::vector<Vec3f>& centers, std::vector<Vec3f>& sizes, Box& box);

void calculateNodeCentersAndSizesCPU(std::vector<uint64_t>& prefixes, std::vector<Vec3f>& centers, std::vector<Vec3f>& sizes, Box& box, int idx);

void findNeighborsCPU(std::vector<Vec3f> particles, std::vector<float>& radiuses, OctreeNs& octreeNs, Box& box, int ngmax, std::vector<int>& neighbors, std::vector<int>& numNeighbors);

int findNeighborsCPU(int idx, std::vector<Vec3f> particles, std::vector<float>& radiuses, OctreeNs& octreeNs, Box& box, int ngmax, int* neighbors);


// Below are implementation of octree generation functions in GPU
__global__ void makeSplitsDecisionsGPUKernel(uint64_t* treeDevice, uint64_t* countsDevice, int bucketSize, uint64_t* nodeOpsDevice, int treeSize);

__global__ void updateTreeArrayGPUKernel(uint64_t* nodeOpsLayoutDevice, uint64_t* oldTreeDevice, uint64_t* newTreeDevice, int treeSize);

__global__ void findCoverNodesGPUKernel(uint64_t* treeDevice, uint64_t* mortonCodesDeviceStart, uint64_t* mortonCodesDeviceEnd, uint64_t* coverNodesDevice, int numNodes);

__global__ void updateNodeCountsGPUKernel(uint64_t* treeDevice, int nodeNum, uint64_t* mortonCodesDeviceStart, uint64_t* mortonCodesDeviceEnd, uint64_t* countsDevice);

__global__ void createUnsortedLayoutKernel(uint64_t* leavesDevice, int numInternalNodes, int numLeafNodes, uint64_t* prefixesDevice, int* internalToLeafDevice);

__global__ void invertOrderKernel(int* internalToLeafDevice, int* leafToInternalDevice, int numNodes, int numInternalNodes);

__global__ void getLevelRangeKernel(uint64_t* prefixesDevice, int numNodes, int* levelRangeDevice);

__global__ void linkOctreeKernel(uint64_t* prefixesDevice, int numInternalNodes, int* leafToInternalDevice, int* levelRangeDevice, int* childOffsetsDevice, int* parentsDevice);

__global__ void calculateNodeCentersAndSizesKernel(uint64_t* prefixesDevice, int prefixesDeviceSize, Vec3f* centersDevice, Vec3f* sizesDevice, Box* boxDevice);

__global__ void findNeighborsKernel(Vec3f* coordsDevice, int coords_size, float* radiusesDevice, OctreeNs* octreeNs, Box* boxDevice, int ngmax, int* neighborsDevice, int* numNeighborsDevice);

}