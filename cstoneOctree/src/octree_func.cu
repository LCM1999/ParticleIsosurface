#include <octree_func.h>

namespace cstoneOctree{

// Below are assistant functions for octree (vectors) genertions.

/**
 * @brief Identify the number of levels should have for the splitRange (split decision).
 *        Only for 64-bit unsigned integer.
 * 
 * @param splitRange Split decision number
 * @return HOST_DEVICE 
 */
HOST_DEVICE uint64_t log8ceil(const uint64_t splitRange){
    if(splitRange == 0) return 0;

    uint64_t lz = countLeadingZeros(splitRange - 1);
    return 21 - (lz - 1) / 3;
}

/*! @brief Extract the n-th octal digit from an SFC key, starting from the most significant. 
 *         Only for 64-bit unsigned integer.
 *
 * @param code       Input SFC key code
 * @param position   Which digit place to extract. Return values will be meaningful for
 *                   @p position in [1:22] for 64-bit keys and
 *                   will be zero otherwise, but a value of 0 for @p position can also be specified
 *                   to detect whether the 31st or 63rd bit for the last cornerstone is non-zero.
 *                   (The last cornerstone has a value of 2^63)
 * @return           The value of the digit at place @p position
 *
 * The position argument correspondence to octal digit places has been chosen such that
 * octalDigit(code, pos) returns the octant at octree division level pos.
 */
//For code = 111110010000 for example, position = level = 3, maxTreeLevel=4 here, so code >> (3u * (maxTreeLevel<KeyType>{} - position)) = 111110010, it removes the ending zeroes
// it & 7u equals 010, which only gets the last octaldigit

HOST_DEVICE unsigned octalDigit(uint64_t code, unsigned position){
    return (code >> (3u * (21 - position))) & 7u;
}

/*! @brief compute the maximum range of an octree node at a given subdivision level. Only for 64-bit unsigned integer.
 *
 * @param  treeLevel  octree subdivision level
 * @return            the range
 *
 * At treeLevel 0, the range is the entire 63 bits used in the SFC code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
// HOST_DEVICE inline uint64_t nodeRange(const uint64_t treeLevel){
//     assert(treeLevel <= 21);
//     unsigned shifts = 3 * (21 - treeLevel);

//     return uint64_t(1ull << shifts);
// }

/*! @brief return octree subdivision level corresponding to codeRange (one node's begin and end morton code). The root node is level 0.
 *
 * @param range      input morton code range
 * @return           octree subdivision level 0-21 (64-bit)
 */
// for codeRange = 0001000000 for example, codeRange-1=0000111111, result is (4-1)/3 = 1
HOST_DEVICE uint64_t treeLevel(const uint64_t range){
    // std::cout << "range: " << range << std::endl;
    // printf("range: %llu\n", (unsigned long long)range);
    if(!isPowerOf8(range)) printf("The error range: %llu is not the power of 8\n", (unsigned long long)range);
    assert(isPowerOf8(range));
    return (countLeadingZeros(range - 1) - 1) / 3;
}


/**
 * @brief Check whether the number is power of 8. Only for 64-bit unsigned integer.
 * 
 * @param range range for morton code 
 * @return HOST_DEVICE  
 */
HOST_DEVICE bool isPowerOf8(uint64_t range){
    uint64_t lz = countLeadingZeros(range - 1) - 1;
    return lz % 3 == 0 && !(range & (range - 1));
}

/*! @brief convert a plain SFC key into the placeholder bit format (Warren-Salmon 1993)
 *
 * @param code             input SFC key
 * @param prefixLength     number of leading bits which are part of the code
 * @return                 code shifted by trailing zeros and prepended with 1-bit
 *
 * Example: encodePlaceholderBit(06350000000, 9) -> 01635 (in octal)
 */ 
HOST_DEVICE uint64_t encodePlaceholderBit(uint64_t code, int prefixLength){
    //if the code is 06350000000 and prefixLength=9 here (3*level), nShifts = 3 * maxTreeLevel<int32_t>{} - prefixLength=3*10-9=21
    //0 at the start denotes octal-digit form, not part of the coding
    int nShifts             = 3 * 21 - prefixLength;
    //ret = 0635
    uint64_t ret             = code >> nShifts;
    //placeHolderMask = 01000
    uint64_t placeHolderMask = uint64_t(1) << prefixLength;
    //placeHolderMask | ret = 01000 | 0635 = 01635
    return placeHolderMask | ret;
}

/*! @brief decode an SFC key in Warren-Salmon placeholder bit format
 *
 * @param code       input SFC key with 1-bit prepended
 * @return           SFC-key without 1-bit and shifted to most significant bit
 *
 * Inverts encodePlaceholderBit.
 */
HOST_DEVICE uint64_t decodePlaceholderBit(uint64_t code){
    int prefixLength        = decodePrefixLength(code);
    uint64_t placeHolderMask = uint64_t(1) << prefixLength;
    uint64_t ret             = code ^ placeHolderMask;

    return ret << (3 * 21 - prefixLength);
}

//! @brief returns the number of key-bits in the input @p code
// HOST_DEVICE inline uint64_t decodePrefixLength(uint64_t code)
// {
//     return 8 * sizeof(uint64_t) - 1 - countLeadingZeros(code);
// }

__global__ void makeSplitsDecisionsGPUKernel(uint64_t* treeDevice, uint64_t* countsDevice, int bucketSize, uint64_t* nodeOpsDevice, int treeSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= treeSize) return;

    uint64_t thisNode = treeDevice[idx];
    uint64_t mortonRange = treeDevice[idx + 1] - thisNode;
    uint64_t level = treeLevel(mortonRange);
    int siblingIdx = -1;

    // Calculate the split decision
    if (level == 0){
        siblingIdx = -1;
    }
    else{
        // check for split
        siblingIdx = octalDigit(thisNode, level);
        bool siblings =
            (treeDevice[idx - siblingIdx + 8] == treeDevice[idx - siblingIdx] + nodeRange(level - 1));
        if (!siblings)
        {
            siblingIdx = -1;
        }
    }
    /////////////////////////////////////////////////// could be common

    // 8 siblings next to each other, node can potentially be merged
    if (siblingIdx > 0){
        uint64_t parentCount = 0;
        for (unsigned i = 0; i < 8; i++)
        {
            // from first node in sibling group
            parentCount += countsDevice[idx - siblingIdx + i];
        }
        bool countMerge = parentCount <= bucketSize;
        //  0 means the nodes can be merged to its parent
        if (countMerge){
            nodeOpsDevice[idx] = 0;
            return;
        }
    }

    // calculate the split level
    if (countsDevice[idx] > bucketSize * 512 && level + 3 < 21){
        nodeOpsDevice[idx] = 4096;
        return;
    }
    if (countsDevice[idx] > bucketSize * 64 && level + 2 < 21) {
        nodeOpsDevice[idx] = 512;
        return;
    }
    if (countsDevice[idx] > bucketSize * 8 && level + 1 < 21){
        nodeOpsDevice[idx] = 64;
        return;
    }
    if (countsDevice[idx] > bucketSize && level < 21){
        nodeOpsDevice[idx] = 8;
        return;
    } 

    // default level
    nodeOpsDevice[idx] = 1;
}


__global__ void updateTreeArrayGPUKernel(uint64_t* nodeOpsLayoutDevice, uint64_t* oldTreeDevice, uint64_t* newTreeDevice, int treeSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= treeSize) return;


    uint64_t opCode = nodeOpsLayoutDevice[idx + 1] - nodeOpsLayoutDevice[idx];
    uint64_t thisNode = oldTreeDevice[idx];
    uint64_t range = oldTreeDevice[idx + 1] - thisNode;
    uint64_t level = treeLevel(range);

    // Calculate the new node index based on the split decision in nodeOps array
    int newNodeIdx = nodeOpsLayoutDevice[idx];

    // If the node is a leaf node, just copy it to the new tree
    if(opCode == 1){
        newTreeDevice[newNodeIdx] = thisNode;
    }
    // If the split decision is greater than 8, begin to split the node
    else if(opCode == 8){
        for(int sibling = 0; sibling < 8; sibling++){
            newTreeDevice[newNodeIdx + sibling] = thisNode + sibling * nodeRange(level + 1);
        }
    }
    else if(opCode > 8){
        uint64_t levelDiff = log8ceil(opCode);
        for(int sibling = 0; sibling < opCode; sibling++){
            newTreeDevice[newNodeIdx + sibling] = thisNode + sibling * nodeRange(level + levelDiff);
        }
    }

}

__global__ void findCoverNodesGPUKernel(uint64_t* treeDevice, uint64_t* mortonCodesDeviceStart, uint64_t* mortonCodesDeviceEnd, uint64_t* coverNodesDevice, int numNodes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= 1) return;


    
    if(mortonCodesDeviceStart != mortonCodesDeviceEnd){
        coverNodesDevice[0] = cal::upper_bound(treeDevice, treeDevice + numNodes, *mortonCodesDeviceStart) - treeDevice - 1;
        coverNodesDevice[1] = cal::upper_bound(treeDevice, treeDevice + numNodes, *(mortonCodesDeviceEnd - 1)) - treeDevice;
    }
    else{
        coverNodesDevice[0] = numNodes;
        coverNodesDevice[1] = numNodes;
    }
}

// only change the necessary counts
__global__ void updateNodeCountsGPUKernel(uint64_t* treeDevice, int nodeNum, uint64_t* mortonCodesDeviceStart, uint64_t* mortonCodesDeviceEnd, uint64_t* countsDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nodeNum) return;

    uint64_t nodeStartVal = treeDevice[idx];
    uint64_t nodeEndVal = treeDevice[idx + 1]; 

    auto rangeStart = cal::lower_bound(mortonCodesDeviceStart, mortonCodesDeviceEnd, nodeStartVal);
    auto rangeEnd = cal::lower_bound(mortonCodesDeviceStart, mortonCodesDeviceEnd, nodeEndVal);
    size_t count = rangeEnd - rangeStart;
    countsDevice[idx] = count;
}


__global__ void createUnsortedLayoutKernel(uint64_t* leavesDevice, int numInternalNodes, int numLeafNodes, uint64_t* prefixesDevice, int* internalToLeafDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numLeafNodes) return;

    uint64_t curr_key = leavesDevice[idx];
    unsigned curr_level = treeLevel(leavesDevice[idx + 1] - curr_key);
    prefixesDevice[idx + numInternalNodes] = encodePlaceholderBit(curr_key, 3 * curr_level); 
    internalToLeafDevice[idx + numInternalNodes] = idx + numInternalNodes;

    unsigned prefixLength = countLeadingZeros(curr_key ^ leavesDevice[idx + 1]) - 1;
    const int idx_map[8] = {0, -1, -2, -3, 3, 2, 1, 0};
    if(prefixLength % 3 == 0 && idx < numLeafNodes - 1){ // Make sure the key and prefix length is valid
        // map a binary node index to an octree node index
        int ret = 0;
        for(int l = 1; l <= curr_level + 1; l++){
            const int digit = octalDigit(curr_key, l);
            ret += idx_map[digit];
        }
        int octIndex = (idx + ret) / 7;

        prefixesDevice[octIndex]       = encodePlaceholderBit(curr_key, prefixLength);
        internalToLeafDevice[octIndex] = octIndex;
    }
}

__global__ void invertOrderKernel(int* internalToLeafDevice, int* leafToInternalDevice, int numNodes, int numInternalNodes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numNodes) return;

    leafToInternalDevice[internalToLeafDevice[idx]] = idx;
    internalToLeafDevice[idx] -= numInternalNodes;

}

__global__ void getLevelRangeKernel(uint64_t* prefixesDevice, int numNodes, int* levelRangeDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= 21 + 1 + 1) return;

    if(idx == 22){
        levelRangeDevice[idx] = numNodes;
    }
    else{
        auto it = cal::lower_bound(prefixesDevice, prefixesDevice + numNodes, encodePlaceholderBit(uint64_t(0), 3 * idx));
        levelRangeDevice[idx] = int(it - prefixesDevice);
    }
    
}

__global__ void linkOctreeKernel(uint64_t* prefixesDevice, int numInternalNodes, int* leafToInternalDevice, int* levelRangeDevice, int* childOffsetsDevice, int* parentsDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numInternalNodes) return;

    int idxA = leafToInternalDevice[idx]; // 0 - numInternal: unsorted internal nodes
    uint64_t prefix = prefixesDevice[idxA];
    uint64_t nodeKey = decodePlaceholderBit(prefix);
    uint64_t prefixLength = decodePrefixLength(prefix);
    uint64_t level = prefixLength / 3;
    assert(level < 21);
    uint64_t childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

    int leafSearchStart = levelRangeDevice[level + 1];
    int leafSearchEnd = levelRangeDevice[level + 2];
    int childIdx = cal::lower_bound(prefixesDevice+ leafSearchStart, prefixesDevice + leafSearchEnd, childPrefix) - prefixesDevice;

    if(childIdx != leafSearchEnd && childPrefix == prefixesDevice[childIdx]){
        childOffsetsDevice[idxA] = childIdx;
        // We only store the parent once for every group of 8 siblings.
        // This works as long as each node always has 8 siblings.
        // Subtract one because the root has no siblings.
        parentsDevice[(childIdx - 1) / 8] = idxA;
    } 
}

__global__ void calculateNodeCentersAndSizesKernel(uint64_t* prefixesDevice, int prefixesDeviceSize, Vec3f* centersDevice, Vec3f* sizesDevice, Box* boxDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= prefixesDeviceSize) return;

    uint64_t prefix = prefixesDevice[idx];
    uint64_t start_key = decodePlaceholderBit(prefix);
    unsigned level = decodePrefixLength(prefix) / 3;

    // constexpr int maxCoord = 1u << 21;
    unsigned cubeLength = (1u << (21 - level));
    int ivec_x = compactBits(start_key >> 2), ivec_y = compactBits(start_key >> 1), ivec_z = compactBits(start_key);
    Vec3<int> ivec = Vec3<int>(ivec_x, ivec_y, ivec_z);

    IBox nodeBox(ivec.x, ivec.x + cubeLength, ivec.y, ivec.y + cubeLength, ivec.z, ivec.z + cubeLength);
    cal::tie(centersDevice[idx], sizesDevice[idx]) = centerAndSizeGPU(&nodeBox, boxDevice);
}

__global__ void findNeighborsKernel(Vec3f* coordsDevice, int coords_size, float* radiusesDevice, OctreeNs* octreeNs, Box* boxDevice, int ngmax, int* neighborsDevice, int* numNeighborsDevice){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= coords_size) return;

    // assign the partial pointers to each threads its owns
    neighborsDevice = neighborsDevice + idx * ngmax;

    float x = coordsDevice[idx].x;
    float y = coordsDevice[idx].y;
    float z = coordsDevice[idx].z;
    Vec3f particle = coordsDevice[idx];
    float h = radiusesDevice[idx];
    int neighbors_num = 0;

    float radiusSquare = 4.0 * h * h;

    auto overlaps = [particle, radiusSquare, centers = octreeNs->centers, sizes = octreeNs->sizes, boxDevice](int idx){
        auto nodeCenter = centers[idx];
        auto nodeSize = sizes[idx];
        Vec3f min_distance = cal::minDistanceGPU(particle, nodeCenter, nodeSize);
        float squared_distance = min_distance.x * min_distance.x + min_distance.y * min_distance.y + min_distance.z * min_distance.z;
        return squared_distance < radiusSquare;
    };

    auto searchBox = [idx, particle, radiusSquare, octreeNs, coordsDevice, ngmax, neighborsDevice, &neighbors_num, boxDevice](int i){
        int leafIdx    = octreeNs->internalToLeaf[i];
        int firstParticle = octreeNs->layout[leafIdx];
        int lastParticle  = octreeNs->layout[leafIdx + 1];

        for (int j = firstParticle; j < lastParticle; ++j)
        {
            if (j == idx) { continue; }
            Vec3f diffs = particle - coordsDevice[j];
            float squared_distance = diffs.x * diffs.x + diffs.y * diffs.y + diffs.z * diffs.z;
            if(squared_distance < radiusSquare){
                if (neighbors_num < ngmax) { neighborsDevice[neighbors_num] = j; }
                neighbors_num++;
            }
        }



    };        

    singleTraversal(octreeNs->childOffsets, overlaps, searchBox);
    numNeighborsDevice[idx] = neighbors_num;

}


}
