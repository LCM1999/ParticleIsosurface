#include <octree_func.h>
#include <bitset>
namespace cstoneOctree{

/* --------------------------------------------------- */
// Below are implementation of octree generation functions


/**
 * @brief Calculate the split decision for each node in the octree array.
 * 
 * @param tree Octree array
 * @param counts An array which stores how many points (or other stuff) are in each node
 * @param bucketSize maximum number of points (or other stuff) in a node
 * @param idx octree index
 * @return uint64_t 
 */
uint64_t makeSplitsDecisionsFunctorCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, int idx){
    uint64_t thisNode = tree[idx];
    uint64_t mortonRange = tree[idx + 1] - thisNode;
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
            (tree[idx - siblingIdx + 8] == tree[idx - siblingIdx] + nodeRange(level - 1));
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
            parentCount += counts[idx - siblingIdx + i];
        }
        bool countMerge = parentCount <= bucketSize;
        //  0 means the nodes can be merged to its parent
        if (countMerge) { return 0; }
    }

    // calculate the split level
    if (counts[idx] > bucketSize * 512 && level + 3 < 21) { return 4096; }
    if (counts[idx] > bucketSize * 64 && level + 2 < 21) { return 512; }
    if (counts[idx] > bucketSize * 8 && level + 1 < 21) { return 64; }
    if (counts[idx] > bucketSize && level < 21) { return 8; }

    // default level
    return 1;
}

void makeSplitsDecisionsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, std::vector<uint64_t>& nodeOps, int idx){
    nodeOps[idx] = makeSplitsDecisionsFunctorCPU(tree, counts, bucketSize, idx);
}

void makeSplitsDecisionsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& counts, int bucketSize, std::vector<uint64_t>& nodeOps){
    int n = tree.size() - 1;
    for(int i = 0; i < n; i++){
        makeSplitsDecisionsCPU(tree, counts, bucketSize, nodeOps, i);
    }
    return;
}

void updateTreeArrayCPU(std::vector<uint64_t>& nodeOpsLayout, std::vector<uint64_t>& oldTree, std::vector<uint64_t>& newTree){
    int n = oldTree.size() - 1;

    for(int i = 0; i < n; i++){
        updateTreeArrayCPU(nodeOpsLayout, oldTree, newTree, i);
    }
}

/**
 * @brief Update the octree array based on the split decisions
 * 
 * @param nodeOpsLayout Stores the index of each node 
 * @param oldTree Current octree array
 * @param newTree New octree array
 * @param idx Octree index
 */
void updateTreeArrayCPU(std::vector<uint64_t>& nodeOpsLayout, std::vector<uint64_t>& oldTree, std::vector<uint64_t>& newTree, int idx){
    uint64_t opCode = nodeOpsLayout[idx + 1] - nodeOpsLayout[idx];
    uint64_t thisNode = oldTree[idx];
    uint64_t range = oldTree[idx + 1] - thisNode;
    uint64_t level = treeLevel(range);

    // Calculate the new node index based on the split decision in nodeOps array
    int newNodeIdx = nodeOpsLayout[idx];

    // If the node is a leaf node, just copy it to the new tree
    if(opCode == 1){
        newTree[newNodeIdx] = thisNode;
    }
    // If the split decision is greater than 8, begin to split the node
    else if(opCode == 8){
        for(int sibling = 0; sibling < 8; sibling++){
            newTree[newNodeIdx + sibling] = thisNode + sibling * nodeRange(level + 1);
        }
    }
    else if(opCode > 8){
        uint64_t levelDiff = log8ceil(opCode);
        for(int sibling = 0; sibling < opCode; sibling++){
            newTree[newNodeIdx + sibling] = thisNode + sibling * nodeRange(level + levelDiff);
        }
    }
}

/**
 * @brief Find the nodes that should cover all the morton codes
 * 
 * @param tree octree array
 * @param mortonCodes coordinates' morton codes
 * @param coverNodes octree node indices that should cover all the morton codes
 */
void findCoverNodesCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, std::vector<uint64_t>& coverNodes){
    int numNodes = tree.size() - 1;
    auto mortonCodesStart = mortonCodes.begin();
    auto mortonCodesEnd = mortonCodes.end();
    if(mortonCodesStart != mortonCodesEnd){
        coverNodes[0] = cal::upper_bound(tree.data(), tree.data() + numNodes, *mortonCodesStart) - tree.data() - 1;
        coverNodes[1] = cal::upper_bound(tree.data(), tree.data() + numNodes, *(mortonCodesEnd - 1)) - tree.data();
    }
    else{
        coverNodes[0] = numNodes;
        coverNodes[1] = numNodes;
    }
};

void updateNodeCountsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, std::vector<uint64_t>& counts, std::vector<uint64_t>& coverNodes){
    uint64_t treeStart = coverNodes[0];
    uint64_t treeNodes = coverNodes[1] - coverNodes[0];
    auto countsStart = counts.data() + coverNodes[0];

    #pragma omp parallel for
    for(uint64_t i = treeStart; i < treeStart + treeNodes; i++){
        counts[i] = updateNodeCountsCPU(tree, mortonCodes, i);
    }
}

uint64_t updateNodeCountsCPU(std::vector<uint64_t>& tree, std::vector<uint64_t>& mortonCodes, uint64_t idx){
    // count particles in range
    int mortonCodesSize = mortonCodes.size();
    uint64_t nodeStartVal = tree[idx];
    uint64_t nodeEndVal = tree[idx + 1];
    auto rangeStart = cal::lower_bound(mortonCodes.data(), mortonCodes.data() + mortonCodesSize, nodeStartVal);
    auto rangeEnd = cal::lower_bound(mortonCodes.data(), mortonCodes.data() + mortonCodesSize, nodeEndVal);
    size_t count = rangeEnd - rangeStart;
    return count;
}

void createUnsortedLayoutCPU(std::vector<uint64_t>& leaves, int numInternalNodes, int numLeafNodes, std::vector<uint64_t>& prefixes, std::vector<int>& internalToLeaf){
    #pragma omp parallel for
    for(int i = 0; i < numLeafNodes; i++){
        createUnsortedLayoutCPU(leaves, numInternalNodes, numLeafNodes, prefixes, internalToLeaf, i);
    }
}

/*! @brief Combine internal and leaf tree parts into a single array with the nodeKey prefixes
 *
 * @param[in]  leaves (tree)     store each leaf nodes, cornerstone SFC keys length numLeafNodes + 1
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  numLeafNodes      total number of nodes
 * @param[out] prefixes          output octree SFC keys, length @p numInternalNodes + numLeafNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] internalToLeaf    iota 0,1,2,3,... sequence for later use, length same as @p prefixes
 */
void createUnsortedLayoutCPU(std::vector<uint64_t>& leaves, int numInternalNodes, int numLeafNodes, std::vector<uint64_t>& prefixes, std::vector<int>& internalToLeaf, int idx){
    uint64_t curr_key = leaves[idx];
    unsigned curr_level = treeLevel(leaves[idx + 1] - curr_key);
    prefixes[idx + numInternalNodes] = encodePlaceholderBit(curr_key, 3 * curr_level); // calculate the leaf node's prefix value
    // if(prefixes[idx + numInternalNodes] == 0){
    //     printf("prefixes is 0! ");
    //     printf("the original key is %lu\n", curr_key);
    // }
    internalToLeaf[idx + numInternalNodes] = idx + numInternalNodes;

    unsigned prefixLength = countLeadingZeros(curr_key ^ leaves[idx + 1]) - 1;
    const int idx_map[8] = {0, -1, -2, -3, 3, 2, 1, 0};
    if(prefixLength % 3 == 0 && idx < numLeafNodes - 1){ // Make sure the key and prefix length is valid
        // map a binary node index to an octree node index
        int ret = 0;
        // traverse over the 3 bits of the octree node index (traverse each levels of the octree)
        for(int l = 1; l <= curr_level + 1; l++){
            const int digit = octalDigit(curr_key, l); // get the current traversed level's 3 bits info (3 bits = 1 octal)
            ret += idx_map[digit];
        }
        int octIndex = (idx + ret) / 7;

        prefixes[octIndex]       = encodePlaceholderBit(curr_key, prefixLength);
        // if(prefixes[octIndex] == 0){
        //     printf("prefixes is 0! ");
        //     printf("the original key is %lu\n", curr_key);
        // }
        internalToLeaf[octIndex] = octIndex;
    }
}

void invertOrderCPU(std::vector<int>& internalToLeaf, std::vector<int>& leafToInternal, int numNodes, int numInternalNodes){
    #pragma omp parallel for
    for(int i = 0; i < numNodes; i++){
        invertOrderCPU(internalToLeaf, leafToInternal, numNodes, numInternalNodes, i);
    }
}

void invertOrderCPU(std::vector<int>& internalToLeaf, std::vector<int>& leafToInternal, int numNodes, int numInternalNodes, int idx){
    leafToInternal[internalToLeaf[idx]] = idx;
    internalToLeaf[idx] -= numInternalNodes;
}

void getLevelRangeCPU(std::vector<uint64_t>& prefixes, int numNodes, std::vector<int>& levelRange){
    #pragma omp parallel for
    for(int level = 0; level < 21 + 1 + 1; level++){
        getLevelRangeCPU(prefixes, numNodes, levelRange, level);
    }
    levelRange[22] = numNodes;
}
/**
 * @brief Calculate node range for each tree level
 * 
 * @param prefixes SFC keys combined with placeholder bits
 * @param numNodes The total number of tree nodes 
 * @param levelRange vectors contains node range for each tree level
 * @param level Current tree level
 */
void getLevelRangeCPU(std::vector<uint64_t>& prefixes, int numNodes, std::vector<int>& levelRange, int level){
    auto it = cal::lower_bound(prefixes.data(), prefixes.data() + numNodes, encodePlaceholderBit(uint64_t(0), 3 * level));
    levelRange[level] = int(it - prefixes.data());
}


void linkOctreeCPU(std::vector<uint64_t>& prefixes, int numInternalNodes, std::vector<int>& leafToInternal, std::vector<int>& levelRange, std::vector<int>& childOffsets, std::vector<int>& parents){
    #pragma omp parallel for
    for(int i = 0; i < numInternalNodes; i++){
        linkOctreeCPU(prefixes, numInternalNodes, leafToInternal, levelRange, childOffsets, parents, i);
    }
}

/** 
 * @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @param[in]  prefixes          octree node prefixes in Warren-Salmon format
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  leafToInternal    translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[in]  levelRange        indices of the first node at each level
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
void linkOctreeCPU(std::vector<uint64_t>& prefixes, int numInternalNodes, std::vector<int>& leafToInternal, std::vector<int>& levelRange, std::vector<int>& childOffsets, std::vector<int>& parents, int idx){
    int idxA = leafToInternal[idx]; // 0 - numInternal: unsorted internal nodes
    uint64_t prefix = prefixes[idxA];
    uint64_t nodeKey = decodePlaceholderBit(prefix);
    uint64_t prefixLength = decodePrefixLength(prefix);

    uint64_t level = prefixLength / 3;
    if(level >= 21){
        std::bitset<64> b(prefix);
        std::cout << "prefix: " << b << std::endl;
        printf("wrong prefixLength: %lu\n", prefixLength);
        printf("wrong level: %lu\n", level);
    }
    assert(level < 21);
    uint64_t childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

    int leafSearchStart = levelRange[level + 1];
    int leafSearchEnd = levelRange[level + 2];
    int childIdx = cal::lower_bound(prefixes.data() + leafSearchStart, prefixes.data() + leafSearchEnd, childPrefix) - prefixes.data();

    if(childIdx != leafSearchEnd && childPrefix == prefixes[childIdx]){
        childOffsets[idxA] = childIdx;
        // We only store the parent once for every group of 8 siblings.
        // This works as long as each node always has 8 siblings.
        // Subtract one because the root has no siblings.
        parents[(childIdx - 1) / 8] = idxA;
    }
}

void calculateNodeCentersAndSizesCPU(std::vector<uint64_t>& prefixes, std::vector<Vec3f>& centers, std::vector<Vec3f>& sizes, Box& box){
    #pragma omp parallel for
    for(int i = 0; i < prefixes.size(); i++){
        calculateNodeCentersAndSizesCPU(prefixes, centers, sizes, box, i);
    }
}

/**
 * @brief Calculate each nodes's center position and sizes (node's length)
 * 
 * @param prefixes sfc keys combined with placeholder bits
 * @param centers to store center positions
 * @param sizes to store node sizes
 * @param box the big box that stores the whole space range information
 * @param idx index
 */
void calculateNodeCentersAndSizesCPU(std::vector<uint64_t>& prefixes, std::vector<Vec3f>& centers, std::vector<Vec3f>& sizes, Box& box, int idx){
    uint64_t prefix = prefixes[idx];
    uint64_t start_key = decodePlaceholderBit(prefix);
    unsigned level = decodePrefixLength(prefix) / 3;

    constexpr int maxCoord = 1u << 21;
    unsigned cubeLength = (1u << (21 - level));
    Vec3<int> ivec = decodeMorton(start_key);

    IBox nodeBox(ivec.x, ivec.x + cubeLength, ivec.y, ivec.y + cubeLength, ivec.z, ivec.z + cubeLength);
    cal::tie(centers[idx], sizes[idx]) = centerAndSizeCPU(nodeBox, box);
    
}

void findNeighborsCPU(std::vector<Vec3f> particles, std::vector<float>& radiuses, OctreeNs& octreeNs, Box& box, int ngmax, std::vector<int>& neighbors, std::vector<int>& numNeighbors){
    #pragma omp parallel for
    for(int i = 0; i < particles.size(); i++){
        numNeighbors[i] = findNeighborsCPU(i, particles, radiuses, octreeNs, box, ngmax, neighbors.data() + i * ngmax);
    }
}

int findNeighborsCPU(int idx, std::vector<Vec3f> particles, std::vector<float>& radiuses, OctreeNs& octreeNs, Box& box, int ngmax, int* neighbors){
    float x = particles[idx].x;
    float y = particles[idx].y;
    float z = particles[idx].z;
    Vec3f particle = particles[idx];
    float h = radiuses[idx];
    int numNeighbors = 0;

    float radiusSquare = 4.0 * h * h;

    auto overlaps = [particle, radiusSquare, centers = octreeNs.centers, sizes = octreeNs.sizes, &box](int idx){
        auto nodeCenter = centers[idx];
        auto nodeSize = sizes[idx];
        Vec3f min_distance = cal::minDistanceCPU(particle, nodeCenter, nodeSize);
        float squared_distance = min_distance.x * min_distance.x + min_distance.y * min_distance.y + min_distance.z * min_distance.z;
        return squared_distance < radiusSquare;
    };

    auto searchBox = [idx, particle, radiusSquare, &octreeNs, particles, ngmax, neighbors, &numNeighbors, &box](int i){
        int leafIdx    = octreeNs.internalToLeaf[i];
        int firstParticle = octreeNs.layout[leafIdx];
        int lastParticle  = octreeNs.layout[leafIdx + 1];

        for (int j = firstParticle; j < lastParticle; ++j)
        {
            if (j == idx) { continue; }
            Vec3f diffs = particle - particles[j];
            float squared_distance = diffs.x * diffs.x + diffs.y * diffs.y + diffs.z * diffs.z;
            if(squared_distance < radiusSquare){
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }



    };        

    singleTraversal(octreeNs.childOffsets, overlaps, searchBox);
    return numNeighbors;
}

}
