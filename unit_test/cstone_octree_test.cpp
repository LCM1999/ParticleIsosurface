#include <iostream>
#include "gtest/gtest.h"
// #include "mesh.h"

// #include <NeighborSearching.h>
// #include <NeighborSearching.cuh>
// #include <utils.h>

#include <box.h>
#include <morton.h>
#include <coord_struct.h>
#include <calculator.h>
#include <utils_helper.h>
#include <octree_func.h>
#include <bitset>
#include <algorithm>

using namespace cstoneOctree;


TEST(CSTONE, check_fill_data_cpu_correctness){
	int NODE_TO_TREEARRAY_PLUS_ONE = 1; // just for notification
    std::vector<uint64_t> tree(1 + NODE_TO_TREEARRAY_PLUS_ONE);
	cal::fill_data_cpu(tree.data(), 1, 0);
	cal::fill_data_cpu(tree.data() + 1, 1, uint64_t(1) << 63);
	std::vector<uint64_t> tree_expected = {0, 9223372036854775808};
	EXPECT_EQ(tree, tree_expected);
}

TEST(CSTONE, test_morton_calculation){
	// generateGaussianParticles("/home/letian/Downloads/isosurface_test/sample_coords_data/coords.txt", 10000, 1.0, 1.0);
	// std::cout << "test box usage" << std::endl;
	// cstoneOctree::Box box(-1.0, 1.0, -1.0, 2.0, -1.0, 3.0);
	// EXPECT_EQ(2.0, box.lx());
	
	// std::cout << box.maxExtent() << std::endl;
	// std::cout << box.minExtent() << std::endl;
	EXPECT_EQ(0b1000, encodePlaceholderBit(0lu, 3));
	EXPECT_EQ(01635, encodePlaceholderBit(0635000000000000000000ul, 9));

	EXPECT_EQ(0, decodePrefixLength(1ul));
    EXPECT_EQ(3, decodePrefixLength(0b1000ul));
    EXPECT_EQ(3, decodePrefixLength(0b1010ul));
    EXPECT_EQ(9, decodePrefixLength(01635ul));

	EXPECT_EQ(0, decodePlaceholderBit(1ul));
    EXPECT_EQ(0, decodePlaceholderBit(0b1000ul));
    // EXPECT_EQ(pad(0b010ul, 3), decodePlaceholderBit(0b1010ul));
    EXPECT_EQ(0635000000000000000000ul, decodePlaceholderBit(01635ul));
}

TEST(CSTONE, test_gen_octree){

	#ifdef __CUDACC__
		std::cout << "CUDA is available" << std::endl;	
	#endif
	// using namespace cstoneOctree;

	// ========================1 load points start======================
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::vector<float> r;
	readCoordinatesFromFile("/home/letian/Downloads/isosurface_test/sample_coords_data/dam.csv",r, x, y, z);
	float max_num = -100.0;
    float min_num = 100.0;
	Vec3<float> t = Vec3<float>(1.0, 2.0, 3.0);
	size_t coord_size = x.size();
	max_num = cal::max(*std::max_element(x.begin(), x.end()), cal::max(*std::max_element(y.begin(), y.end()), *std::max_element(z.begin(), z.end())));
	min_num = cal::min(*std::min_element(x.begin(), x.end()), cal::min(*std::min_element(y.begin(), y.end()), *std::min_element(z.begin(), z.end())));

	std::cout << "max_num: " << max_num << std::endl;
    std::cout << "min_num: " << min_num << std::endl;
    // exit(0);

    std::vector<Vec3f> posHost;

    for (int i = 0; i < coord_size; ++i){
        Vec3f new_pos;
        new_pos.x = x[i];
        new_pos.y = y[i];
        new_pos.z = z[i];
        posHost.push_back(new_pos);
    }

	std::vector<uint64_t> mortonCodes(coord_size);
	Box box(min_num, max_num);
	std::cout << "box min: " << box.xmin() << " " << box.ymin() << " " << box.zmin() << std::endl;
	std::cout << "box max: " << box.xmax() << " " << box.ymax() << " " << box.zmax() << std::endl;
	std::cout << "box inverse length: " << box.ilx() << " " << box.ily() << " " << box.ilz() << std::endl;
	unsigned cubeLength = (1u << 21);
	std::cout << "box cubeLength: " << cubeLength << std::endl;
	double out = cubeLength * box.ilx();
	std::cout << "box cubeLength * box.ilx(): " << out << std::endl;
	calMortonCodeCPU(posHost, mortonCodes, box);


	// 排序
	std::vector<size_t> indices(posHost.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return mortonCodes[i] < mortonCodes[j]; });
	std::vector<Vec3f> posHostSorted(posHost.size());
	std::vector<float> rSorted(r.size());
	std::vector<uint64_t> mortonCodesSorted(mortonCodes.size());
	for (size_t i = 0; i < indices.size(); ++i) {
        posHostSorted[i] = posHost[indices[i]];
        rSorted[i] = r[indices[i]];
		mortonCodesSorted[i] = mortonCodes[indices[i]];
    }
	posHost = posHostSorted;
	r = rSorted;
	mortonCodes = mortonCodesSorted;

	// cal::sort_by_key_cpu(mortonCodes.data(), mortonCodes.data() + mortonCodes.size(), posHost.data());
	// writeMortonCodesToFile("/home/letian/Downloads/isosurface_test/sample_coords_data/morton_codes_sorted.txt", mortonCodes);
	// writeCoordinatesToFile("/home/letian/Downloads/isosurface_test/sample_coords_data/coords_sorted.txt", posHost);
	// ========================1 load points end======================

	// ========================2 octree karray(leaf) construction loop start======================
	int NODE_TO_TREEARRAY_PLUS_ONE = 1; // just for notification
    std::vector<uint64_t> tree(1 + NODE_TO_TREEARRAY_PLUS_ONE); // tree array to store node (leaf node for now) morton code 
	cal::fill_data_cpu(tree.data(), 1, 0);
	cal::fill_data_cpu(tree.data() + 1, 1, uint64_t(1) << 63);

	int bucketSize = 32;
    std::vector<uint64_t> counts(1); // store how many particles in each node
	cal::fill_data_cpu(counts.data(), 1, coord_size);
    int maxCount = std::numeric_limits<int>::max();

	int count = 0;
	while(1){
		// init operation vector
		std::vector<uint64_t> nodeOps(tree.size()); // Store the split decision for each node (dynamically during makeSplitDecision function). 
													// Start from (one root node + 1) size for scan.
        
		cal::fill_data_cpu(nodeOps.data(), nodeOps.size(), 0);		
		makeSplitsDecisionsCPU(tree, counts, bucketSize, nodeOps);
		
		std::cout << "tree size in loop " << count << ": " << tree.size() << std::endl;

		uint64_t allOpsSum = std::accumulate(nodeOps.begin(), nodeOps.end(), 0);

		// exclusive_csan ops to get new octree indices
		std::vector<uint64_t> nodeOpsLayout(nodeOps.size());
		std::exclusive_scan(nodeOps.begin(), nodeOps.end(), nodeOpsLayout.begin(), 0);
		uint64_t newTreeNodesNum;	
		newTreeNodesNum = nodeOpsLayout[tree.size() - 1];
		std::vector<uint64_t> newTree(newTreeNodesNum + NODE_TO_TREEARRAY_PLUS_ONE);  // updated tree array

		updateTreeArrayCPU(nodeOpsLayout, tree, newTree);

		std::copy_n(tree.data() + tree.size() - 1, 1, newTree.data() + newTree.size() - 1);
		std::swap(newTree, tree);

		// update node counts
		counts.resize(tree.size() - 1);
		std::vector<uint64_t> coverNodes(2);	
		findCoverNodesCPU(tree, mortonCodes, coverNodes);	

		// set the out of bound nodes (absolutely does not contain coordinates) to minCount(0).
		cal::fill_data_cpu(counts.data(), coverNodes[0], 0);
		cal::fill_data_cpu(counts.data() + coverNodes[1], tree.size() - coverNodes[1] - 1, 0);

		updateNodeCountsCPU(tree, mortonCodes, counts, coverNodes);

		count++;
		if(allOpsSum == nodeOps.size() - 1) break;
	}
	// ========================2 octree karray(leaf) construction loop end======================

	// ========================3 octree internal nodes construction start======================
	// construct octree internal nodes, which is equal to buildOctreeCpu() in cornerstone octree source code
	int numLeafNodes = tree.size() - NODE_TO_TREEARRAY_PLUS_ONE; // tree size minus 1 because the range end variable is added before.
    int numInternalNodes = (numLeafNodes - 1) / 7; // number of nodes which is not leaf nodes
    int numNodes         = numLeafNodes + numInternalNodes; // total number of nodes (leaf nodes + internal nodes)
	std::vector<uint64_t> prefixes;  // used to delete the righthand side 0 bits for each morton code
    std::vector<TreeNodeIndex> internalToLeaf;
    std::vector<TreeNodeIndex> leafToInternal;
    std::vector<TreeNodeIndex> childOffsets;
	prefixes.resize(numNodes);
	internalToLeaf.resize(numNodes);
    leafToInternal.resize(numNodes);
    childOffsets.resize(numNodes + 1);
    std::vector<TreeNodeIndex> parents;
    std::vector<int> levelRange;
    parents.resize(std::max(1, (numNodes - 1) / 8));
    levelRange.resize(21 + 2);	

	
	// combine internal and leaf tree parts into a single array with the nodeKey prefixes
	createUnsortedLayoutCPU(tree, numInternalNodes, numLeafNodes, prefixes, internalToLeaf);	
	

	cal::sort_by_key_cpu(prefixes.data(), prefixes.data() + prefixes.size(), internalToLeaf.data());

	// for(int i = 0; i < 21; i++){
	// 	std::bitset<64> b(prefixes[i]);
	// 	std::cout << "sorted prefixes in leaf node[" << i << "]: " << b << std::endl;
	// }

	invertOrderCPU(internalToLeaf, leafToInternal, numNodes, numInternalNodes);
	// Calculate node range for each tree level
	getLevelRangeCPU(prefixes, numNodes, levelRange);
	cal::fill_data_cpu(childOffsets.data(), numNodes + 1, 0);

	linkOctreeCPU(prefixes,
				numInternalNodes,
				leafToInternal,
				levelRange,
				childOffsets,
				parents);

	// ========================3 octree internal nodes construction end======================

	// calculate centers and sizes for each node
	std::cout << "end" << std::endl;	
	std::vector<Vec3f> centers(numNodes);
	std::vector<Vec3f> sizes(numNodes);
	calculateNodeCentersAndSizesCPU(prefixes, centers, sizes, box);

	// ========================4 neighbor search start======================
	
    std::vector<int> layout(numLeafNodes + 1); // index of first particle for each leaf node
    std::exclusive_scan(counts.begin(), counts.end(), layout.begin(), 0);
	// std::vector<float> radiuses(coord_size, 0.5f);
	int ngmax = 27; 
	OctreeNs octreeNs(prefixes.data(), childOffsets.data(), internalToLeaf.data(), levelRange.data(), layout.data(), centers.data(), sizes.data());
	std::vector<int> neighbors(coord_size * ngmax);
	std::vector<int> numNeighbors(coord_size);
	findNeighborsCPU(posHost, r, octreeNs, box, ngmax, neighbors, numNeighbors);


	// for(int i = 0; i < numNeighbors.size(); i++){
	// 	std::cout << "numNeighbors[" << i << "]: " << numNeighbors[i] << std::endl;
	// }
	// for(int i = 0; i < neighbors.size(); i++){
	// 	std::cout << "neighbors[" << i << "]: " << neighbors[i] << std::endl;
	// }
	
}
