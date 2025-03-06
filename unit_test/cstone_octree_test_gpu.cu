#include <iostream>
#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <box.h>
#include <morton.h>
#include <coord_struct.h>
#include <calculator.h>
#include <utils_helper.h>
#include <octree_func.h>
#include <bitset>

using namespace cstoneOctree;


TEST(CSTONE, test_box_usage){
	std::cout << "test box usage GPU" << std::endl;
	#ifdef __CUDACC__
		std::cout << "CUDA is enabled" << std::endl;
	#else
		std::cout << "CUDA is not enabled" << std::endl;
	#endif
	cstoneOctree::Box box(-1.0, 1.0, -1.0, 2.0, -1.0, 3.0);
	EXPECT_EQ(2.0, box.lx());
	
	// std::cout << box.maxExtent() << std::endl;
	// std::cout << box.minExtent() << std::endl;
}

TEST(CSTONE, test_sfc_correctness){
	Vec3f pos = Vec3f(0.271647, 0.755289, 2.027130);
	Box box(-3.40118, 5.5492);

	// sfc calculation in CPU
	uint64_t morton_code = sfc3D(pos.x, pos.y, pos.z, box);
	assert(morton_code == 2133948275538812772);
	std::cout << "(0.271647, 0.755289, 2.027130)'s morton code: " << morton_code << std::endl;

	// sfc calculation in GPU
	Vec3f* pos_device;
	cudaMalloc(&pos_device, sizeof(Vec3f));
	cudaMemcpy(pos_device, &pos, sizeof(Vec3f), cudaMemcpyHostToDevice);
	uint64_t morton_code_host = 0;
	uint64_t* morton_code_device;
	cudaMalloc(&morton_code_device, sizeof(uint64_t));
	cudaMemcpy(morton_code_device, &morton_code_host, sizeof(uint64_t), cudaMemcpyHostToDevice);
	Box* box_device;
	cudaMalloc(&box_device, sizeof(Box));
	cudaMemcpy(box_device, &box, sizeof(Box), cudaMemcpyHostToDevice);
	calMortonCodeGPUKenrel<<<1, 32>>>(pos_device, morton_code_device, box_device, 1);

	cudaMemcpy(&morton_code_host, morton_code_device, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	assert(morton_code_host == 2133948275538812772);
	std::cout << "morton code cal in GPU: " << morton_code_host << std::endl;	



	// std::vector<Vec3f> coords;
	// Vec3f pos2 = Vec3f(1.19218, 0.695718, -3.40118);
	// coords.push_back(pos);
	// coords.push_back(pos2);

	// Vec3f* coordsDevice;
	// cudaMalloc(&coordsDevice, coords.size() * sizeof(Vec3f));
	// cudaMemcpy(coordsDevice, coords.data(), coords.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);

	// std::vector<uint64_t> mortonCodes(coords.size(), 0);
	// // calMortonCodeCPU(coords, mortonCodes, box);
	// uint64_t* mortonCodesDevice;
	// cudaMalloc(&mortonCodesDevice, coords.size() * sizeof(uint64_t));
	// cudaMemcpy(mortonCodesDevice, mortonCodes.data(), coords.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
	// calMortonCodeGPUKenrel<<<1, 2>>>(coordsDevice, mortonCodesDevice, box_device, coords.size());
	// std::vector<uint64_t> mortonCodesHost(coords.size(), 0);
	// cudaMemcpy(mortonCodesHost.data(), mortonCodesDevice, coords.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// assert(mortonCodesHost[0] == 2133948275538812772);
	// assert(mortonCodesHost[1] == 4940540091531134992);



}

TEST(CSTONE, test_morton_usage){
	// GPU's idea is to calculate on GPU and then synchronize the data back to CPU for each function step.
	// In other words, the concept is that for each function step, the data on CPU and GPU should be the same.
	// The function with GPU suffix takes device data as input and then copy to host after the function finished.

	// ========================1 load points start======================
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::vector<float> r;
	readCoordinatesFromFile("/home/letian/Downloads/isosurface_test/sample_coords_data/dam.csv", r, x, y, z);
	float max_num = -100.0;
    float min_num = 100.0;

	size_t coord_size = x.size();
	max_num = cal::max(*std::max_element(x.begin(), x.end()), cal::max(*std::max_element(y.begin(), y.end()), *std::max_element(z.begin(), z.end())));
	min_num = cal::min(*std::min_element(x.begin(), x.end()), cal::min(*std::min_element(y.begin(), y.end()), *std::min_element(z.begin(), z.end())));

	std::cout << "max_num: " << max_num << std::endl;
    std::cout << "min_num: " << min_num << std::endl;
    // exit(0);

	thrust::host_vector<Vec3f> coordsHost;

    for (int i = 0; i < coord_size; ++i){
        Vec3f new_pos;
        new_pos.x = x[i];
        new_pos.y = y[i];
        new_pos.z = z[i];
        coordsHost.push_back(new_pos);
    }

	thrust::host_vector<uint64_t> mortonCodesHost(coord_size, 0);
	Box box(min_num, max_num);
	std::cout << "box min: " << box.xmin() << " " << box.ymin() << " " << box.zmin() << std::endl;
	std::cout << "box max: " << box.xmax() << " " << box.ymax() << " " << box.zmax() << std::endl;
	std::cout << "box inverse length: " << box.ilx() << " " << box.ily() << " " << box.ilz() << std::endl;
	unsigned cubeLength = (1u << 21);
	std::cout << "box cubeLength: " << cubeLength << std::endl;
	double out = cubeLength * box.ilx();
	std::cout << "box cubeLength * box.ilx(): " << out << std::endl;


	// ---  calculate morton code on GPU ---
	// initialize device data for coordinates, morton codes, and box
	int coordsSizeHost = coordsHost.size();
	DeviceConfig cudaConfig(coordsHost.size());

	thrust::device_vector<Vec3f> coordsDevice = coordsHost;
	thrust::device_vector<uint64_t> mortonCodesDevice = mortonCodesHost;
	Box* boxDevice;

	Vec3f* coordsDevicePtr = thrust::raw_pointer_cast(coordsDevice.data());
	uint64_t* mortonCodesDevicePtr = thrust::raw_pointer_cast(mortonCodesDevice.data());
	cudaMalloc(&boxDevice, sizeof(Box)); 
	cudaMemcpy(boxDevice, &box, sizeof(Box), cudaMemcpyHostToDevice);	
	

	// calculate morton code in GPU
	calMortonCodeGPUKenrel<<<cudaConfig.blocks, cudaConfig.threads>>>(coordsDevicePtr, mortonCodesDevicePtr, boxDevice, coordsSizeHost);
	cudaDeviceSynchronize();

	cudaMemcpy(mortonCodesHost.data(), mortonCodesDevicePtr, coord_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// 排序
	std::vector<size_t> indices(coordsHost.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}	
	thrust::copy(mortonCodesDevice.begin(), mortonCodesDevice.end(), mortonCodesHost.begin());
	std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return mortonCodesHost[i] < mortonCodesHost[j]; });

	thrust::host_vector<Vec3f> posHostSorted(coordsHost.size());
	thrust::host_vector<float> rSorted(r.size());
	thrust::host_vector<uint64_t> mortonCodesSorted(mortonCodesHost.size());
	for (size_t i = 0; i < indices.size(); ++i) {
        posHostSorted[i] = coordsHost[indices[i]];
        rSorted[i] = r[indices[i]];
		mortonCodesSorted[i] = mortonCodesHost[indices[i]];
    }
	coordsHost = posHostSorted;
	mortonCodesHost = mortonCodesSorted;

	// thrust::sort_by_key(thrust::device, mortonCodesDevice.begin(), mortonCodesDevice.end(), coordsDevice.begin());
	// thrust::copy(mortonCodesDevice.begin(), mortonCodesDevice.end(), mortonCodesHost.begin());
	// thrust::copy(coordsDevice.begin(), coordsDevice.end(), coordsHost.begin());


	// reassign the data and address after sort --> very important!!!
	coordsDevice = coordsHost;
	mortonCodesDevice = mortonCodesHost;
	coordsDevicePtr = thrust::raw_pointer_cast(coordsDevice.data());
	mortonCodesDevicePtr = thrust::raw_pointer_cast(mortonCodesDevice.data());


	// ========================1 load points end======================

	// ========================2 octree karray(leaf) construction loop start======================
	int NODE_TO_TREEARRAY_PLUS_ONE = 1; // just for notification
    thrust::host_vector<uint64_t> treeHost(1 + NODE_TO_TREEARRAY_PLUS_ONE);
	cal::fill_data_cpu(treeHost.data(), 1, 0);
	cal::fill_data_cpu(treeHost.data() + 1, 1, uint64_t(1) << 63);

	int bucketSize = 32;
	thrust::host_vector<uint64_t> countsHost(1);
	cal::fill_data_cpu(countsHost.data(), 1, coord_size);
    // int maxCount = std::numeric_limits<int>::max();

	thrust::device_vector<uint64_t> treeDevice = treeHost;
	thrust::device_vector<uint64_t> countsDevice = countsHost;
	uint64_t* treeDevicePtr = thrust::raw_pointer_cast(treeDevice.data());
	uint64_t* countsDevicePtr = thrust::raw_pointer_cast(countsDevice.data());


	// start the loop
	int count = 0;
	while(1){
		// ========== make split decisions ==========
		thrust::host_vector<uint64_t> nodeOpsHost(treeDevice.size()); // start from (one root node + 1) size for scan.	
		cal::fill_data_cpu(nodeOpsHost.data(), nodeOpsHost.size(), 0);
		thrust::device_vector<uint64_t> nodeOpsDevice = nodeOpsHost;
		uint64_t* nodeOpsDevicePtr = thrust::raw_pointer_cast(nodeOpsDevice.data());

		DeviceConfig treeSizeConfig(treeDevice.size() - 1);
		makeSplitsDecisionsGPUKernel<<<treeSizeConfig.blocks, treeSizeConfig.threads>>>(treeDevicePtr, countsDevicePtr, bucketSize, nodeOpsDevicePtr, treeDevice.size() - 1);
		cudaDeviceSynchronize();
		treeHost = treeDevice;
		std::cout << "tree size in loop " << count << ": " << treeHost.size() << std::endl;

		// sum up the nodeOpsDevice values
		uint64_t allOpsSum = thrust::reduce(thrust::device, nodeOpsDevice.begin(), nodeOpsDevice.end());

		// ========== rebalance the octree ==========
		// scan ops to get new tree indices
		thrust::device_vector<uint64_t> nodeOpsLayoutDevice(nodeOpsHost.size());
		thrust::exclusive_scan(thrust::device, nodeOpsDevice.begin(), nodeOpsDevice.end(), nodeOpsLayoutDevice.begin(), 0);
		uint64_t* nodeOpsLayoutDevicePtr = thrust::raw_pointer_cast(nodeOpsLayoutDevice.data());
		uint64_t newTreeNodesNum;
    	thrust::copy(nodeOpsLayoutDevice.end() - 1, nodeOpsLayoutDevice.end(), &newTreeNodesNum);

		thrust::device_vector<uint64_t> newTreeDevice(newTreeNodesNum + NODE_TO_TREEARRAY_PLUS_ONE);
		uint64_t* newTreeDevicePtr = thrust::raw_pointer_cast(newTreeDevice.data());
		updateTreeArrayGPUKernel<<<treeSizeConfig.blocks, treeSizeConfig.threads>>>(nodeOpsLayoutDevicePtr, treeDevicePtr, newTreeDevicePtr, treeDevice.size() - 1);
		cudaDeviceSynchronize();
		
		thrust::copy_n(thrust::device, treeDevice.end() - 1, 1, newTreeDevice.end() - 1);
		thrust::swap(treeDevice, newTreeDevice);

		// reassign the treeDevicePtr --> very important
		treeDevicePtr = thrust::raw_pointer_cast(treeDevice.data());

		
		// ========== update node counts ==========
		countsDevice.resize(treeDevice.size() - NODE_TO_TREEARRAY_PLUS_ONE);
		countsDevicePtr = thrust::raw_pointer_cast(countsDevice.data());

		thrust::device_vector<uint64_t> coverNodesDevice(2);
		uint64_t* coverNodesDevicePtr = thrust::raw_pointer_cast(coverNodesDevice.data());	
		uint64_t* mortonCodesDeviceStartPtr = mortonCodesDevicePtr;
		uint64_t* mortonCodesDeviceEndPtr = mortonCodesDevicePtr + mortonCodesDevice.size();
		findCoverNodesGPUKernel<<<1, 1>>>(treeDevicePtr, mortonCodesDeviceStartPtr, mortonCodesDeviceEndPtr, coverNodesDevicePtr, treeDevice.size() - NODE_TO_TREEARRAY_PLUS_ONE);	
		cudaDeviceSynchronize();

		thrust::host_vector<uint64_t> coverNodesHost = coverNodesDevice;
		thrust::fill_n(countsDevice.begin(), coverNodesHost[0], 0);
		thrust::fill_n(countsDevice.begin() + coverNodesHost[1], treeDevice.size() - coverNodesHost[1], 0);

		DeviceConfig coverNodesConfig(coverNodesHost[1] - coverNodesHost[0], 256);
		updateNodeCountsGPUKernel<<<coverNodesConfig.blocks, coverNodesConfig.threads>>>(treeDevicePtr + coverNodesHost[0], coverNodesHost[1] - coverNodesHost[0], mortonCodesDeviceStartPtr, mortonCodesDeviceEndPtr, countsDevicePtr + coverNodesHost[0]);	
		cudaDeviceSynchronize();

		count++;

		if(allOpsSum == nodeOpsHost.size() - 1) break;
	}
	// ========================2 octree karray(leaf) construction loop end======================


	// ========================3 octree internal nodes construction start======================
	// construct octree internal nodes, which is equal to buildOctreeCpu() in cornerstone octree source code	
	int numLeafNodes = treeDevice.size() - NODE_TO_TREEARRAY_PLUS_ONE; // tree size minus 1 because the range end variable is added before.
    int numInternalNodes = (numLeafNodes - 1) / 7;
    int numNodes         = numLeafNodes + numInternalNodes;
	thrust::device_vector<uint64_t> prefixesDevice;
    thrust::device_vector<TreeNodeIndex> internalToLeafDevice;
    thrust::device_vector<TreeNodeIndex> leafToInternalDevice;
    thrust::device_vector<TreeNodeIndex> childOffsetsDevice;
	prefixesDevice.resize(numNodes);
	internalToLeafDevice.resize(numNodes);
    leafToInternalDevice.resize(numNodes);
    childOffsetsDevice.resize(numNodes + 1);
    thrust::device_vector<TreeNodeIndex> parentsDevice;
    thrust::device_vector<int> levelRangeDevice;
    parentsDevice.resize(std::max(1, (numNodes - 1) / 8));
    levelRangeDevice.resize(21 + 2);	

	uint64_t* prefixesDevicePtr = thrust::raw_pointer_cast(prefixesDevice.data());
	TreeNodeIndex* internalToLeafDevicePtr = thrust::raw_pointer_cast(internalToLeafDevice.data());
	TreeNodeIndex* leafToInternalDevicePtr = thrust::raw_pointer_cast(leafToInternalDevice.data());
	TreeNodeIndex* childOffsetsDevicePtr = thrust::raw_pointer_cast(childOffsetsDevice.data());
	TreeNodeIndex* parentsDevicePtr = thrust::raw_pointer_cast(parentsDevice.data());
	int* levelRangeDevicePtr = thrust::raw_pointer_cast(levelRangeDevice.data());


	DeviceConfig leafNodesConfig(numLeafNodes);
	createUnsortedLayoutKernel<<<leafNodesConfig.blocks, leafNodesConfig.threads>>>(treeDevicePtr, numInternalNodes, numLeafNodes, prefixesDevicePtr, internalToLeafDevicePtr);
	cudaDeviceSynchronize();
	
	thrust::sort_by_key(prefixesDevice.begin(), prefixesDevice.end(), internalToLeafDevice.begin());

	DeviceConfig numNodesConfig(numNodes);
	invertOrderKernel<<<numNodesConfig.blocks, numNodesConfig.threads>>>(internalToLeafDevicePtr, leafToInternalDevicePtr, numNodes, numInternalNodes);
	cudaDeviceSynchronize();
	getLevelRangeKernel<<<1, 32>>>(prefixesDevicePtr, numNodes, levelRangeDevicePtr);
	cudaDeviceSynchronize();

	
	thrust::fill_n(childOffsetsDevice.data(), numNodes + 1, 0);

	DeviceConfig internalNodesConfig(numInternalNodes);
	linkOctreeKernel<<<internalNodesConfig.blocks, internalNodesConfig.threads>>>(prefixesDevicePtr,
				numInternalNodes,
				leafToInternalDevicePtr,
				levelRangeDevicePtr,
				childOffsetsDevicePtr,
				parentsDevicePtr);

	std::cout << "link octree end" << std::endl;	
	// ========================3 octree internal nodes construction end======================

	
	thrust::device_vector<Vec3f> centersDevice(numNodes);
	thrust::device_vector<Vec3f> sizesDevice(numNodes);
	Vec3f* centersDevicePtr = thrust::raw_pointer_cast(centersDevice.data());
	Vec3f* sizesDevicePtr = thrust::raw_pointer_cast(sizesDevice.data());
	DeviceConfig prefixesConfig(prefixesDevice.size());
	calculateNodeCentersAndSizesKernel<<<prefixesConfig.blocks, prefixesConfig.threads>>>(prefixesDevicePtr, prefixesDevice.size(), centersDevicePtr, sizesDevicePtr, boxDevice);
	cudaDeviceSynchronize();
	std::cout << "center calculation end !" << std::endl;	

	// ========================4 neighbor search start======================
	thrust::device_vector<int> layoutDevice(numLeafNodes + 1); // index of first particle for each leaf node
	thrust::exclusive_scan(countsDevice.begin(), countsDevice.end(), layoutDevice.begin(), 0);
	int ngmax = 27;
	int* layoutDevicePtr = thrust::raw_pointer_cast(layoutDevice.data());
	float* radiusesDevicePtr = thrust::raw_pointer_cast(r.data());

	OctreeNs octreeNs(prefixesDevicePtr, childOffsetsDevicePtr, internalToLeafDevicePtr, levelRangeDevicePtr, layoutDevicePtr, centersDevicePtr, sizesDevicePtr);
	
	thrust::device_vector<int> neighborsDevice(coord_size * ngmax);
	thrust::device_vector<int> numNeighborsDevice(coord_size);
	int* neighborsDevicePtr = thrust::raw_pointer_cast(neighborsDevice.data());
	int* numNeighborsDevicePtr = thrust::raw_pointer_cast(numNeighborsDevice.data());

	DeviceConfig particlesConfig(coord_size);
	findNeighborsKernel<<<particlesConfig.blocks, particlesConfig.threads>>>(coordsDevicePtr, 
						coord_size,
						radiusesDevicePtr,
						&octreeNs,
						boxDevice,
						ngmax,
						neighborsDevicePtr,
						numNeighborsDevicePtr);	
	cudaDeviceSynchronize();

	thrust::host_vector<int> numNeighborsHost = numNeighborsDevice;
	thrust::host_vector<int> neighborsHost = neighborsDevice;
	thrust::host_vector<Vec3f> centersHost = centersDevice;
	thrust::host_vector<Vec3f> sizesHost = sizesDevice;
	countsHost = countsDevice;
	treeHost = treeDevice;

	// for(int i = 0; i < numNeighborsHost.size(); i++){
	// 	std::cout << "numNeighbors[" << i << "]: " << numNeighborsHost[i] << std::endl;
	// }
	// for(int i = 0; i < neighborsHost.size(); i++){
	// 	std::cout << "neighbors[" << i << "]: " << neighborsHost[i] << std::endl;
	// }	

	// --- free the device memory --- 

}

// TEST(CSTONE, test_morton_usage_2){
// 	// GPU's idea is to calculate on GPU and then synchronize the data back to CPU for each function step.
// 	// In other words, the concept is that for each function step, the data on CPU and GPU should be the same.
// 	// The function with GPU suffix takes device data as input and then copy to host after the function finished.
// 	std::vector<float> x;
// 	std::vector<float> y;
// 	std::vector<float> z;
// 	readCoordinatesFromFile("/home/letian/Downloads/isosurface_test/sample_coords_data/coords.txt",x,y,z);
// 	float max_num = -100.0;
//     float min_num = 100.0;
// 	Vec3<float> t = Vec3<float>(1.0, 2.0, 3.0);
// 	std::cout << t.x << " " << t.y << " " << t.z << std::endl;
// 	size_t coord_size = x.size();
// 	max_num = cal::max(*std::max_element(x.begin(), x.end()), cal::max(*std::max_element(y.begin(), y.end()), *std::max_element(z.begin(), z.end())));
// 	min_num = cal::min(*std::min_element(x.begin(), x.end()), cal::min(*std::min_element(y.begin(), y.end()), *std::min_element(z.begin(), z.end())));

// 	std::cout << "max_num: " << max_num << std::endl;
//     std::cout << "min_num: " << min_num << std::endl;
//     // exit(0);

//     std::vector<Vec3f> coordsHost;

//     for (int i = 0; i < coord_size; ++i){
//         Vec3f new_pos;
//         new_pos.x = x[i];
//         new_pos.y = y[i];
//         new_pos.z = z[i];
//         coordsHost.push_back(new_pos);
//     }

// 	std::vector<uint64_t> mortonCodes(coord_size, 0);
// 	Box box(min_num, max_num);
// 	std::cout << "box min: " << box.xmin() << " " << box.ymin() << " " << box.zmin() << std::endl;
// 	std::cout << "box max: " << box.xmax() << " " << box.ymax() << " " << box.zmax() << std::endl;
// 	std::cout << "box inverse length: " << box.ilx() << " " << box.ily() << " " << box.ilz() << std::endl;
// 	printf("box's information: %f %f %f %f %f %f %f %f %f\n", box.xmin(), box.ymin(), box.zmin(), box.xmax(), box.ymax(), box.zmax(), box.ilx(), box.ily(), box.ilz());


// 	// --- calculate morton code on GPU ---
// 	// initialize device data for coordinates, morton codes, and box
// 	int coordsSizeHost = coordsHost.size();
// 	Vec3f* coordsDevice;
// 	cudaMalloc((void**)&coordsDevice, coordsHost.size() * sizeof(Vec3f));
// 	cudaMemcpy(coordsDevice, coordsHost.data(), coordsHost.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
// 	uint64_t* mortonCodesDevice;
// 	cudaMalloc((void**)&mortonCodesDevice, coordsHost.size() * sizeof(uint64_t));
// 	cudaMemcpy(mortonCodesDevice, mortonCodes.data(), coordsHost.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
// 	Box* boxDevice;
// 	cudaMalloc((void**)&boxDevice, sizeof(Box));
// 	cudaMemcpy(boxDevice, &box, sizeof(Box), cudaMemcpyHostToDevice);
// 	DeviceConfig cudaConfig(coordsHost.size());


// 	calMortonCodeGPUKenrel<<<cudaConfig.blocks, cudaConfig.threads>>>(coordsDevice, mortonCodesDevice, boxDevice, coordsSizeHost);
// 	cudaMemcpy(mortonCodes.data(), mortonCodesDevice, coord_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
// 	std::cout << "morton code [0]: " << mortonCodes[0] << std::endl;
// 	std::cout << "morton code [1]: " << mortonCodes[1] << std::endl;


// 	// --- free the device memory --- 
// 	cudaFree(coordsDevice);
// }