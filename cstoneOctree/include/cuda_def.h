#pragma once

#include <iostream>
#include <random>
// #include "cuda_runtime.h"


#ifdef __CUDACC__ 
    #define HOST_DEVICE __host__ __device__
    #define DEVICE __device__
    #define HOST __host__
#else
    #define HOST_DEVICE
    #define DEVICE
    #define HOST
#endif

struct DeviceConfig {
    int threads = 32;
    int blocks = 0;
    int smemBytes = 0;
    DeviceConfig(int totalThreads) {
        blocks = (totalThreads + threads - 1) / threads;
        blocks = (blocks == 0 ? 1 : blocks); // TODO: do not launch kernel when blocks == 0
    }
    DeviceConfig(int totalThreads, int threadsPerBlock) {
        threads = threadsPerBlock;
        blocks = (totalThreads + threads - 1) / threads;
        blocks = (blocks == 0 ? 1 : blocks); // TODO: do not launch kernel when blocks == 0
    }
    DeviceConfig(int totalThreads, int threadsPerBlock, int smem) {
        threads = threadsPerBlock;
        blocks = (totalThreads + threads - 1) / threads;
        blocks = (blocks == 0 ? 1 : blocks); // TODO: do not launch kernel when blocks == 0
        smemBytes = smem;
    }
};


struct DeviceBlock {
    int blockIdx;
    int threadIdx;
    int blockSize;
    int smemBytes;
    volatile void* smem = nullptr;

    HOST_DEVICE DeviceBlock(int _blockIdx, int _threadIdx, int _blockSize)
        : blockIdx(_blockIdx), threadIdx(_threadIdx), blockSize(_blockSize) {}
    HOST_DEVICE int globalIdx() const {
        return blockIdx * blockSize + threadIdx;
    }
    
};