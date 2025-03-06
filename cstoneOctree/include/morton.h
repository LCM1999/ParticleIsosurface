#pragma once

#include <cuda_def.h>
#include <box.h>
#include <coord_struct.h>
#include <calculator.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace cstoneOctree{

//! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
// HOST_DEVICE constexpr uint64_t expandBits(uint64_t v);
HOST_DEVICE constexpr uint64_t expandBits(uint64_t v){
    uint64_t x = v & 0x1fffffu; // discard bits higher 21
    x          = (x | x << 32u) & 0x001f00000000fffflu;
    x          = (x | x << 16u) & 0x001f0000ff0000fflu;
    x          = (x | x << 8u) & 0x100f00f00f00f00flu;
    x          = (x | x << 4u) & 0x10c30c30c30c30c3lu;
    x          = (x | x << 2u) & 0x1249249249249249lu;
    return x;
}

// //! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
// HOST_DEVICE constexpr uint64_t expandBits(int v);
HOST_DEVICE constexpr uint64_t expandBits(int v){
    uint64_t x = v & 0x1fffffu; // discard bits higher 21
    x          = (x | x << 32u) & 0x001f00000000fffflu;
    x          = (x | x << 16u) & 0x001f0000ff0000fflu;
    x          = (x | x << 8u) & 0x100f00f00f00f00flu;
    x          = (x | x << 4u) & 0x10c30c30c30c30c3lu;
    x          = (x | x << 2u) & 0x1249249249249249lu;
    return x;
}


/*! @brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
HOST_DEVICE constexpr uint64_t compactBits(uint64_t v){
    v &= 0x1249249249249249lu;
    v = (v ^ (v >> 2u)) & 0x10c30c30c30c30c3lu;
    v = (v ^ (v >> 4u)) & 0x100f00f00f00f00flu;
    v = (v ^ (v >> 8u)) & 0x001f0000ff0000fflu;
    v = (v ^ (v >> 16u)) & 0x001f00000000fffflu;
    v = (v ^ (v >> 32u)) & 0x00000000001ffffflu;
    return v;
}
// HOST_DEVICE constexpr uint64_t compactBits(uint64_t v);

HOST_DEVICE Vec3<int> decodeMorton(uint64_t code);

// template<class T>
HOST_DEVICE uint64_t sfc3D(float x, float y, float z, float xmin, float ymin, float zmin, float mx, float my, float mz);

// template<class T>
HOST_DEVICE uint64_t sfc3D(float x, float y, float z, const Box& box);


void calMortonCodeCPU(std::vector<Vec3<float>>& coords, std::vector<uint64_t>& mortonCodes, const Box& box);

__global__ void calMortonCodeGPUKenrel(Vec3f* coordsDevice, uint64_t* mortonCodesDevice, Box* box, int numParticles);

}