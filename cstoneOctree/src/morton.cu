#include <morton.h>
#include <inttypes.h>

namespace cstoneOctree{

// //! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
// HOST_DEVICE constexpr uint64_t expandBits(uint64_t v){
//     uint64_t x = v & 0x1fffffu; // discard bits higher 21
//     x          = (x | x << 32u) & 0x001f00000000fffflu;
//     x          = (x | x << 16u) & 0x001f0000ff0000fflu;
//     x          = (x | x << 8u) & 0x100f00f00f00f00flu;
//     x          = (x | x << 4u) & 0x10c30c30c30c30c3lu;
//     x          = (x | x << 2u) & 0x1249249249249249lu;
//     return x;
// }

// //! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
// HOST_DEVICE constexpr uint64_t expandBits(int v){
//     uint64_t x = v & 0x1fffffu; // discard bits higher 21
//     x          = (x | x << 32u) & 0x001f00000000fffflu;
//     x          = (x | x << 16u) & 0x001f0000ff0000fflu;
//     x          = (x | x << 8u) & 0x100f00f00f00f00flu;
//     x          = (x | x << 4u) & 0x10c30c30c30c30c3lu;
//     x          = (x | x << 2u) & 0x1249249249249249lu;
//     return x;
// }


/*! @brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
// HOST_DEVICE constexpr uint64_t compactBits(uint64_t v){
//     v &= 0x1249249249249249lu;
//     v = (v ^ (v >> 2u)) & 0x10c30c30c30c30c3lu;
//     v = (v ^ (v >> 4u)) & 0x100f00f00f00f00flu;
//     v = (v ^ (v >> 8u)) & 0x001f0000ff0000fflu;
//     v = (v ^ (v >> 16u)) & 0x001f00000000fffflu;
//     v = (v ^ (v >> 32u)) & 0x00000000001ffffflu;
//     return v;
// }

HOST_DEVICE Vec3<int> decodeMorton(uint64_t code){
    return Vec3<int>(compactBits(code >> 2), compactBits(code >> 1), compactBits(code));
}

// template<class T>
HOST_DEVICE uint64_t sfc3D(float x, float y, float z, float xmin, float ymin, float zmin, float mx, float my, float mz){
    constexpr int mcoord = (1u << 21) - 1;


    // int ix = floorf(x * mx - xmin * mx);
    // int iy = floorf(y * my - ymin * my);
    // int iz = floorf(z * mz - zmin * mz);

    int ix = floorf(x * mx) - xmin * mx;
    int iy = floorf(y * my) - ymin * my;
    int iz = floorf(z * mz) - zmin * mz;

    ix = fminf(ix, mcoord);
    iy = fminf(iy, mcoord);
    iz = fminf(iz, mcoord);
    // if(iz < 0){
    //     printf("z * mz: %f, zmin * mz: %f\n", z * mz, zmin * mz);
    // } 
    // if(iz > (1u << 21)){
    //     printf("z * mz: %f, zmin * mz: %f\n", z * mz, zmin * mz);
    // }
    ix = ix < 0 ? 0 : ix;
    iy = iy < 0 ? 0 : iy;
    iz = iz < 0 ? 0 : iz; 
    assert(ix >= 0);
    assert(iy >= 0);
    assert(iz >= 0);

    assert(ix < (1u << 21));
    assert(iy < (1u << 21));
    assert(iz < (1u << 21)); 


    uint64_t xx = expandBits(ix);
    uint64_t yy = expandBits(iy);
    uint64_t zz = expandBits(iz);

    // uint64_t result = xx * 4 + yy * 2 + zz;
    // printf("result: %llu\n", (unsigned long long)result);
    return xx * 4 + yy * 2 + zz;
    
}

// template<class T>
HOST_DEVICE uint64_t sfc3D(float x, float y, float z, const Box& box){
    unsigned cubeLength = (1u << 21);

    return sfc3D(x, y, z, box.xmin(), box.ymin(), box.zmin(), cubeLength * box.ilx(), cubeLength * box.ily(),
                          cubeLength * box.ilz());
}



// void calMortonCodeCPU(std::vector<Vec3<float>>& coords, std::vector<uint64_t>& mortonCodes, const Box& box){
//     unsigned cubeLength = (1u << 21);
//     for (size_t i = 0; i < coords.size(); ++i)
//     {
//         mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, box.xmin(), box.ymin(), box.zmin(), cubeLength * box.ilx(), cubeLength * box.ily(),
//                           cubeLength * box.ilz());
//         // mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, -3.40118, -3.40118, -3.40118, cubeLength * 0.111727, cubeLength * 0.111727,
//         //                   cubeLength * 0.111727);
//         // mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, box);
//     }

// }

__global__ void calMortonCodeGPUKenrel(Vec3f* coordsDevice, uint64_t* mortonCodesDevice, Box* boxDevice, int numParticles){
    // DeviceBlock block(blockIdx.x, threadIdx.x, blockDim.x);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numParticles){
        return;
    }
    
    unsigned cubeLength = (1u << 21);
    mortonCodesDevice[idx] = sfc3D(coordsDevice[idx].x, coordsDevice[idx].y, coordsDevice[idx].z, boxDevice->xmin(), boxDevice->ymin(), boxDevice->zmin(), 
                        cubeLength * boxDevice->ilx(), cubeLength * boxDevice->ily(), cubeLength * boxDevice->ilz());

} 

}