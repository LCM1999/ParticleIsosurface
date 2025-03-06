#include <morton.h>

namespace cstoneOctree{

    void calMortonCodeCPU(std::vector<Vec3<float>>& coords, std::vector<uint64_t>& mortonCodes, const Box& box){
        unsigned cubeLength = (1u << 21);
        for (size_t i = 0; i < coords.size(); ++i)
        {
            mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, box.xmin(), box.ymin(), box.zmin(), cubeLength * box.ilx(), cubeLength * box.ily(),
                            cubeLength * box.ilz());
            
            // mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, -3.40118, -3.40118, -3.40118, cubeLength * 0.111727, cubeLength * 0.111727,
            //                   cubeLength * 0.111727);
            // mortonCodes[i] = sfc3D(coords[i].x, coords[i].y, coords[i].z, box);
        }

}
}