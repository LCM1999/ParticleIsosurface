#pragma once
#ifndef ISO_CUH
#define ISO_CUH

#include <vector>
#include <coord_struct.h>
#include <global.h>

namespace iso {
    void generateIso(
        const std::vector<uint64_t>& mortons,
        const std::vector<cstoneOctree::Vec3i>& lowers, 
        const std::vector<unsigned>& levels, 
        const std::vector<float>& scalars,
        const float isoValue,
        Mesh* mesh
    );
}

#endif