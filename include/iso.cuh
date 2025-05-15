#pragma once
#ifndef ISO_CUH
#define ISO_CUH

#include <vector>
#include <coord_struct.h>
#include <global.h>

#include <box.h>

struct HashGridGPU;
struct EvaluatorGPU;

namespace iso {
    void generateIso(
        const std::vector<uint64_t>& mortons,
        const std::vector<cstoneOctree::Vec3i>& lowers, 
        const std::vector<unsigned>& levels, 
        const std::vector<float>& scalars,
        const float isoValue,
        Mesh* mesh
    );

    void generateIsoDirectGPU(
        thrust::device_vector<uint64_t>& d_mortons,
        thrust::device_vector<cstoneOctree::Vec3f>& d_centers,
        thrust::device_vector<cstoneOctree::Vec3i>& d_lowers, 
        thrust::device_vector<unsigned>& d_levels, 
        thrust::device_vector<float>& d_scalars,
        int d_iso_tree_size,
        HashGridGPU** d_searchers, int d_searchers_size, EvaluatorGPU* d_evaluator, 
        const float isoValue, const float errorBound, 
        cstoneOctree::Box* d_box, 
        Mesh* mesh
    );
}

#endif