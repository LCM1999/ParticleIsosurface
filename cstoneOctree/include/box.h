#pragma once

#include <cassert>
#include <cmath>
#include <cuda_def.h>
#include <coord_struct.h>
#include <calculator.h>
#include <cuda_runtime.h>
#include <thrust/tuple.h>

namespace cstoneOctree {

/**
 * @brief Store the domain's information
 */
class Box {
public:
    // Constructors
    HOST_DEVICE Box(float xyzMin, float xyzMax) : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax},
      lengths_{xyzMax - xyzMin, xyzMax - xyzMin, xyzMax - xyzMin},
      inverseLengths_{1.0f / (xyzMax - xyzMin), 1.0f / (xyzMax - xyzMin), 
                     1.0f / (xyzMax - xyzMin)} {}
    
    HOST_DEVICE  Box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
                    : limits{xmin, xmax, ymin, ymax, zmin, zmax}, 
                    lengths_{xmax - xmin, ymax - ymin, zmax - zmin}, 
                    inverseLengths_{1.0f / (xmax - xmin), 
                      1.0f / (ymax - ymin),
                      1.0f / (zmax - zmin)} {}

    // Getter methods
    HOST_DEVICE  float xmin() const { return limits[0]; }
    HOST_DEVICE  float xmax() const { return limits[1]; }
    HOST_DEVICE  float ymin() const { return limits[2]; }
    HOST_DEVICE  float ymax() const { return limits[3]; }
    HOST_DEVICE  float zmin() const { return limits[4]; }
    HOST_DEVICE  float zmax() const { return limits[5]; }

    // Edge lengths
    HOST_DEVICE  float lx() const { return lengths_[0]; }
    HOST_DEVICE  float ly() const { return lengths_[1]; }
    HOST_DEVICE  float lz() const { return lengths_[2]; }

    // Inverse edge lengths
    HOST_DEVICE  float ilx() const { return inverseLengths_[0]; }
    HOST_DEVICE  float ily() const { return inverseLengths_[1]; }
    HOST_DEVICE  float ilz() const { return inverseLengths_[2]; }

    // Extent methods
    HOST_DEVICE float minExtent() const {
        return cal::min(cal::min(lengths_[0], lengths_[1]), lengths_[2]);
    }
    HOST_DEVICE float maxExtent() const {
        return cal::max(cal::max(lengths_[0], lengths_[1]), lengths_[2]);
    }

    // Friend comparison operator
    friend  bool operator==(const Box& a, const Box& b){
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] &&
           a.limits[2] == b.limits[2] && a.limits[3] == b.limits[3] &&
           a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5];
    }

private:
    float limits[6];
    float lengths_[3];
    float inverseLengths_[3];
};

/**
 * @brief Integer version of Box
 */
class IBox {
public:
    // Constructors
    HOST_DEVICE  IBox() 
                : limits{0, 0, 0, 0, 0, 0} {} 
    
    HOST_DEVICE  IBox(int xyzMin, int xyzMax)
                : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax} {}
    
    HOST_DEVICE  IBox(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
                : limits{xmin, xmax, ymin, ymax, zmin, zmax} {}

    // Getter methods
    HOST_DEVICE  int xmin() const { return limits[0]; }
    HOST_DEVICE  int xmax() const { return limits[1]; }
    HOST_DEVICE  int ymin() const { return limits[2]; }
    HOST_DEVICE  int ymax() const { return limits[3]; }
    HOST_DEVICE  int zmin() const { return limits[4]; }
    HOST_DEVICE  int zmax() const { return limits[5]; }

    // Extent method
    HOST_DEVICE  int minExtent() const {
        return cal::min(cal::min(xmax() - xmin(), ymax() - ymin()), zmax() - zmin());
    }

    // Friend comparison operators
    friend  bool operator==(const IBox& a, const IBox& b){
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] &&
           a.limits[2] == b.limits[2] && a.limits[3] == b.limits[3] &&
           a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5];
    }

    friend bool operator<(const IBox& a, const IBox& b){
        return cal::tie(a.limits[0], a.limits[1], a.limits[2],
                   a.limits[3], a.limits[4], a.limits[5]) <
                cal::tie(b.limits[0], b.limits[1], b.limits[2],
                   b.limits[3], b.limits[4], b.limits[5]);
    }

private:
    int limits[6];
};

inline cal::tuple<Vec3f, Vec3f> centerAndSizeCPU(const IBox& ibox, const Box& box){
    int maxCoord = 1u << 21;
    // smallest octree cell edge length in unit cube
    float uL = float(1.) / maxCoord;

    float halfUnitLengthX = float(0.5) * uL * box.lx();
    float halfUnitLengthY = float(0.5) * uL * box.ly();
    float halfUnitLengthZ = float(0.5) * uL * box.lz();
    Vec3f boxCenter = {box.xmin() + (ibox.xmax() + ibox.xmin()) * halfUnitLengthX,
                        box.ymin() + (ibox.ymax() + ibox.ymin()) * halfUnitLengthY,
                        box.zmin() + (ibox.zmax() + ibox.zmin()) * halfUnitLengthZ};
    Vec3f boxSize   = {(ibox.xmax() - ibox.xmin()) * halfUnitLengthX, (ibox.ymax() - ibox.ymin()) * halfUnitLengthY,
                        (ibox.zmax() - ibox.zmin()) * halfUnitLengthZ}; 
    return cal::tuple<Vec3f, Vec3f>{boxCenter, boxSize};
}

DEVICE inline thrust::tuple<Vec3f, Vec3f> centerAndSizeGPU(IBox* ibox, Box* box){
    int maxCoord = 1u << 21;
    // smallest octree cell edge length in unit cube
    float uL = float(1.) / maxCoord;

    float halfUnitLengthX = float(0.5) * uL * box->lx();
    float halfUnitLengthY = float(0.5) * uL * box->ly();
    float halfUnitLengthZ = float(0.5) * uL * box->lz();
    Vec3f boxCenter = {box->xmin() + (ibox->xmax() + ibox->xmin()) * halfUnitLengthX,
                        box->ymin() + (ibox->ymax() + ibox->ymin()) * halfUnitLengthY,
                        box->zmin() + (ibox->zmax() + ibox->zmin()) * halfUnitLengthZ};
    Vec3f boxSize   = {(ibox->xmax() - ibox->xmin()) * halfUnitLengthX, (ibox->ymax() - ibox->ymin()) * halfUnitLengthY,
                        (ibox->zmax() - ibox->zmin()) * halfUnitLengthZ}; 
    return thrust::tuple<Vec3f, Vec3f>{boxCenter, boxSize};
}

    
}
