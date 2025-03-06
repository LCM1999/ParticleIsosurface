#include <box.h>

namespace cstoneOctree{

// DEVICE thrust::tuple<Vec3f, Vec3f> centerAndSizeGPU(IBox* ibox, Box* box){
//     int maxCoord = 1u << 21;
//     // smallest octree cell edge length in unit cube
//     float uL = float(1.) / maxCoord;

//     float halfUnitLengthX = float(0.5) * uL * box->lx();
//     float halfUnitLengthY = float(0.5) * uL * box->ly();
//     float halfUnitLengthZ = float(0.5) * uL * box->lz();
//     Vec3f boxCenter = {box->xmin() + (ibox->xmax() + ibox->xmin()) * halfUnitLengthX,
//                         box->ymin() + (ibox->ymax() + ibox->ymin()) * halfUnitLengthY,
//                         box->zmin() + (ibox->zmax() + ibox->zmin()) * halfUnitLengthZ};
//     Vec3f boxSize   = {(ibox->xmax() - ibox->xmin()) * halfUnitLengthX, (ibox->ymax() - ibox->ymin()) * halfUnitLengthY,
//                         (ibox->zmax() - ibox->zmin()) * halfUnitLengthZ}; 
//     return thrust::tuple<Vec3f, Vec3f>{boxCenter, boxSize};
// }

}