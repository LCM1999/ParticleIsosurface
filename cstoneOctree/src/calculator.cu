#include <calculator.h>

namespace cal{

//! @brief returns the smallest distance vector of point X to box b, 0 if X is in b
// HOST_DEVICE Vec3f minDistanceGPU(const Vec3f& X, const Vec3f& bCenter, const Vec3f& bSize){
//     Vec3f dX = {fabsf(bCenter.x - X.x), fabsf(bCenter.y - X.y), fabsf(bCenter.z - X.z)};
//     dX -= bSize;
//     dX = dX + Vec3f{fabsf(dX.x), fabsf(dX.y), fabsf(dX.z)};
//     dX *= float(0.5);
//     return dX;
// }


}