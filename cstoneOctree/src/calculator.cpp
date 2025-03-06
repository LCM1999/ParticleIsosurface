#include <calculator.h>

namespace cal{

//! @brief returns the smallest distance vector of point X to box b, 0 if X is in b
// Vec3f minDistanceCPU(const Vec3f& X, const Vec3f& bCenter, const Vec3f& bSize){
//     Vec3f dX = {std::abs(bCenter.x - X.x), std::abs(bCenter.y - X.y), std::abs(bCenter.z - X.z)};
//     dX -= bSize;
//     dX = dX + Vec3f{std::abs(dX.x), std::abs(dX.y), std::abs(dX.z)};
//     dX *= float(0.5);
//     return dX;
// }


}