#pragma once
#include <cuda_def.h>

namespace iso{

template <class T>
inline int sign(T x){
    return (x > 0) ? 1 : -1;
}

}