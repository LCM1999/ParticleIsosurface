#pragma once

#include <cuda_def.h>
#include <coord_struct.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <thrust/tuple.h>
#else
#include <tuple>
#endif

using namespace cstoneOctree;
namespace cal{

template <class T>
HOST_DEVICE const T& min(const T& a, const T& b)
{
    if (b < a) return b;
    return a;
}

template <class T>
HOST_DEVICE const T& max(const T& a, const T& b)
{
    if (a < b) return b;
    return a;
}

/* a simplified version of std::lower_bound that can be compiled both on host and device */
template <class ForwardIt, class T>
HOST_DEVICE ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
{
    ForwardIt it;
    long long int step;
    long long int count = last - first;

    while (count > 0)
    {
        it   = first;
        step = count / 2;
        it += step;
        if (*it < value)
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

/* a simplified version of std::upper_bound that can be compiled both on host and device */
template <class ForwardIt, class T>
HOST_DEVICE inline ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value)
{
    ForwardIt it;
    long long int step;
    long long int count = last - first;

    while (count > 0)
    {
        it   = first;
        step = count / 2;
        it += step;
        if (!(value < *it)) // NOLINT
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

template <typename Iterator1, typename Iterator2>
void sort_by_key_cpu(Iterator1 key_first, Iterator1 key_last, Iterator2 value_first) {
    using KeyType = typename std::iterator_traits<Iterator1>::value_type;
    using ValueType = typename std::iterator_traits<Iterator2>::value_type;

    auto n = std::distance(key_first, key_last);

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
        return key_first[i] < key_first[j];
    });

    std::vector<KeyType> sorted_keys(n);
    std::vector<ValueType> sorted_values(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_keys[i] = key_first[idx[i]];
        sorted_values[i] = value_first[idx[i]];
    }

    for (size_t i = 0; i < n; ++i) {
        key_first[i] = sorted_keys[i];
        value_first[i] = sorted_values[i];
    }
}

template <class T, class U>
void fill_data_cpu(T* start, int n, U val) {
    T converted_val = static_cast<T>(val); 
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        start[i] = converted_val;
    }
}

HOST_DEVICE inline Vec3f abs_diff(const Vec3f& a, const Vec3f& b) {
#ifdef __CUDACC__
    return Vec3f( fabsf(a.x - b.x), fabsf(a.y - b.y), fabsf(a.z - b.z) );
#else
    return Vec3f( std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.z - b.z) );
#endif
}

// calculate the minimum distance from the point X to the chosen box
inline Vec3f minDistanceCPU(const Vec3f& X, const Vec3f& bCenter, const Vec3f& bSize){
    Vec3f dX = {std::abs(bCenter.x - X.x), std::abs(bCenter.y - X.y), std::abs(bCenter.z - X.z)};
    dX -= bSize;
    dX = dX + Vec3f{std::abs(dX.x), std::abs(dX.y), std::abs(dX.z)};
    dX *= float(0.5);
    return dX;
}

// calculate the minimum distance from the point X to the chosen box
HOST_DEVICE inline Vec3f minDistanceGPU(const Vec3f& X, const Vec3f& bCenter, const Vec3f& bSize){
    Vec3f dX = {fabsf(bCenter.x - X.x), fabsf(bCenter.y - X.y), fabsf(bCenter.z - X.z)};
    dX =  dX - bSize;
    dX = dX + Vec3f{fabsf(dX.x), fabsf(dX.y), fabsf(dX.z)};
    dX = dX * float(0.5);
    return dX;
}

// ! @brief returns the smallest distance vector of point X to box b, 0 if X is in b
// HOST_DEVICE Vec3f minDistance(const Vec3f& X, const Vec3f& bCenter, const Vec3f& bSize){

// #ifdef __CUDACC__
//     Vec3f dX = {fabsf(bCenter.x - X.x), fabsf(bCenter.y - X.y), fabsf(bCenter.z - X.z)};
//     dX -= bSize;
//     dX = dX + Vec3f{fabsf(dX.x), fabsf(dX.y), fabsf(dX.z)};
//     // Vec3f dX = fabsf(bCenter - X) - bSize;
//     // dX += fabsf(dX);
// #else
//     Vec3f dX = {std::abs(bCenter.x - X.x), std::abs(bCenter.y - X.y), std::abs(bCenter.z - X.z)};
//     dX -= bSize;
//     dX = dX + Vec3f{std::abs(dX.x), std::abs(dX.y), std::abs(dX.z)};
// #endif
//     dX *= float(0.5);
//     return dX;
// }

#ifdef __CUDACC__

template<class... Ts>
using tuple = thrust::tuple<Ts...>;

template<size_t N, class T>
HOST_DEVICE auto get(T&& tup) noexcept
{
    return thrust::get<N>(tup);
}

template<class... Ts>
HOST_DEVICE tuple<Ts&...> tie(Ts&... args) noexcept
{
    return thrust::tuple<Ts&...>(args...);
}


#else

template<class... Ts>
using tuple = std::tuple<Ts...>;

template<std::size_t N, class T>
constexpr auto get(T&& tup) noexcept
{
    return std::get<N>(tup);
}

template<class... Ts>
constexpr tuple<Ts&...> tie(Ts&... args) noexcept
{
    return std::tuple<Ts&...>(args...);
}

#endif

}