#pragma once
#include <string>
#include <vector>


inline int OMP_USE_DYNAMIC_THREADS = 0;
inline int OMP_THREADS_NUM = 16;

inline bool IS_CONST_RADIUS = false;
inline bool USE_ANI = true;

// variants for test
inline bool NEED_RECORD = false;
inline int TARGET_FRAME = 0;
// std::string PREFIX = "";
inline std::string SUFFIX = "";    // CSV, H5
inline bool WITH_NORMAL;
inline std::vector<std::string> DATA_PATHES;
inline std::string OUTPUT_TYPE = "ply";
inline float RADIUS = 0;
inline float SMOOTH_FACTOR = 2.0;
inline float ISO_FACTOR = 1.9;
inline float ISO_VALUE = 0.0f;
//inline bool USE_CUDA = false;
inline bool CALC_P_NORMAL = true;
inline bool GEN_SPLASH = true;
inline bool SINGLE_LAYER = false;
inline bool USE_OURS = true;
inline bool USE_POLY6 = 0;