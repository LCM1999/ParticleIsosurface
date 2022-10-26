#pragma once
#include <string>

extern int OVERSAMPLE_QEF;
extern float BORDER;
extern int DEPTH_MAX;
extern int DEPTH_MIN;

extern int FIND_ROOT_DEPTH;

class HashGrid;
extern HashGrid* hashgrid;

class Evaluator;
extern Evaluator* evaluator;

extern float P_RADIUS;
extern float ISO_VALUE;
extern float INFLUENCE;

extern int KERNEL_TYPE;

extern float TOLERANCE;
extern float RATIO_TOLERANCE;
extern float MESH_TOLERANCE;
extern float LOW_MESH_QUALITY;
extern float VARIANCE_TOLERANCE;

extern float MAX_VALUE;
extern float MIN_VALUE;
extern float BAD_QEF;
extern float FLATNESS;
extern int MIN_NEIGHBORS_NUM;
extern bool USE_ANI;
extern bool USE_XMEAN;
extern float XMEAN_DELTA;

extern int USE_DYNAMIC_THREADS;
extern int THREAD_NUMS;

extern std::string CASE_PATH;
extern bool LOAD_RECORD;
extern bool NEED_RECORD;
extern int RECORD_STEP;
extern std::string OUTPUT_PREFIX;
extern std::string RECORD_PREFIX;

extern int GlobalParticlesNum;
extern double BoundingBox[6];
extern double RootHalfLength;

extern float RootCenter[3];

extern int INDEX;
extern int STATE;


//#define JOIN_VERTS
//#define USE_DMT
#define USE_DMC
#define USE_DMC_ERR
