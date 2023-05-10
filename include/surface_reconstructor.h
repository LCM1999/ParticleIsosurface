#pragma once
#ifndef SURF_CONSTRUCT_H
#define SURF_CONSTRUCT_H

#include <vector>
#include <string.h>
#include <queue>
#include <atomic>
#include "timer.h"
#include "utils.h"

class HashGrid;
class MultiLevelSearcher;
class Evaluator;
class TNode;
class Mesh;

class SurfReconstructor
{
private:
    // Global Parameters
    int _OVERSAMPLE_QEF = 2;
    float _BORDER = (1.0 / 16.0);
    int _DEPTH_MAX = 6; // 7
    int _DEPTH_MIN = 5; // 4
    float _INFLUENCE_FACTOR = 0.0f;
    float _ISO_VALUE = 0.0f;

    float _MAX_SCALAR = -1.0f;
    float _MIN_SCALAR = 0.0f;
    float _BAD_QEF = 0.0001f;
    float _FLATNESS = 0.99f;
    int _MIN_NEIGHBORS_NUM = 25;
    int _SPARSE_NEIGHBORS_NUM = 16;

    bool _USE_ANI = true;
    bool _USE_XMEAN = true;
    float _XMEAN_DELTA = 0.0f;

    float _TOLERANCE = 1e-8;
    float _RATIO_TOLERANCE = 0.1f;
    //float _LOW_MESH_QUALITY = 0.17320508075688772935274463415059;

    HashGrid* _hashgrid = NULL;
    MultiLevelSearcher* _searcher = NULL;
    Evaluator* _evaluator = NULL;

    std::vector<Eigen::Vector3f> _GlobalParticles;
    std::vector<float>* _GlobalRadiuses;
    
    float _RADIUS = 0;

    int _GlobalParticlesNum = 0;
    int _STATE = 0;

    static const int inProcessSize = 10000000; //

    float _BoundingBox[6] = {0.0f};
    float _RootHalfLength;
    float _RootCenter[3] = {0.0f};

    TNode* _OurRoot;
	Mesh* _OurMesh;

    std::vector<TNode*> WaitingStack;
    TNode* ProcessArray[inProcessSize];

    std::atomic<int> queue_flag;
protected:
    void loadRootBox();

    void generalModeRun();

    void shrinkBox();

    void resizeRootBoxConstR();

    void resizeRootBoxVarR();

    // Method for CSV mode
    void genIsoOurs();
    void checkEmptyAndCalcCurv(TNode* tnode, bool& empty, float& curv, float& min_radius);
    void eval(TNode* tnode);
    void beforeSampleEval(TNode* tnode, float& curv, float& min_radius, bool& empty, int& oversample);
    void afterSampleEval(
        TNode* tnode, float& curv, float& min_radius, int* oversample, const int index,
        float* sample_points, float* sample_grads);

public:
    SurfReconstructor() {};
    SurfReconstructor(
        std::vector<Eigen::Vector3f>& particles,
        std::vector<float>* radiuses, 
        Mesh* mesh, 
        float radius, 
        float flatness,
        float inf_factor);

    ~SurfReconstructor() {};

    void Run();

    inline int getOverSampleQEF() {return _OVERSAMPLE_QEF;}
    inline float getBorder() {return _BORDER;}
    inline int getDepthMax() {return _DEPTH_MAX;}
    inline int getDepthMin() {return _DEPTH_MIN;}
    inline float getInfluenceFactor() {return _INFLUENCE_FACTOR;}
    inline float getIsoValue() {return _ISO_VALUE;}
    inline float getMaxScalar() {return _MAX_SCALAR;}
    inline float getMinScalar() {return _MIN_SCALAR;}
    inline float getBadQef() {return _BAD_QEF;}
    inline float getFlatness() {return _FLATNESS;}
    inline int getMinNeighborsNum() {return _MIN_NEIGHBORS_NUM;}
    inline int getSparseNeighborsNum() {return _SPARSE_NEIGHBORS_NUM;};
    inline bool getUseAni() {return _USE_ANI;}
    inline bool getUseXMean() {return _USE_XMEAN;}
    inline float getXMeanDelta() {return _XMEAN_DELTA;}
    inline float getTolerance() {return _TOLERANCE;}
    inline float getRatioTolerance() {return _RATIO_TOLERANCE;}
    inline HashGrid* getHashGrid() {return _hashgrid;}
    inline MultiLevelSearcher* getSearcher() {return _searcher;}
    inline Evaluator* getEvaluator() {return _evaluator;}
    inline std::vector<Eigen::Vector3f>* getGlobalParticles() {return &_GlobalParticles;}
    inline int getGlobalParticlesNum() {return _GlobalParticlesNum;}
    inline std::vector<float>* getRadiuses() {return _GlobalRadiuses;}
    inline float getConstRadius() {return _RADIUS;}
    inline int getSTATE() {return _STATE;}
    inline TNode* getRoot() {return _OurRoot;}
};

#endif
